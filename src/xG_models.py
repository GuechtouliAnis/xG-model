from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, MultilayerPerceptronClassifier
from pyspark.ml.classification import GBTClassifier, DecisionTreeClassifier, LinearSVC
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pandas as pd
from .xG_constants import *

class ModelTrainer:
    def __init__(self,
                 train_data : DataFrame,
                 test_data : DataFrame,
                 model_type : str = 'logistic',
                 label_col : str = "goal",
                 features_col : str = "features_vector",
                 layers : list[int] | None = None,
                 num_trees : int | None = None,
                 max_iter : int = 100):
        
        """
        Initializes and trains a model, and calculates evaluation metrics.

        :param model_type: Type of model to use (str). Options: 'logistic', 'rf', 'mlp', 'gbt', 'dt'.
        :param train_data: The training dataset.
        :param test_data: The testing dataset.
        :param label_col: The name of the label column.
        :param features_col: The name of the features column.
        :param layers: Layers for MLP (only needed if using 'mlp').
        :param num_trees: Number of trees for RandomForest (only needed if using 'rf').
        :param max_iter: Maximum iterations (default: 100). Used for models that support maxIter like LogisticRegression.
        """
        
        self.train_data = train_data
        self.test_data = test_data
        self.label_col = label_col
        self.features_col = features_col
        self.max_iter = max_iter
        self.model_type = model_type
        self.layers = layers
        self.num_trees = num_trees

        # Initialize the model
        self.model = self.initialize_model()

        # Train the model
        self.model_trained = self.train_model()

        # Get predictions
        self.df = self.get_predictions()
        
        self.goal_proba()

    def initialize_model(self):
        
        """Initialize the model based on model_type."""
        
        if self.model_type == 'logistic':
            model = LogisticRegression(featuresCol=self.features_col,
                                       labelCol=self.label_col,
                                       maxIter=self.max_iter)
        elif self.model_type == 'rf':
            model = RandomForestClassifier(featuresCol=self.features_col,
                                           labelCol=self.label_col,
                                           numTrees=self.num_trees or 100)
        elif self.model_type == 'mlp':
            if not self.layers:
                raise ValueError("The 'layers' parameter must be specified for the Multilayer Perceptron model.")            
            model = MultilayerPerceptronClassifier(featuresCol=self.features_col,
                                                   labelCol=self.label_col,
                                                   maxIter=self.max_iter,
                                                   layers=self.layers,
                                                   blockSize=128,
                                                   seed=1234)
        elif self.model_type == 'gbt':
            model = GBTClassifier(featuresCol=self.features_col,
                                  labelCol=self.label_col,
                                  maxIter=self.max_iter)
        elif self.model_type == 'dt':
            model = DecisionTreeClassifier(featuresCol=self.features_col,
                                           labelCol=self.label_col,
                                           maxDepth=5)
        else:
            raise ValueError("Unknown model type. Choose from ['logistic', 'rf', 'mlp', 'gbt', 'dt']")

        return model

    def train_model(self):
        
        """Train the model on the training data."""
        
        model_fitted = self.model.fit(self.train_data)
        return model_fitted

    def get_predictions(self):
        
        """Make predictions on the test data."""
        
        predictions = self.model_trained.transform(self.test_data)
        return predictions
    
    def get_feature_importance(self):
        
        """
        Retrieves feature importance or coefficients for the trained model.

        :param feature_names: List of feature names (optional). Matches scores to feature names if provided.
        :return: Dictionary mapping features to scores (if feature_names provided), else list of scores.
        """
        
        if hasattr(self.model_trained, "featureImportances"):
            # For tree-based models (e.g., RandomForest, GBT)
            importance = self.model_trained.featureImportances.toArray()
        elif hasattr(self.model_trained, "coefficients"):
            # For linear models (e.g., LogisticRegression)
            importance = self.model_trained.coefficients.toArray()
        else:
            raise AttributeError(f"Feature importance or coefficients are not available for the {self.model_type} model.")

        return importance
    
    def feature_importance(self,
                           feature_names : list[str] = FEATURES) -> pd.DataFrame:

        """
        Converts feature importance to a DataFrame.

        :param feature_names: List of feature names.
        :return: DataFrame with feature names and importance scores.
        """
        
        feature_importance = self.get_feature_importance()
        df = pd.DataFrame(list(zip(feature_names, feature_importance)),
                          columns=['Feature', 'Importance'])
        return df
    
    def goal_proba(self):

        """
        Processes the goal probability column in the given df DataFrame.

        :param df: PySpark DataFrame with a 'probability' column containing lists.
        :return: Updated DataFrame with the 'xG' column as a float.
        """
        
        # Define the function to extract the second element from the probability list
        def extract_xg(probability):
            return float(probability[1])

        # Register the function as a UDF
        extract_xg_udf = F.udf(extract_xg, T.DoubleType())

        # Overwrite the prediction column using the UDF
        # Format the goal_probability to remove scientific notation
        # Convert goal_probability to float
        self.df = self.df.withColumn("xG", extract_xg_udf(F.col("probability"))) \
                         .withColumn("xG", F.format_number(F.col("xG"), 10))\
                         .withColumn("xG", F.col("xG").cast(T.DoubleType()))\
                         .withColumn('prediction', F.col("prediction").cast(T.IntegerType()))
                         