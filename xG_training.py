import matplotlib.pyplot as plt
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, MultilayerPerceptronClassifier, GBTClassifier, NaiveBayes, DecisionTreeClassifier, LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator

MODELS = ['logistic', 'rf', 'mlp', 'gbt', 'nb', 'dt', 'svm']

class ModelTrainer:
    def __init__(self, model_type, train_data, test_data, label_col="goal", features_col="features_vector", layers=None, num_trees=None, max_iter=100):
        """
        Initializes and trains a model, and calculates evaluation metrics.

        :param model_type: Type of model to use (str). Options: 'logistic', 'rf', 'mlp', 'gbt', 'nb', 'dt', 'svm'.
        :param train_data: The training dataset.
        :param test_data: The testing dataset.
        :param label_col: The name of the label column.
        :param features_col: The name of the features column.
        :param layers: Layers for MLP (only needed if using 'mlp').
        :param num_trees: Number of trees for RandomForest (only needed if using 'rf').
        :param max_iter: Maximum iterations (default: 10). Used for models that support maxIter like LogisticRegression.
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
        self.predictions = self.get_predictions()

        # Evaluate the model
        self.roc_auc = self.evaluate_model()

    def initialize_model(self):
        """Initialize the model based on model_type."""
        if self.model_type == 'logistic':
            model = LogisticRegression(featuresCol=self.features_col, labelCol=self.label_col, maxIter=self.max_iter)
        elif self.model_type == 'rf':
            model = RandomForestClassifier(featuresCol=self.features_col, labelCol=self.label_col, numTrees=self.num_trees or 100)
        elif self.model_type == 'mlp':
            model = MultilayerPerceptronClassifier(featuresCol=self.features_col, labelCol=self.label_col, maxIter=self.max_iter, layers=self.layers, blockSize=128, seed=1234)
        elif self.model_type == 'gbt':
            model = GBTClassifier(featuresCol=self.features_col, labelCol=self.label_col, maxIter=self.max_iter)
        elif self.model_type == 'nb':
            model = NaiveBayes(featuresCol=self.features_col, labelCol=self.label_col, modelType="multinomial")
        elif self.model_type == 'dt':
            model = DecisionTreeClassifier(featuresCol=self.features_col, labelCol=self.label_col, maxDepth=5)
        elif self.model_type == 'svm':
            model = LinearSVC(featuresCol=self.features_col, labelCol=self.label_col)
        else:
            raise ValueError("Unknown model type. Choose from ['logistic', 'rf', 'mlp', 'gbt', 'nb', 'dt', 'svm']")
        return model

    def train_model(self):
        """Train the model on the training data."""
        model_fitted = self.model.fit(self.train_data)
        return model_fitted

    def get_predictions(self):
        """Make predictions on the test data."""
        predictions = self.model_trained.transform(self.test_data)
        return predictions

    def evaluate_model(self):
        """Evaluate the model using ROC-AUC."""
        evaluator = BinaryClassificationEvaluator(labelCol=self.label_col, rawPredictionCol="rawPrediction")
        roc_auc = evaluator.evaluate(self.predictions)
        return roc_auc

    def plot_learning_curve(self, max_iter_range=None):
        """
        Plots the learning curve for the provided model.
        """
        if max_iter_range is None:
            max_iter_range = range(1, 11)  # Default range for maxIter (1 to 10)

        roc_auc_list = []
        
        for max_iter in max_iter_range:
            # Set the maxIter for iterative models (like LogisticRegression)

            if hasattr(self.model, 'setMaxIter'):
                self.model.setMaxIter(max_iter)
            self.model_trained = self.train_model()
            self.predictions = self.get_predictions()
            roc_auc = self.evaluate_model()
            
            # Convert ROC-AUC to percentage and print it
            roc_auc_percentage = roc_auc * 100
            print(f"ROC-AUC for max_iter={max_iter}: {roc_auc_percentage:.2f}%")
            
            roc_auc_list.append(roc_auc_percentage)

        # Plot the ROC-AUC learning curve
        plt.plot(max_iter_range, roc_auc_list, marker='o')
        plt.xlabel('Number of Iterations')
        plt.ylabel('ROC-AUC (%)')
        plt.title('Learning Curve for Model (ROC-AUC vs Iterations)')
        plt.grid(True)
        plt.show()