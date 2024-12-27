from pyspark.sql import functions as F
from pyspark.ml.evaluation import RegressionEvaluator

class ModelEvaluation:
    def __init__(self, df, result_col, prediction_col, model_type="classification"):
        """
        Initializes the ModelEvaluation class with the DataFrame, column names, and model type.

        :param df: PySpark DataFrame containing the results and predictions.
        :param result_col: Column name for the actual results.
        :param prediction_col: Column name for the predicted results.
        :param model_type: The type of model ("classification" or "regression").
        """
        self.df = df
        self.result_col = result_col
        self.prediction_col = prediction_col
        self.model_type = model_type

        # Calculate metrics based on the model type
        if self.model_type == "classification":
            self.metrics = self.calculate_classification_metrics()
        elif self.model_type == "regression":
            self.metrics = self.calculate_regression_metrics()
        
    def calculate_classification_metrics(self):
        """
        Calculates TP, TN, FP, FN based on the actual results and predicted results.
        
        :return: A dictionary with the values of TP, TN, FP, FN.
        """
        # True Positive: actual = 1 and predicted = 1
        tp = self.df.filter((F.col(self.result_col) == 1) & (F.col(self.prediction_col) == 1)).count()
        
        # True Negative: actual = 0 and predicted = 0
        tn = self.df.filter((F.col(self.result_col) == 0) & (F.col(self.prediction_col) == 0)).count()
        
        # False Positive: actual = 0 and predicted = 1
        fp = self.df.filter((F.col(self.result_col) == 0) & (F.col(self.prediction_col) == 1)).count()
        
        # False Negative: actual = 1 and predicted = 0
        fn = self.df.filter((F.col(self.result_col) == 1) & (F.col(self.prediction_col) == 0)).count()
        
        return {
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn
        }
     
    def calculate_regression_metrics(self):
        """
        Calculates regression metrics such as MSE, RMSE, MAE, and R2.
        
        :return: A dictionary with regression metrics.
        """
        
        evaluator_mse = RegressionEvaluator(labelCol=self.result_col, predictionCol=self.prediction_col, metricName="mse")
        evaluator_rmse = RegressionEvaluator(labelCol=self.result_col, predictionCol=self.prediction_col, metricName="rmse")
        evaluator_mae = RegressionEvaluator(labelCol=self.result_col, predictionCol=self.prediction_col, metricName="mae")
        evaluator_r2 = RegressionEvaluator(labelCol=self.result_col, predictionCol=self.prediction_col, metricName="r2")
        
        mse = evaluator_mse.evaluate(self.df)
        rmse = evaluator_rmse.evaluate(self.df)
        mae = evaluator_mae.evaluate(self.df)
        r2 = evaluator_r2.evaluate(self.df)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    
    def accuracy(self):
        """Calculates accuracy (classification only)."""
        if self.model_type != "classification":
            raise ValueError("Accuracy is only applicable for classification models.")
        tp, tn, fp, fn = self.metrics.values()
        return (tp + tn) / (tp + tn + fp + fn)
    
    def precision(self):
        """Calculates precision (classification only)."""
        if self.model_type != "classification":
            raise ValueError("Precision is only applicable for classification models.")
        tp, tn, fp, fn = self.metrics.values()
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    def recall(self):
        """Calculates recall (classification only)."""
        if self.model_type != "classification":
            raise ValueError("Recall is only applicable for classification models.")
        tp, tn, fp, fn = self.metrics.values()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def sensitivity(self):
        """Calculates sensitivity (classification only)."""
        return self.recall()
    
    def specificity(self):
        """Calculates specificity (classification only)."""
        if self.model_type != "classification":
            raise ValueError("Specificity is only applicable for classification models.")
        tp, tn, fp, fn = self.metrics.values()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def f1(self):
        """Calculates F1 score (classification only)."""
        if self.model_type != "classification":
            raise ValueError("F1 score is only applicable for classification models.")
        p = self.precision()
        r = self.recall()
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
    
    def fpr(self):
        """Calculates False Positive Rate (classification only)."""
        if self.model_type != "classification":
            raise ValueError("False Positive Rate (FPR) is only applicable for classification models.")
        tp, tn, fp, fn = self.metrics.values()
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    def fnr(self):
        """Calculates False Negative Rate (classification only)."""
        if self.model_type != "classification":
            raise ValueError("False Negative Rate (FNR) is only applicable for classification models.")
        tp, tn, fp, fn = self.metrics.values()
        return fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    def get_all_metrics(self):
        """Returns all metrics in a dictionary."""
        if self.model_type == "classification":
            return {
                'Accuracy': self.accuracy(),
                'Precision': self.precision(),
                'Recall': self.recall(),
                'Sensitivity': self.sensitivity(),
                'Specificity': self.specificity(),
                'F1': self.f1(),
                'FPR': self.fpr(),
                'FNR': self.fnr()
            }
        elif self.model_type == "regression":
            return self.metrics