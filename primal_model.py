# doing an xg model based on the features extracted using MLlib
# shot_statsbomb_xg is the target variable, the rest are the features
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor, DecisionTreeRegressor, GeneralizedLinearRegression, IsotonicRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import pp_events as pp
from pyspark.sql import SparkSession


# xgboost model
from xgboost import XGBRegressor
# XGBoost, LightGBM, CatBoost
#from pyspark.ml.classification import MultilayerPerceptronClassifier
#from pyspark.ml.evaluation import MulticlassClassificationEvaluator



spark = SparkSession.builder.appName('export_sff').getOrCreate()
events = spark.read.csv('Data/events.csv',header=True,inferSchema=True,sep=';')

events_shot = events.filter(events.type=='Shot')

events_shot = pp.split_location(events_shot)
events_shot = pp.distance_to_goal(events_shot)
events_shot = pp.get_shot_angle(events_shot)


features = ["shot_location_x", "shot_location_y", "distance_to_goal", "shot_angle"]
assembler = VectorAssembler(inputCols=features, outputCol="features")
data = assembler.transform(events_shot).select("features", "shot_statsbomb_xg")

# Split into training and testing sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Initialize all models
lr = LinearRegression(labelCol="shot_statsbomb_xg", featuresCol="features")
rf = RandomForestRegressor(labelCol="shot_statsbomb_xg", featuresCol="features")
gbt = GBTRegressor(labelCol="shot_statsbomb_xg", featuresCol="features")
dt = DecisionTreeRegressor(labelCol="shot_statsbomb_xg", featuresCol="features")
glr = GeneralizedLinearRegression(labelCol="shot_statsbomb_xg", featuresCol="features")
ir = IsotonicRegression(labelCol="shot_statsbomb_xg", featuresCol="features")

# Create a list of models
models = [lr, rf, gbt, dt, glr, ir]

# Create a list of model names
model_names = ["Linear Regression", "Random Forest", "Gradient Boosted Trees", "Decision Tree", "Generalized Linear Regression", "Isotonic Regression"]

# Create a list of model pipelines
model_pipelines = [Pipeline(stages=[model]) for model in models]

# Fit each model pipeline
fitted_models = [pipeline.fit(train_data) for pipeline in model_pipelines]

# Evaluate each model
evaluator = RegressionEvaluator(labelCol="shot_statsbomb_xg", predictionCol="prediction", metricName="rmse")
predictions = [fitted_model.transform(test_data) for fitted_model in fitted_models]
rmse = [evaluator.evaluate(prediction) for prediction in predictions]

# Display the RMSE of each model
for i in range(len(models)):
    print(model_names[i] + " RMSE: " + str(rmse[i]))
    
# Time spent to run : 1m 54s for ["shot_location_x", "shot_location_y", "distance_to_goal", "shot_angle"] 0.09
# Time spent to run : 1m 50s for ["distance_to_goal", "shot_angle"]                                       0.10

# Consider using Classification instead of Regression