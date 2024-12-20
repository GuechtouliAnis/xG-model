from pyspark.sql.functions import col, regexp_extract, when,round
from pyspark.sql.functions import sqrt, pow, lit

def split_location(df):
    df_l = df.withColumn("shot_location_x", regexp_extract(col("location"), r'\[(.*?),', 1).cast("float")) \
             .withColumn("shot_location_y", regexp_extract(col("location"), r', (.*?)\]', 1).cast("float")).drop('location')
    return df_l

def distance_to_goal(df):
    goal_x = 120
    goal_y = 40
    df = df.withColumn("distance_to_goal", round(sqrt(pow(df.shot_location_x - lit(goal_x), 2) + pow(df.shot_location_y - lit(goal_y), 2)),4))
    return df

