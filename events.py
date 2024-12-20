from pyspark.sql.functions import col, regexp_extract, when,round
from pyspark.sql.functions import sqrt, pow, lit

# Function to split the location column into x and y coordinates
def split_location(df):
    df_l = df.withColumn("shot_location_x", regexp_extract(col("location"), r'\[(.*?),', 1).cast("float")) \
             .withColumn("shot_location_y", regexp_extract(col("location"), r', (.*?)\]', 1).cast("float")).drop('location')
    return df_l

# Function to calculate the distance to the goal
def distance_to_goal(df):
    goal_x = 120
    goal_y = 40
    df = df.withColumn("distance_to_goal", round(sqrt(pow(df.shot_location_x - lit(goal_x), 2) + pow(df.shot_location_y - lit(goal_y), 2)),4))
    return df

# Function to export shot data

# Function to export pass data

# Function to calculate the angle to the goal

# Function to export shot_freeze_frame to a separate dataframe

# Function to calculate the number of defenders between the shooter and the goal

# Function to calculate the number of players inside the area of shooting

# Function to get the favorite foot of the shooter

# Function to get the position that the shooter plays

# Function to add a multiplier according to the position of the shooter

# Function to get pass information