from pyspark.sql.functions import col, regexp_extract,round, sqrt, pow, lit,udf,when
from pyspark.sql.types import FloatType
import pandas as pd
import ast, math


######### Function to export shot data #########
def shot_data(df):
    # Only shots
    # 5th period is penalties
    # set shot_freeze_frame to null when it's penalty
    
    columns = []
    df = df.filter((df.type=='Shot') & (df.period < 5)).select(columns)
    return df

######### Function to export pass data #########

######### Function to split the location column into x and y coordinates #########
def split_location(df):
    df_l = df.withColumn("shot_location_x", regexp_extract(col("location"), r'\[(.*?),', 1).cast("float")) \
             .withColumn("shot_location_y", regexp_extract(col("location"), r', (.*?)\]', 1).cast("float")).drop('location')
    return df_l

######### Function to calculate the distance to the goal #########
def distance_to_goal(df):
    goal_x, goal_y = 120, 40
    df = df.withColumn("distance_to_goal",
                       round(sqrt(pow(df.shot_location_x - lit(goal_x), 2) + pow(df.shot_location_y - lit(goal_y), 2)),4))
    return df

######### Function to calculate the angle to the goal #########
def calculate_shot_angle(shot_x, shot_y):
    goal_x1, goal_y1 = 120, 36  # Left post
    goal_x2, goal_y2 = 120, 44  # Right post
    # Vectors to the posts
    u_x, u_y = goal_x1 - shot_x, goal_y1 - shot_y
    v_x, v_y = goal_x2 - shot_x, goal_y2 - shot_y
    
    # Dot product and magnitudes
    dot_product = u_x * v_x + u_y * v_y
    magnitude_u = math.sqrt(u_x**2 + u_y**2)
    magnitude_v = math.sqrt(v_x**2 + v_y**2)
    
    # Avoid division by zero
    if magnitude_u == 0 or magnitude_v == 0:
        return 0.0
    
    # Calculate angle in radians
    angle_radians = math.acos(dot_product / (magnitude_u * magnitude_v))
    
    # Convert to degrees
    return math.degrees(angle_radians)

def get_shot_angle(df):
    # Register the function as a UDF
    calculate_shot_angle_udf = udf(calculate_shot_angle, FloatType())
    
    # Apply the UDF to the DataFrame
    df = df.withColumn(
        "shot_angle",
        calculate_shot_angle_udf(df["shot_location_x"], df["shot_location_y"]))
    return df

######### Function to get the favorite foot of the shooter #########
def preferred_foot(df):
    # Filter Pass and Shot df, selecting relevant columns
    pass_bp = df.filter(df.type == 'Pass')\
        .select('player_id', col('pass_body_part').alias('body_part'))\
        .filter(col('body_part').isin('Right Foot', 'Left Foot'))
    
    shot_bp = df.filter(df.type == 'Shot')\
        .select('player_id', col('shot_body_part').alias('body_part'))\
        .filter(col('body_part').isin('Right Foot', 'Left Foot'))

    # Union the two DataFrames
    bp = pass_bp.union(shot_bp)

    # Map 'body_part' into separate columns for left and right foot counts
    bp_mapped = bp.withColumn('left_foot', (col('body_part') == 'Left Foot').cast('int'))\
        .withColumn('right_foot', (col('body_part') == 'Right Foot').cast('int'))\
        .drop('body_part')

    # Group by player and calculate sums for left and right foot counts
    foot_counts = bp_mapped.groupBy('player_id')\
        .sum('left_foot', 'right_foot')\
        .withColumnRenamed('sum(left_foot)', 'left_foot')\
        .withColumnRenamed('sum(right_foot)', 'right_foot')

    # Add total actions column
    foot_counts = foot_counts.withColumn("total_actions", col("left_foot") + col("right_foot"))

    # Determine preferred foot based on the 66% rule
    foot_counts = foot_counts.withColumn(
        "preferred_foot",
        when((col("left_foot") / col("total_actions")) >= 0.66, "Left Foot")
        .when((col("right_foot") / col("total_actions")) >= 0.66, "Right Foot")
        .otherwise("Two-Footed"))

    return foot_counts.drop('left_foot', 'right_foot', 'total_actions')

######### Function to check if the shot was taken with the preferred foot #########
def shot_preferred_foot(df,dff):
    dff = preferred_foot(dff)
    df = df.join(dff, df.player_id == dff.player_id, how='left').drop(dff.player_id)
    df = df.withColumn('preferred_foot_shot', col('preferred_foot') == col('shot_body_part'))
    return df

######### Function to create goal column #########
def goal(df):
    df = df.withColumn('goal', col('shot_outcome') == 'Goal')
    return df

######### Function to convert boolean columns to integer #########
def bool_to_int(df, columns):
    for col_name in columns:
        df = df.withColumn(col_name, when(col(col_name).isNull(), 0).otherwise(col(col_name).cast('int')))
    return df.fillna(0)

######### Function to export shot_freeze_frame dataframe [id, shot_freeze_frame] #########
def shot_freeze_frame(df):
    df = df.select('id', 'shot_freeze_frame').dropna(subset=['shot_freeze_frame'])
    return df

######### Function to export shot_freeze_frame to a separate dataframe #########
def shot_frame_to_df(df):
    processed_rows = []
    for row in df.collect():
        id = row.id
        shot_frame = row.shot_freeze_frame.split('}, {')
        for i in range(len(shot_frame)):
            x, y = ast.literal_eval(shot_frame[i].split('},')[0].split(': ')[1].split('],')[0] + ']')
            position = shot_frame[i].split('},')[1].split(': ')[3].strip("''")
            teammate = 'True' if 'True' in shot_frame[i].split('},')[2] else 'False'
            processed_rows.append({'Shot_id': id, 'x': x, 'y': y, 'position': position, 'teammate': teammate})
    
    df = pd.DataFrame(processed_rows)
    return df

# Function to calculate the number of defenders between the shooter and the goal

# Function to calculate the number of players inside the area of shooting

# Function to get the position that the shooter plays

# Function to add a multiplier according to the position of the shooter

# Function to get pass information

# Functions to create visualizations