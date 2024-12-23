from pyspark.sql.functions import col, regexp_extract,round, sqrt, pow, lit,udf,when
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, FloatType
import pandas as pd
import numpy as np
import ast, math

######### Shot DataFrame necessary columns #########
EVENTS_COLUMNS = ['id','period','duration','location','player_id','position', # Event info
                'play_pattern','shot_body_part','shot_technique','shot_type',        # Shot info
                'related_events', 'shot_freeze_frame', 'shot_key_pass_id', 'shot_end_location', # Complicated info
                'under_pressure','shot_aerial_won','shot_first_time','shot_one_on_one','shot_open_goal','shot_follows_dribble', # Boolean
                'shot_statsbomb_xg','shot_outcome',# Target
                'pass_body_part','type','shot_freeze_frame'
                ]

PASS_COLUMNS = []

ML_READY_DATA = ['id','player_id','shot_location_x','shot_location_y','distance_to_goal','shot_angle','preferred_foot_shot',
                 'shot_body_part','shot_technique','shot_type','play_pattern',
                 'under_pressure','shot_aerial_won','shot_first_time','shot_one_on_one','shot_open_goal','shot_follows_dribble',
                 'players_inside_area',
                 'shot_statsbomb_xg','shot_outcome','goal']

BOOL_TO_INT_COLUMNS = ['preferred_foot_shot','under_pressure','shot_aerial_won','shot_first_time','shot_one_on_one',
                      'shot_open_goal','shot_follows_dribble','goal']


goal_X = 120
goal_Y1, goal_Y2 = 36, 44

######### Function to export shot data #########
def shot_data(df):
    df = df.filter(df.type=='Shot').select(ML_READY_DATA)
    return df

######### Function to export pass data #########
def pass_data(df):
    pass

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

######### Function to get Spatial Data #########
def spatial_data(df):
    df = split_location(df)
    df = distance_to_goal(df)
    return get_shot_angle(df)

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
def shot_preferred_foot(df):
    dff = preferred_foot(df)
    df = df.join(dff, df.player_id == dff.player_id, how='left').drop(dff.player_id)

    df = df.withColumn(
        'preferred_foot_shot',
        when(
            (col('preferred_foot') == 'Two-Footed') & (col('shot_body_part').isin('Right Foot', 'Left Foot')),
            True).otherwise(col('preferred_foot') == col('shot_body_part')))
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
    df = df.filter(col('shot_type')!='Penalty').select('id', 'shot_freeze_frame').dropna(subset=['shot_freeze_frame'])
    return df

######### Function to export shot_freeze_frame to a separate dataframe #########
def shot_frame_to_df(df):
    df = shot_freeze_frame(df)
    processed_rows = []
    for row in df.collect():
        id = row.id
        shot_frame = row.shot_freeze_frame.split('}, {')
        for i in range(len(shot_frame)):
            x, y = ast.literal_eval(shot_frame[i].split('},')[0].split(': ')[1].split('],')[0] + ']')
            position = shot_frame[i].split('},')[1].split(': ')[3].strip("''")
            teammate = 'True' if 'True' in shot_frame[i].split('},')[2] else 'False'
            processed_rows.append({'Shot_id': id, 'x': x, 'y': y, 'position': position, 'teammate': teammate})
    
    return pd.DataFrame(processed_rows)

######### Function to calculate the number of players inside the area of shooting #########
def calculate_area(x1, y1, x2, y2, x3, y3):
    return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0

def is_point_inside_triangle(player_x,player_y,shot_x,shot_y):
    # Total area of the triangle ABC
    area_abc = calculate_area(shot_x,shot_y,goal_X,goal_Y1,goal_X,goal_Y2)

    # Area of sub-triangles ABP, BCP, CAP
    area_abp = calculate_area(shot_x,shot_y,goal_X,goal_Y1,player_x,player_y)
    area_bcp = calculate_area(shot_x,shot_y,goal_X,goal_Y2,player_x,player_y)
    area_cap = calculate_area(goal_X,goal_Y2,goal_X,goal_Y1,player_x,player_y)

    return np.isclose(area_abc, (area_abp + area_bcp + area_cap))

def check_point_inside(coordinates, shot_x, shot_y):
    count = 0
    for coord in coordinates:
        player_x, player_y = coord
        if is_point_inside_triangle(player_x, player_y, shot_x, shot_y):
            count += 1
    return count

def number_of_players_in_area(df,spark):

    frames = shot_frame_to_df(df)
    frames = (frames.groupby('Shot_id').apply(lambda group: group[['x', 'y']].values.tolist()).reset_index(name='coordinates'))
    frames = spark.createDataFrame(frames)
    frames = frames.join(df.select('id','shot_location_x','shot_location_y'),frames.Shot_id == df.id).drop('id')

    # Register the UDF
    check_point_udf = F.udf(check_point_inside, returnType=IntegerType())  # Use IntegerType from pyspark.sql.types

    # Add a column with the count of points inside the area for each row in the DataFrame
    frames = frames.withColumn("players_inside_area", check_point_udf(F.col("coordinates"), F.col("shot_location_x"), F.col("shot_location_y")))
    df = df.join(frames.select('Shot_id','players_inside_area'),df.id == frames.Shot_id,how='left').drop('Shot_id')
    df = df.withColumn('players_inside_area',F.when(F.col('shot_type') == 'Penalty',1).otherwise(F.col('players_inside_area')))
    
    return df

# Function to get pass information

# Functions to create visualizations

######### Function to call on the main dataset that returns a MLlib ready dataframe #########
def preprocessing(df,spark):
    
    df = df.select(EVENTS_COLUMNS)
    print('Data loaded')
    # Spatial data
    df = spatial_data(df)
    print('Spatial data calculated')
    # Check if the shot was taken with the preferred foot
    df = shot_preferred_foot(df)
    print('Preferred foot calculated')
    # Create goal column
    df = goal(df)
    print('Goal column created')
    # Number of players inside the area
    df = number_of_players_in_area(df,spark)
    print('Number of players inside the area calculated')
    # Convert Boolean data to integer
    df = bool_to_int(df, columns=BOOL_TO_INT_COLUMNS)
    print('Boolean data converted to integer')

    return shot_data(df)


