from pyspark.sql.functions import col, regexp_extract,round, sqrt, pow, lit, udf, when, format_number
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import IntegerType, FloatType, DoubleType
import pandas as pd
import numpy as np
import ast, math

######### Shot DataFrame necessary columns #########
EVENTS_COLUMNS = ['id','period','duration','location','player_id','position', # Event info
                'play_pattern','shot_body_part','shot_technique','shot_type',        # Shot info
                'related_events', 'shot_freeze_frame', 'shot_key_pass_id', 'shot_end_location', # Complicated info
                'under_pressure','shot_aerial_won','shot_first_time','shot_one_on_one','shot_open_goal',
                'shot_follows_dribble', # Boolean
                'shot_statsbomb_xg','shot_outcome',# Target
                'pass_body_part','type','shot_freeze_frame'
                ]

ML_READY_DATA_DUMMIES = ['id','player_id','shot_location_x','shot_location_y','distance_to_goal','shot_angle','preferred_foot_shot',
                 'other_pp','from_fk','from_ti','from_corner','from_counter','from_gk','from_keeper','from_ko',
                 'header','corner_type','fk_type','pk_type',
                 'half_volley_technique','volley_technique','lob_technique','overhead_technique','backheel_technique',
                 'diving_h_technique',
                 'under_pressure','shot_aerial_won','shot_first_time','shot_one_on_one','shot_open_goal','shot_follows_dribble',
                 'players_inside_area',
                 'shot_statsbomb_xg','shot_outcome','goal']

ML_READY_DATA = ['id','shot_location_x','shot_location_y','distance_to_goal','shot_angle','preferred_foot_shot',
                 'shot_body_part','shot_technique','shot_type','play_pattern',
                 'under_pressure','shot_aerial_won','shot_first_time','shot_one_on_one','shot_open_goal','shot_follows_dribble',
                 'players_inside_area',
                 'shot_statsbomb_xg','shot_outcome','goal']

BOOL_TO_INT_COLUMNS = ['preferred_foot_shot','under_pressure','shot_aerial_won','shot_first_time','shot_one_on_one',
                      'shot_open_goal','shot_follows_dribble','goal']

FEATURES = ['other_pp','from_fk','from_ti','from_corner','from_counter','from_gk','from_keeper','from_ko',
            'header','corner_type','fk_type','pk_type',
            'half_volley_technique','volley_technique','lob_technique','overhead_technique','backheel_technique',
            'diving_h_technique',
            'distance_to_goal', 'shot_angle', 'preferred_foot_shot', 'under_pressure',
            'shot_aerial_won','shot_first_time','shot_one_on_one','shot_open_goal','shot_follows_dribble','players_inside_area']

goal_X = 120
goal_Y1, goal_Y2 = 36, 44

######### Function to export shot data #########
def shot_data(df):
    df = df.filter(df.type=='Shot').select(ML_READY_DATA_DUMMIES)
    return df.withColumn('sb_prediction', round(col('shot_statsbomb_xg')))

######### Function to split the location column into x and y coordinates #########
def split_location(df):
    """
    Splits the 'location' column into separate x and y coordinate columns.

    :param df: PySpark DataFrame with a 'location' column containing string representations of coordinates.
    :return: Updated DataFrame with 'shot_location_x' and 'shot_location_y' as float columns, and the 'location' column removed.
    """
    df_l = df.withColumn("shot_location_x", regexp_extract(col("location"), r'\[(.*?),', 1).cast("float")) \
             .withColumn("shot_location_y", regexp_extract(col("location"), r', (.*?)\]', 1).cast("float")).drop('location')
    return df_l

######### Function to calculate the distance to the goal #########
def distance_to_goal(df):
    """
    Calculates the distance from the shot location to the goal.

    :param df: PySpark DataFrame with 'shot_location_x' and 'shot_location_y' columns.
    :return: Updated DataFrame with an additional 'distance_to_goal' column, rounded to 4 decimal places.
    """
    goal_x, goal_y = 120, 40
    df = df.withColumn("distance_to_goal",
                       round(sqrt(pow(df.shot_location_x - lit(goal_x), 2) + pow(df.shot_location_y - lit(goal_y), 2)),4))
    return df

######### Function to calculate the angle to the goal #########
def calculate_shot_angle(shot_x, shot_y):
    """
    Calculates the angle between two vectors pointing to the left and right goalposts from a shot location.

    :param shot_x: Float representing the x-coordinate of the shot location.
    :param shot_y: Float representing the y-coordinate of the shot location.
    :return: Angle in degrees between the shot location and the two goalposts.
    """

    goal_x1, goal_y1 = 120, 36  # Left post
    goal_x2, goal_y2 = 120, 44  # Right post

    # Vectors to the posts
    u_x, u_y = goal_x1 - shot_x, goal_y1 - shot_y
    v_x, v_y = goal_x2 - shot_x, goal_y2 - shot_y
    
    # Dot product and magnitudes
    dot_product = u_x * u_x + u_y * v_y
    magnitude_u = math.sqrt(u_x**2 + u_y**2)
    magnitude_v = math.sqrt(u_x**2 + v_y**2)
    
    # Avoid division by zero
    if magnitude_u == 0 or magnitude_v == 0:
        return 0.0
    
    # Calculate angle in radians
    angle_radians = math.acos(dot_product / (magnitude_u * magnitude_v))
    
    # Convert to degrees
    return math.degrees(angle_radians)

def get_shot_angle(df):
    """
    Calculates the shot angle relative to the goalposts and adds it to the DataFrame.

    :param df: PySpark DataFrame with 'shot_location_x' and 'shot_location_y' columns.
    :return: Updated DataFrame with an additional 'shot_angle' column containing the angle in degrees.
    """

    # Register the function as a UDF
    calculate_shot_angle_udf = udf(calculate_shot_angle, FloatType())
    
    # Apply the UDF to the DataFrame
    df = df.withColumn(
        "shot_angle",
        calculate_shot_angle_udf(df["shot_location_x"], df["shot_location_y"]))
    return df

######### Function to get Spatial Data #########
def spatial_data(df):
    """
    Processes spatial data by splitting the shot location, calculating the distance to goal, 
    and determining the shot angle.

    :param df: PySpark DataFrame with the necessary columns ('location', 'shot_location_x', 'shot_location_y').
    :return: Updated DataFrame with 'shot_location_x', 'shot_location_y', 'distance_to_goal', and 'shot_angle' columns.
    """

    df = split_location(df)
    df = distance_to_goal(df)
    return get_shot_angle(df)

######### Function to get the favorite foot of the shooter #########
def preferred_foot(df):
    """
    Determines the preferred foot of each player based on their pass and shot body parts.

    :param df: PySpark DataFrame with 'type', 'player_id', 'pass_body_part', and 'shot_body_part' columns.
    :return: DataFrame with 'player_id' and 'preferred_foot' columns, indicating the player's dominant foot ('Left Foot', 'Right Foot', or 'Two-Footed').
    """
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
    bp_mapped = bp.withColumn(
        'left_foot',
        (col('body_part') == 'Left Foot').cast('int'))\
            .withColumn('right_foot',
                    (col('body_part') == 'Right Foot').cast('int'))\
                        .drop('body_part')

    # Group by player and calculate sums for left and right foot counts
    foot_counts = bp_mapped.groupBy('player_id')\
        .sum('left_foot', 'right_foot')\
        .withColumnRenamed('sum(left_foot)', 'left_foot')\
        .withColumnRenamed('sum(right_foot)', 'right_foot')

    # Add total actions column
    foot_counts = foot_counts.withColumn(
        "total_actions",
        col("left_foot") + col("right_foot"))

    # Determine preferred foot based on the 66% rule
    foot_counts = foot_counts.withColumn(
        "preferred_foot",
        when((col("left_foot") / col("total_actions")) >= 0.66, "Left Foot")
        .when((col("right_foot") / col("total_actions")) >= 0.66, "Right Foot")
        .otherwise("Two-Footed"))

    return foot_counts.drop('left_foot', 'right_foot', 'total_actions')

######### Function to check if the shot was taken with the preferred foot #########
def shot_preferred_foot(df):
    """
    Checks if the shot was taken with the player's preferred foot.

    :param df: PySpark DataFrame with 'player_id', 'shot_body_part', and other relevant columns.
    :return: Updated DataFrame with an additional 'preferred_foot_shot' column, indicating whether the shot was taken with the player's preferred foot (True or False).
    """
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
    """
    Adds a 'goal' column indicating whether the shot resulted in a goal.

    :param df: PySpark DataFrame with a 'shot_outcome' column.
    :return: Updated DataFrame with an additional 'goal' column (True if the shot outcome is 'Goal', False otherwise).
    """
    df = df.withColumn('goal', col('shot_outcome') == 'Goal')
    return df

######### Function to convert boolean columns to integer #########
def bool_to_int(df, columns):
    """
    Converts boolean columns to integer (0 or 1) in the DataFrame.

    :param df: PySpark DataFrame with boolean columns.
    :param columns: List of column names to be converted.
    :return: Updated DataFrame with the specified boolean columns converted to integers.
    """
    for col_name in columns:
        df = df.withColumn(col_name, when(col(col_name).isNull(), 0).otherwise(col(col_name).cast('int')))
    return df.fillna(0)

######### Function to export shot_freeze_frame dataframe [id, shot_freeze_frame] #########
def shot_freeze_frame(df):
    """
    Exports a DataFrame with shot IDs and their corresponding freeze frames, excluding penalties.

    :param df: PySpark DataFrame with 'shot_type', 'id', and 'shot_freeze_frame' columns.
    :return: A DataFrame with 'id' and 'shot_freeze_frame' columns, excluding rows with missing or penalty shots.
    """
    return df.filter(col('shot_type')!='Penalty').select('id', 'shot_freeze_frame').dropna(subset=['shot_freeze_frame'])

######### Function to export shot_freeze_frame to a separate dataframe #########
def shot_frame_to_df(df):
    """
    Exports shot freeze frames into a separate DataFrame, processing the 'shot_freeze_frame' column.

    :param df: PySpark DataFrame with 'id' and 'shot_freeze_frame' columns.
    :return: A Pandas DataFrame with 'Shot_id', 'x', 'y', 'position', and 'teammate' columns extracted from the shot freeze frames.
    """
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
    """
    Calculates the area of a triangle given the coordinates of its three vertices.

    :param x1, y1: Coordinates of the first vertex.
    :param x2, y2: Coordinates of the second vertex.
    :param x3, y3: Coordinates of the third vertex.
    :return: The area of the triangle formed by the three points.
    """
    return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0

def is_point_inside_triangle(player_x,player_y,shot_x,shot_y):
    """
    Checks if a player is inside the shooting area defined by the shot and goalposts.

    :param player_x, player_y: Coordinates of the player's position.
    :param shot_x, shot_y: Coordinates of the shot location.
    :return: True if the player is inside the triangle formed by the shot and goalposts, False otherwise.
    """
    # Total area of the triangle ABC
    area_abc = calculate_area(shot_x,shot_y,goal_X,goal_Y1,goal_X,goal_Y2)

    # Area of sub-triangles ABP, BCP, CAP
    area_abp = calculate_area(shot_x,shot_y,goal_X,goal_Y1,player_x,player_y)
    area_bcp = calculate_area(shot_x,shot_y,goal_X,goal_Y2,player_x,player_y)
    area_cap = calculate_area(player_x,player_y,goal_X,goal_Y2,goal_X,goal_Y1)

    return np.isclose(area_abc, (area_abp + area_bcp + area_cap))

def check_point_inside(coordinates, shot_x, shot_y):
    """
    Counts how many players are inside the shooting area defined by the shot and goalposts.

    :param coordinates: List of tuples, each containing the (x, y) coordinates of a player.
    :param shot_x, shot_y: Coordinates of the shot location.
    :return: The number of players inside the shooting area.
    """
    count = 0
    for coord in coordinates:
        player_x, player_y = coord
        if is_point_inside_triangle(player_x, player_y, shot_x, shot_y):
            count += 1
    return count

def number_of_players_in_area(df,spark):
    """
    Calculates the number of players inside the shooting area for each shot and adds it to the DataFrame.

    :param df: PySpark DataFrame containing shot data, including shot location.
    :param spark: Spark session object.
    :return: Updated DataFrame with an additional 'players_inside_area' column indicating the number of players inside the shooting area for each shot.
    """

    frames = shot_frame_to_df(df)
    frames = (frames.groupby('Shot_id').apply(lambda group: group[['x', 'y']].values.tolist()).reset_index(name='coordinates'))
    frames = spark.createDataFrame(frames)
    frames = frames.join(df.select('id','shot_location_x','shot_location_y'),frames.Shot_id == df.id).drop('id')

    # Register the UDF
    check_point_udf = udf(check_point_inside, returnType=IntegerType())  # Use IntegerType from pyspark.sql.types

    # Add a column with the count of points inside the area for each row in the DataFrame
    frames = frames.withColumn("players_inside_area", check_point_udf(col("coordinates"), col("shot_location_x"), col("shot_location_y")))
    df = df.join(frames.select('Shot_id','players_inside_area'),df.id == frames.Shot_id,how='left').drop('Shot_id')
    df = df.withColumn('players_inside_area',when(col('shot_type') == 'Penalty',1).otherwise(col('players_inside_area')))
    
    return df

######### Function to create Dummies #########
def play_pattern_dummies(df):
    """
    Creates dummy variables for different play patterns and removes the original 'play_pattern' column.

    :param df: PySpark DataFrame with a 'play_pattern' column.
    :return: Updated DataFrame with dummy columns representing various play patterns (e.g., 'from_fk', 'from_ti', etc.).
    """
    df = df.withColumn('other_pp',when((col('play_pattern')=='Other'),1).otherwise(0)) \
    .withColumn('from_fk',        when((col('play_pattern')=='From Free Kick'),1).otherwise(0)) \
    .withColumn('from_ti',        when((col('play_pattern')=='From Throw In'),1).otherwise(0)) \
    .withColumn('from_corner',    when((col('play_pattern')=='From Corner'),1).otherwise(0)) \
    .withColumn('from_counter',   when((col('play_pattern')=='From Counter'),1).otherwise(0)) \
    .withColumn('from_gk',        when((col('play_pattern')=='From Goal Kick'),1).otherwise(0)) \
    .withColumn('from_keeper',    when((col('play_pattern')=='From Keeper'),1).otherwise(0)) \
    .withColumn('from_ko',             when((col('play_pattern')=='From Kick Off'),1).otherwise(0)).drop('play_pattern')
    return df

def header_dummies(df):
    """
    Creates a dummy variable for header shots based on the 'shot_body_part' column and removes the original column.

    :param df: PySpark DataFrame with a 'shot_body_part' column.
    :return: Updated DataFrame with a 'header' column indicating whether the shot was a header (1) or not (0).
    """
    return df.withColumn('header', when((col('shot_body_part')=='Head'),1).otherwise(0)).drop('shot_body_part')

def shot_type_dummies(df):
    """
    Creates dummy variables for different shot types and removes the original 'shot_type' column.

    :param df: PySpark DataFrame with a 'shot_type' column.
    :return: Updated DataFrame with dummy columns representing shot types (e.g., 'corner_type', 'fk_type', 'pk_type').
    """

    df = df.withColumn('corner_type', when((col('shot_type')=='Corner'),1).otherwise(0)) \
    .withColumn('fk_type',            when((col('shot_type')=='Free Kick'),1).otherwise(0)) \
    .withColumn('pk_type',            when((col('shot_type')=='Penalty'),1).otherwise(0)) \
    .drop('shot_type')
    return df

def shot_technique_dummies(df):
    """
    Creates dummy variables for different shot techniques and removes the original 'shot_technique' column.

    :param df: PySpark DataFrame with a 'shot_technique' column.
    :return: Updated DataFrame with dummy columns representing shot techniques (e.g., 'half_volley_technique', 'volley_technique', etc.).
    """

    df = df.withColumn('half_volley_technique',when((col('shot_technique')=='Half Volley'),1).otherwise(0)) \
    .withColumn('volley_technique',           when((col('shot_technique')=='Volley'),1).otherwise(0)) \
    .withColumn('lob_technique',              when((col('shot_technique')=='Lob'),1).otherwise(0)) \
    .withColumn('overhead_technique',         when((col('shot_technique')=='Overhead Kick'),1).otherwise(0)) \
    .withColumn('backheel_technique',         when((col('shot_technique')=='Backheel'),1).otherwise(0)) \
    .withColumn('diving_h_technique',         when((col('shot_technique')=='Diving Header'),1).otherwise(0)) \
    .drop('shot_technique')
    return df


def create_dummies(df):
    """
    Creates dummy variables for play patterns, shot body part, shot type, and shot technique.

    :param df: PySpark DataFrame with columns for 'play_pattern', 'shot_body_part', 'shot_type', and 'shot_technique'.
    :return: Updated DataFrame with dummy variables for each of the mentioned columns.
    """
    df = play_pattern_dummies(df)
    df = header_dummies(df)
    df = shot_type_dummies(df)
    return shot_technique_dummies(df)

######### Function to call on the main dataset that returns a MLlib ready dataframe #########
def preprocessing(df,spark):
    """
    Preprocesses the main dataset by applying various transformations and calculations to prepare it for machine learning.

    :param df: PySpark DataFrame containing shot data.
    :param spark: Spark session object.
    :return: Preprocessed DataFrame, ready for use in machine learning models.
    """    
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
    # Create Dummies
    df = create_dummies(df)
    print('Dummies created')
    # Convert Boolean data to integer
    df = bool_to_int(df, columns=BOOL_TO_INT_COLUMNS)
    print('Boolean data converted to integer')

    return shot_data(df)

######### Function to split the data into training and testing #########
def pre_training(df,features=FEATURES,train_size=0.8):
    """
    Prepares the dataset for training by assembling features into a vector 
    and splitting it into training and testing subsets.

    :param df: PySpark DataFrame to be processed.
    :param features: List of feature column names to assemble into a vector.
    :param train_size: Float indicating the proportion of data for training (default is 0.8).
    :return: A tuple (train_data, test_data) with the training and testing DataFrames.
    """
    feature_assembler = VectorAssembler(inputCols=features, outputCol="features_vector")
    assembled_data = feature_assembler.transform(df)
    train_data, test_data = assembled_data.randomSplit([train_size, 1-train_size], seed=42)
    return train_data, test_data

######### Function to extract the goal probability #########
def goal_proba(df):
    """
    Processes the goal probability column in the given df DataFrame.

    :param df: PySpark DataFrame with a 'probability' column containing lists.
    :return: Updated DataFrame with the 'goal_probability' column as a float.
    """
    # Define the function to extract the second element from the probability list
    def extract_goal_probability(probability):
        return float(probability[1])

    # Register the function as a UDF
    extract_goal_probability_udf = udf(extract_goal_probability, DoubleType())

    # Overwrite the prediction column using the UDF
    df = df.withColumn("goal_probability", extract_goal_probability_udf(col("probability")))

    # Format the goal_probability to remove scientific notation
    df = df.withColumn("goal_probability", format_number(col("goal_probability"), 10))
    
    # Convert goal_probability to float
    return df.withColumn("goal_probability", col("goal_probability").cast(DoubleType()))
