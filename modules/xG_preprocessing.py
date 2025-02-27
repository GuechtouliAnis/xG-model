import numpy as np
import pandas as pd
import math
import ast
import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.feature import VectorAssembler
from .xG_constants import *

class Preprocessing:
    def __init__(self,
                 spark : SparkSession,
                 df : DataFrame,
                 season : str | None = SEASON,
                 events : list[str] = EVENTS,
                 DUMMIES_dict : dict[str] = DUMMIES,
                 BOOL_TO_INT : list[str] = BOOL_TO_INT,
                 variables : list[str] = VARIABLES,
                 features : list[str] = FEATURES,
                 full_pp : bool = True,
                 keep_shot_frame : bool = True,
                 persist : bool = True,
                 GOAL_X : float = 120,
                 GOAL_Y1 : float = 36,
                 GOAL_Y2 : float = 44):
        """
        Initialize the Preprocessing object.

        This constructor sets up the preprocessing environment by initializing
        various parameters and filtering the input DataFrame. It also defines user
        defined functions (UDFs) for calculating shot angles and counting the number
        of players within a specific area. If full preprocessing is enabled, the
        preprocessing pipeline is executed immediately.

        Parameters
        ----------
        spark : SparkSession
            The active SparkSession instance.
        df : DataFrame
            The input Spark DataFrame containing event data.
        season : str | None, optional
            Season identifier to filter the data. If provided, the DataFrame is filtered
            to include only events from this season (default is SEASON).
        events : list[str], optional
            List of event column names to retain from the DataFrame (default is EVENTS).
        DUMMIES_dict : dict[str], optional
            A dictionary mapping column names to their corresponding dummy variable
            mappings (default is DUMMIES).
        BOOL_TO_INT : list[str], optional
            List of column names that should be converted from boolean to integer (default is BOOL_TO_INT).
        variables : list[str], optional
            List of variables (columns) to retain after full preprocessing (default is VARIABLES).
        full_pp : bool, optional
            Flag indicating whether to run the full preprocessing pipeline upon initialization.
            If True, the `preprocess()` method is called (default is True).
        keep_shot_frame : bool, optional
            Flag indicating whether to store the shot frame data (`shot_frame_df`) for later use.
            If True, `self.shot_frame` will store the processed shot frame data (default is True).
        persist : bool, optional
            Flag indicating whether to persist (cache) the processed DataFrame in memory.
            If True, the DataFrame is cached for faster access during subsequent operations (default is True).
        features : list[str], optional
            List of feature column names to be used later in vector assembly and data splitting (default is FEATURES).
        GOAL_X : float, optional
            The X-coordinate of the goal, used for spatial calculations (default is 120).
        GOAL_Y1 : float, optional
            The lower Y-coordinate of the goal, used for spatial calculations (default is 36).
        GOAL_Y2 : float, optional
            The upper Y-coordinate of the goal, used for spatial calculations (default is 44).

        Notes
        -----
        - If `season` is not None, the DataFrame is filtered to include only rows matching the season.
        - The DataFrame is further filtered to include only shot events or events where 'pass_assisted_shot_id' is not null.
        - Two UDFs are created:
            * `shot_angle_udf`: Computes the angle of a shot based on its location.
            * `check_point_udf`: Determines the number of players within the goal area.
        - If `full_pp` is True, the full preprocessing pipeline (`preprocess()`) is executed.
        - If `keep_shot_frame` is True, the processed shot frame data is stored in `self.shot_frame` and can be accessed later.
        - If `persist` is True, the processed DataFrame is persisted using `pyspark.StorageLevel.MEMORY_AND_DISK`. 
        The `foreach(lambda row: None)` call forces materialization of the cache.
        - The expected columns in the input DataFrame are specified in the constants file (`xG_constants.py`), in the `EVENTS` list.

        """

        # Assign input parameters to instance attributes
        self.df = df
        self.events = events
        self.spark = spark
        self.DUMMIES_dict = DUMMIES_dict
        self.full_pp = full_pp
        self.GOAL_X = GOAL_X
        self.GOAL_Y1 = GOAL_Y1
        self.GOAL_Y2 = GOAL_Y2
        self.BOOL_TO_INT = BOOL_TO_INT
        self.variables = variables
        self.season = season
        self.features = features
        self.keep_shot_frame = keep_shot_frame
        self.shot_frame = None
        self.persist = persist
        
        # Filter the DataFrame by season if a season is specified
        if self.season is not None:
            self.df = self.df.filter(self.df.season == self.season)
        
        # Further filter the DataFrame to include only Shots and Passes leading to a shot or goal
        self.df = self.df.filter((df.type == 'Shot') | (df.pass_assisted_shot_id.isNotNull()))\
                         .select(self.events)

        # Define a UDF to compute the shot angle using the static method 'shot_angle'
        self.shot_angle_udf = F.udf(
            lambda shot_x, shot_y: Preprocessing.shot_angle(
                shot_x, shot_y, GOAL_X, GOAL_Y1, GOAL_Y2),
            T.FloatType())
        
        # Define a UDF to count the number of players within the target area using the static method 'check_point_inside'
        self.check_point_udf = F.udf(
            lambda coords, shot_x, shot_y: Preprocessing.check_point_inside(
                coords, shot_x, shot_y, GOAL_X, GOAL_Y1, GOAL_Y2),
            T.IntegerType())

        # If full preprocessing is enabled, run the complete preprocessing pipeline immediately
        if self.full_pp:
            self.preprocess()

    @staticmethod
    def shot_angle(shot_x : float, shot_y : float,
                   GOAL_X : float, GOAL_Y1 : float,
                   GOAL_Y2 : float) -> float:
        """
        Compute the angle between the vectors from the shot location to each goal post.

        This function calculates the angle of the shot with respect to the goal by
        computing the angle between two vectors:
        - The vector from the shot location (shot_x, shot_y) to the lower goal post (GOAL_X, GOAL_Y1).
        - The vector from the shot location (shot_x, shot_y) to the upper goal post (GOAL_X, GOAL_Y2).

        The angle is computed using the arccosine of the dot product of the two vectors
        divided by the product of their magnitudes, and then converting the result from radians to degrees.

        Parameters
        ----------
        shot_x : float
            The x-coordinate of the shot location.
        shot_y : float
            The y-coordinate of the shot location.
        GOAL_X : float
            The x-coordinate of the goal (common to both goal posts).
        GOAL_Y1 : float
            The y-coordinate of the lower goal post.
        GOAL_Y2 : float
            The y-coordinate of the upper goal post.

        Returns
        -------
        float
            The shot angle in degrees. If either computed vector has a magnitude of zero,
            which would cause a division by zero, the function returns 0.0.
        """
        u_x = GOAL_X - shot_x
        u_y = GOAL_Y1 - shot_y
        v_y = GOAL_Y2 - shot_y
        
        # Calculate the dot product of the two vectors.
        dot_product = u_x ** 2 + u_y * v_y
        
        # Calculate the magnitude of the vector from the shot to each goal post
        magnitude_u = math.sqrt(u_x **2 + u_y ** 2)
        magnitude_v = math.sqrt(u_x **2 + v_y ** 2)
        
        # Check for zero magnitude to avoid division by zero.
        if magnitude_u == 0 or magnitude_v == 0:
            return 0.0

        # Compute the angle in radians between the two vectors using the arccosine of the cosine value.
        angle_radians = math.acos(dot_product / (magnitude_u * magnitude_v))

        # Convert the angle from radians to degrees and return it.
        return math.degrees(angle_radians)

    def split_location(self):
        """
        Split the 'location' column into separate x and y coordinate columns.

        This method extracts the x and y coordinates from the 'location' column of the DataFrame,
        which is in the format "[x, y]". It creates two new columns:
        'shot_location_x' for the x-coordinate and 'shot_location_y' for the y-coordinate,
        casting them as floats.
        After extracting these values, the original 'location' column is dropped.

        Returns
        -------
        None
            The DataFrame is modified in place with the new columns and without the original 'location' column.
        """
        
        self.df = self.df.withColumn("shot_location_x",
                                     F.regexp_extract(F.col("location"), r'\[(.*?),', 1).cast("float"))\
                         .withColumn("shot_location_y",
                                     F.regexp_extract(F.col("location"), r', (.*?)\]', 1).cast("float"))\
                         .drop('location')

        self.df = self.df.withColumn("shot_end_location_clean",
                                     F.regexp_replace(F.col("shot_end_location"), "[\\[\\]]", "")) \
                         .withColumn("shot_end_x",
                                     F.split(F.col("shot_end_location_clean"), ",")[0].cast("double")) \
                         .withColumn("shot_end_y",
                                     F.split(F.col("shot_end_location_clean"), ",")[1].cast("double")) \
                         .drop("shot_end_location_clean")

    def distance_to_goal(self):
        """
        Compute the Euclidean distance from the shot location to the goal and add it as a new column.

        The goal's x-coordinate is defined by `self.GOAL_X` and its y-coordinate is calculated as the midpoint 
        between `self.GOAL_Y1` and `self.GOAL_Y2`. This method computes the distance using the formula:
        
            distance = sqrt((shot_location_x - GOAL_X)^2 + (shot_location_y - midpoint_y)^2)
        
        where `midpoint_y = (GOAL_Y1 + GOAL_Y2) / 2`. The resulting distance is rounded to two decimal places 
        and stored in a new column named "distance_to_goal" in the DataFrame.

        Returns
        -------
        None
        """
            
        self.df = self.df.withColumn("distance_to_goal",
                                     F.round(F.sqrt(
                                         F.pow(self.df.shot_location_x - F.lit(self.GOAL_X), 2) +
                                         F.pow(self.df.shot_location_y - F.lit(self.GOAL_Y1 + self.GOAL_Y2)/2,
                                               2)),
                                     2))

    def get_shot_angle(self):
        """
        Compute the shot angle for each record in the DataFrame.

        This method applies the pre-defined user-defined function (UDF) `shot_angle_udf` to the
        columns `shot_location_x` and `shot_location_y` in order to calculate the angle of the shot
        relative to the goal. The resulting angle is stored in a new column called "shot_angle" in the DataFrame.

        Returns
        -------
        None
            The method modifies the DataFrame in place.
        """
        
        self.df = self.df.withColumn("shot_angle",
                                     self.shot_angle_udf(self.df.shot_location_x, self.df.shot_location_y))

    def spatial_data(self):
        """
        Compute spatial features related to the shot location.

        This method sequentially calls:
        - `split_location()`: Extracts `shot_location_x` and `shot_location_y` from the `location` column.
        - `distance_to_goal()`: Computes the Euclidean distance between the shot location and the goal center.
        - `get_shot_angle()`: Computes the shot angle relative to the goal.

        Returns
        -------
        None
            The method modifies the DataFrame in place.
        """
        
        self.split_location()
        self.distance_to_goal()
        self.get_shot_angle()

    def preferred_foot(self) -> DataFrame:
        """
        Determine the preferred foot of each player based on passing and shooting actions.

        Steps:
        1. Extract foot used from pass and shot actions where the body part is 'Right Foot' or 'Left Foot'.
        2. Merge both datasets into a single DataFrame.
        3. Convert categorical foot usage into numerical indicators (`left_foot`, `right_foot`).
        4. Count the total number of left-footed and right-footed actions for each player.
        5. Calculate the preferred foot:
            - If a player's left foot usage is ≥ 66%, they are left-footed.
            - If a player's right foot usage is ≥ 66%, they are right-footed.
            - Otherwise, they are considered two-footed.

        Returns
        -------
        DataFrame
            A DataFrame with `player_id` and their `preferred_foot`.
        """
        # Extract body part used
        pass_bp = self.df.filter(self.df.type == 'Pass')\
                         .select('player_id', F.col('pass_body_part').alias('body_part'))\
                         .filter(F.col('body_part').isin('Right Foot', 'Left Foot'))
        shot_bp = self.df.filter(self.df.type == 'Shot')\
                         .select('player_id', F.col('shot_body_part').alias('body_part'))\
                         .filter(F.col('body_part').isin('Right Foot', 'Left Foot'))

        # Merge both datasets
        bp = pass_bp.union(shot_bp)

        # Convert categorical body part into numerical indicators
        bp_mapped = bp.withColumn('left_foot',
                                  (F.col('body_part')=='Left Foot').cast('int'))\
                      .withColumn('right_foot',
                                  (F.col('body_part')=='Right Foot').cast('int'))\
                      .drop('body_part')

        # Aggregate foot usage by player
        foot_counts = bp_mapped.groupBy('player_id')\
                               .sum('left_foot', 'right_foot')\
                               .withColumnRenamed('sum(left_foot)', 'left_foot')\
                               .withColumnRenamed('sum(right_foot)', 'right_foot')

        # Compute total actions per player
        foot_counts = foot_counts.withColumn("total_actions",
                                             F.col("left_foot") + F.col("right_foot"))

        # Determine the preferred foot based on usage percentage
        foot_counts = foot_counts.withColumn("preferred_foot",
                                             F.when((F.col("left_foot") / F.col("total_actions")) >= 0.66, "Left Foot")\
                                              .when((F.col("right_foot") / F.col("total_actions")) >= 0.66, "Right Foot")\
                                              .otherwise("Two-Footed"))

        return foot_counts.drop('left_foot', 'right_foot', 'total_actions')

    def shot_preferred_foot(self):
        """
        Determine whether a shot was taken with the player's preferred foot.

        Steps:
        1. Retrieve the preferred foot classification for each player.
        2. Join this information with the main dataset on `player_id`.
        3. Create a new column `preferred_foot_shot`:
            - If the player is two-footed and uses either foot, mark as True.
            - Otherwise, check if the shot was taken with the preferred foot.

        Modifies
        -------
        self.df : DataFrame
            Adds the `preferred_foot_shot` column.
        """
        
        # Retrieve preferred foot classification
        df = self.preferred_foot()
        
        # Rename player_id to avoid conflicts during join
        df = df.withColumnRenamed('player_id','f_player_id')
        
        # Join with the main dataset
        self.df = self.df.join(df, self.df.player_id == df.f_player_id, how='left')\
                         .drop('f_player_id')

        # Determine if the shot was taken with the preferred foot
        self.df = self.df.withColumn('preferred_foot_shot',
                                     F.when((F.col('preferred_foot')=='Two-Footed') &
                                            (F.col('shot_body_part').isin('Right Foot', 'Left Foot')), True)\
                                      .otherwise(F.col('preferred_foot') == F.col('shot_body_part')))

    def shot_freeze_frame(self) -> DataFrame:
        """
        Extracts non-penalty shot freeze frame data.

        Filters out penalty shots and selects only relevant columns.

        Returns
        -------
        DataFrame
            Contains shot IDs and their associated freeze frame data.
        """
        
        return self.df.filter(F.col('shot_type')!='Penalty')\
                      .select('id', 'shot_freeze_frame')\
                      .dropna(subset=['shot_freeze_frame'])

    def shot_frame_df(self) -> pd.DataFrame:
        """
        Processes freeze frame data into a structured pandas DataFrame.

        Steps:
        1. Collects freeze frame data from `shot_freeze_frame`.
        2. Parses the JSON-like string into individual player positions.
        3. Extracts:
            - X, Y coordinates of the player.
            - Player's position.
            - Whether the player is a teammate or opponent.
        4. Constructs a structured DataFrame.

        Returns
        -------
        self.shot_frame
            A structured dataset containing player positioning information during shot events.
        """
        
        # Get freeze frame data
        df = self.shot_freeze_frame()
        processed_rows = []
        
        # Iterate through collected rows        
        for row in df.collect():
            
            id = row.id
            # Split individual player data
            shot_frame = row.shot_freeze_frame.split('}, {')
            
            for i in range(len(shot_frame)):
                
                # Extract X, Y coordinates
                x, y = ast.literal_eval(shot_frame[i].split('},')[0].split(': ')[1].split('],')[0] + ']')

                # Extract player position
                position = shot_frame[i].split('},')[1].split(': ')[3].strip("''")

                # Determine if the player is a teammate
                teammate = 'True' if 'True' in shot_frame[i].split('},')[2] else 'False'

                # Append extracted information
                processed_rows.append({'Shot_id': id, 'x': x, 'y': y, 'position': position, 'teammate': teammate})
        
        if self.keep_shot_frame:
            self.shot_frame = pd.DataFrame(processed_rows)
            return self.shot_frame
        
        return pd.DataFrame(processed_rows)

    @staticmethod
    def check_point_inside(coordinates : list[float],
                           shot_x : float, shot_y : float,
                           GOAL_X : float = 120, GOAL_Y1 : float = 36,
                           GOAL_Y2 : float = 44) -> int:
        """
        Counts the number of players inside the shooting triangle.

        Parameters
        ----------
        coordinates : list of float
            List of (x, y) player positions.
        shot_x : float
            X-coordinate of the shot.
        shot_y : float
            Y-coordinate of the shot.
        GOAL_X : float, default=120
            X-coordinate of the goal.
        GOAL_Y1 : float, default=36
            Y-coordinate of the left post.
        GOAL_Y2 : float, default=44
            Y-coordinate of the right post.

        Returns
        -------
        int
            Number of players inside the triangle formed by the shot and goalposts.
        """
        
        def calculate_area(x1 : float, y1 : float,
                           x2 : float, y2 : float,
                           x3 : float, y3 : float) -> float:
            """
            Calculates the area of a triangle given three points using the Shoelace formula.
            """
            return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0

        def point_inside_triangle(player_x : float, player_y : float,
                                  shot_x : float, shot_y : float,
                                  GOAL_X : float, GOAL_Y1 : float,
                                  GOAL_Y2 : float) -> bool:
            """
            Checks if a given player position is inside the triangle formed by the shot and goalposts.
            """
            
            # Compute the total area of the goal-shot triangle
            area_abc = calculate_area(shot_x, shot_y, GOAL_X, GOAL_Y1, GOAL_X, GOAL_Y2)
            
            # Compute the areas of sub-triangles formed with the player
            area_abp = calculate_area(shot_x, shot_y, GOAL_X, GOAL_Y1, player_x, player_y)
            area_bcp = calculate_area(shot_x, shot_y, GOAL_X, GOAL_Y2, player_x, player_y)
            area_cap = calculate_area(player_x, player_y, GOAL_X, GOAL_Y2, GOAL_X, GOAL_Y1)
            
            # If the sum of sub-triangle areas is approximately equal to the main triangle, the player is inside
            return np.isclose(area_abc, (area_abp + area_bcp + area_cap),rtol=1e-10)

        count = 0
        for coord in coordinates:
            player_x, player_y = coord
            if point_inside_triangle(player_x, player_y, shot_x, shot_y, GOAL_X, GOAL_Y1, GOAL_Y2):
                count += 1
        return count

    def number_of_players(self):
        """
        Computes the number of players inside the shot triangle for each shot.

        Steps:
        1. Extracts player positions from the freeze frame data.
        2. Converts positions into a structured list per shot.
        3. Merges shot locations with player positions.
        4. Uses UDF to count the number of players inside the shot triangle.
        5. Handles penalty shots separately (defaulting to 1 player inside).

        Returns
        -------
        None
            Updates self.df with the new column `players_inside_area`.
        """
        
        # Convert player positions into a list of (x, y) coordinates per shot
        frames = self.shot_frame_df()
        frames = (frames.groupby('Shot_id')\
                        .apply(lambda group: group[['x','y']].values.tolist(), include_groups=False)\
                        .reset_index(name='coordinates'))
        
        # Convert pandas DataFrame to Spark DataFrame
        frames = self.spark.createDataFrame(frames)
        
        # Merge with shot location data
        frames = frames.join(self.df.select('id','shot_location_x','shot_location_y'),
                             frames.Shot_id == self.df.id)\
                       .drop('id')

        # Apply the user-defined function to compute the number of players inside the area
        frames = frames.withColumn('players_inside_area',
                                   self.check_point_udf(F.col('coordinates'), F.col('shot_location_x'), F.col('shot_location_y')))

        # Merge the computed feature into the main dataset, ensuring penalties default to 1 player
        self.df = self.df.join(frames.select('Shot_id','players_inside_area'), self.df.id == frames.Shot_id, how='left')\
                         .drop('Shot_id')\
                         .withColumn('players_inside_area',
                                     F.when(F.col('shot_type')=='Penalty', 1)\
                         .otherwise(F.col('players_inside_area')))

    def create_dummies(self):
        """
        Converts categorical columns into dummy variables and processes specific features.

        Steps:
        1. Iterates through `DUMMIES_dict` to create binary (dummy) columns.
        2. Drops the original categorical columns after transformation.
        3. Creates additional binary features:
        - `goal`: Whether the shot resulted in a goal (1) or not (0).
        - `header`: Whether the shot was taken with the head (1) or not (0).
        4. Drops `shot_body_part` after processing.

        Returns
        -------
        None
            Updates `self.df` with new dummy variables and transformed columns.
        """
        
        # Convert categorical variables into dummy variables
        for col_name, mapping in self.DUMMIES_dict.items():
            for value, dummy_col in mapping.items():
                self.df = self.df.withColumn(dummy_col,
                                             F.when(F.col(col_name) == value, 1)\
                                 .otherwise(0))
            # Drop original categorical column after encoding
            self.df = self.df.drop(col_name)
            
        # Create binary target variable for goal outcome, headers and drop original shot_body_part column
        self.df = self.df.withColumn('goal',
                                     F.col('shot_outcome') == 'Goal')\
                         .withColumn('header',
                                     F.when((F.col('shot_body_part')=='Head'), 1)\
                         .otherwise(0))\
                         .drop('shot_body_part')

    def bool_to_int(self):
        """
        Converts boolean-like columns into integer (0 or 1) format and processes 'shot_one_on_one'.

        Steps:
        1. Iterates over `BOOL_TO_INT` to cast boolean columns to integers.
        - If a column is `NULL`, it defaults to `0`.
        2. Adjusts `shot_one_on_one` based on `pk_type`.
        - If `pk_type == 1`, it ensures `shot_one_on_one` is `1`.

        Returns
        -------
        None
            Updates `self.df` with modified columns.
        """
        
        # Convert boolean columns to integers (0/1)
        for col_name in self.BOOL_TO_INT:
            self.df = self.df.withColumn(col_name,
                                         F.when(F.col(col_name).isNull(), 0)\
                             .otherwise(F.col(col_name).cast('int')))

        # Ensure shot_one_on_one is set to 1 if pk_type is 1
        self.df = self.df.withColumn('shot_one_on_one',
                                     F.when(F.col('pk_type')==1, 1)\
                         .otherwise(F.col('shot_one_on_one')))

    def get_assist_data(self):
        """
        Extracts assist-related features from passing events and joins them with shot data.

        Steps:
        1. Identifies passing-related columns (`pass_events`) except `pass_body_part`.
        2. Creates a subset (`df_p`) with only pass-related data.
        3. Encodes `pass_height` into numerical values:
        - 0: Ground Pass
        - 1: Low Pass
        - 2: High Pass
        4. Renames `under_pressure` to avoid name conflicts (`pass_under_pressure`).
        5. Drops overlapping columns in `self.df` before merging.
        6. Joins `df_p` (pass data) with `self.df` (shot data) on `pass_assisted_shot_id`.
        7. Fills missing values with default values:
        - `assisted` → 1 if pass exists, else 0
        - `pass_height` → -1 if missing
        - `pass_angle` → 4 if missing
        - `pass_length` → 0 if missing
        8. Drops `pass_assisted_shot_id` to avoid redundancy.

        Returns:
        -------
        None
            Updates `self.df` with assist-related columns.
        """
        
        # Extract passing-related events except 'pass_body_part'
        pass_events = [ev for ev in self.events if 'pass_' in ev]
        pass_events.remove('pass_body_part')
        
        # Subset DataFrame for pass-related columns
        df_p = self.df.select(pass_events)
        
        # Encode 'pass_height' into numerical values
        df_p = df_p.withColumn('pass_height',
                               F.when(F.col('pass_height')=='Ground Pass', 0)\
                                .when(F.col('pass_height')=='Low Pass', 1)
                                .otherwise(2))
        
        # Rename 'under_pressure' to avoid conflicts
        df_p = df_p.withColumnRenamed('under_pressure','pass_under_pressure')

        # Remove duplicate columns before merging
        for col in self.df.columns:
            if col in df_p.columns:
                self.df = self.df.drop(col)
        
        # Join pass data with shot data
        self.df = self.df.join(df_p, self.df.id == df_p.pass_assisted_shot_id, how='left')

        # Fill Null values when the shot doesn't have an assist and create 'assisted' column
        self.df = self.df.withColumn('assisted',
                                     F.when(F.col('pass_assisted_shot_id').isNotNull(), 1)\
                                      .otherwise(0))\
                         .withColumn('pass_height',
                                     F.when(F.col('pass_height').isNull(), -1)\
                                      .otherwise(F.col('pass_height')))\
                         .withColumn('pass_angle',
                                     F.when(F.col('pass_angle').isNull(),   4)\
                                      .otherwise(F.col('pass_angle')))\
                         .withColumn('pass_length',
                                     F.when(F.col('pass_length').isNull(),   0)\
                                      .otherwise(F.col('pass_length')))\
                         .drop('pass_assisted_shot_id')
                         
    def preprocess(self):
        """
        Executes the full preprocessing pipeline for shot data.

        Steps:
        1. Computes spatial shot data (distance, angle).
        2. Determines the preferred foot of the shooter.
        3. Counts players inside the shooting angle.
        4. Integrates assist-related data.
        5. Creates dummy variables for categorical features.
        6. Converts boolean-like features to integers.
        7. Filters the dataset to keep only shot events and selected features.
        8. Converts the processed DataFrame into `EDASparkDataFrame`.

        Returns:
        -------
        None
            Updates `self.df` with fully processed data.
        """
        
        self.spatial_data()         # Compute distance, shot angle
        self.shot_preferred_foot()  # Determine preferred shooting foot
        self.number_of_players()    # Count defenders in shooting angle
        self.get_assist_data()      # Merge assist-related pass data
        self.create_dummies()       # Create dummy variables
        self.bool_to_int()          # Convert boolean features to integers

        # Converting Statsbomb xG predictions
        self.df = self.df.withColumn('sb_prediction', F.when(F.col('shot_statsbomb_xg')>=0.5, 1).otherwise(0))
        
        # Keep only shot events and selected features
        self.df = self.df.filter(self.df.type=='Shot').select(self.variables)

        if self.persist:
            self.df = self.df.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
            self.df.foreach(lambda row: None)

    def data_split(self,
                   train_size : float = 0.8,
                   seed : int = 42) -> tuple[DataFrame,DataFrame]:
        """
        Splits the dataset into training and test sets.

        Steps:
        1. Assembles feature columns into a single vector column (`features_vector`).
        2. Uses `randomSplit` to divide the dataset into training and test sets.
        
        Parameters:
        ----------
        train_size : float, optional (default=0.8)
            The proportion of the dataset to be used for training.
        seed : int, optional (default=42)
            Random seed for reproducibility.

        Returns:
        -------
        tuple[DataFrame, DataFrame]
            A tuple containing:
            - `train_data`: Training dataset.
            - `test_data`: Test dataset.
        """
        
        # Convert feature columns into a single vector column for ML models
        feature_assembler = VectorAssembler(inputCols=self.features,
                                            outputCol="features_vector")
        assembled_data = feature_assembler.transform(self.df)

        # Split the data into training and test sets
        train_data, test_data = assembled_data.randomSplit([train_size, 1-train_size],
                                                           seed=seed)

        return train_data, test_data