import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler
import numpy as np
import pandas as pd
from .xG_constants import *

# TO ADD
# - Penalty / FK Logic (if Header!)
# - REVIEW PLAYERS INSIDE AREA FUNCTION
# - Try except (column does not exist)
# - Shot angle math
# - Remove UDFs
# - Add pre-training
# - Add comments
# - Docstring

class Preprocessing:
    def __init__(self,
                 spark,
                 df : DataFrame,
                 season : str | None = SEASON,
                 events : list[str] = EVENTS,
                 DUMMIES_dict : dict[str] = DUMMIES,
                 BOOL_TO_INT : list[str] = BOOL_TO_INT,
                 variables : list[str] = VARIABLES,
                 full_pp : bool = True,
                 features : list[str] = FEATURES,
                 GOAL_X : float = 120,
                 GOAL_Y1 : float = 36,
                 GOAL_Y2 : float = 44):

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
        
        if self.season is not None:
            self.df = self.df.filter(self.df.season == self.season)
        
        self.df = self.df.filter((df.type == 'Shot') | (df.pass_assisted_shot_id.isNotNull()))\
                         .select(self.events)

        self.shot_angle_udf = F.udf(
            lambda shot_x, shot_y: Preprocessing.shot_angle(
                shot_x, shot_y, GOAL_X, GOAL_Y1, GOAL_Y2),
            T.FloatType())

        self.check_point_udf = F.udf(
            lambda coords, shot_x, shot_y: Preprocessing.check_point_inside(
                coords, shot_x, shot_y, GOAL_X, GOAL_Y1, GOAL_Y2),
            T.IntegerType())

        if self.full_pp:
            self.preprocess()

    # def __getattr__(self, attr) -> DataFrame:
    #     return getattr(self.df, attr)

    @staticmethod
    def shot_angle(shot_x : float, shot_y : float,
                   GOAL_X : float, GOAL_Y1 : float,
                   GOAL_Y2 : float) -> float:
        import math
        u_x = GOAL_X - shot_x
        u_y = GOAL_Y1 - shot_y
        v_y = GOAL_Y2 - shot_y
        dot_product = u_x ** 2 + u_y * v_y
        magnitude_u = math.sqrt(u_x **2 + u_y ** 2)
        magnitude_v = math.sqrt(u_x **2 + v_y ** 2)
        if magnitude_u == 0 or magnitude_v == 0:
            return 0.0
        angle_radians = math.acos(dot_product / (magnitude_u * magnitude_v))
        return math.degrees(angle_radians)

    def split_location(self):
        self.df = self.df.withColumn("shot_location_x",
                                     F.regexp_extract(F.col("location"), r'\[(.*?),', 1).cast("float"))\
                         .withColumn("shot_location_y",
                                     F.regexp_extract(F.col("location"), r', (.*?)\]', 1).cast("float"))\
                         .drop('location')

    def distance_to_goal(self):
        self.df = self.df.withColumn("distance_to_goal",
                                     F.round(F.sqrt(
                                         F.pow(self.df.shot_location_x - F.lit(self.GOAL_X), 2) +
                                         F.pow(self.df.shot_location_y - F.lit(self.GOAL_Y1 + self.GOAL_Y2)/2,
                                               2)),
                                     2))

    def get_shot_angle(self):
        self.df = self.df.withColumn("shot_angle",
                                     self.shot_angle_udf(self.df.shot_location_x, self.df.shot_location_y))

    def spatial_data(self):
        self.split_location()
        self.distance_to_goal()
        self.get_shot_angle()

    def preferred_foot(self) -> DataFrame:
        pass_bp = self.df.filter(self.df.type == 'Pass')\
                         .select('player_id', F.col('pass_body_part').alias('body_part'))\
                         .filter(F.col('body_part').isin('Right Foot', 'Left Foot'))

        shot_bp = self.df.filter(self.df.type == 'Shot')\
                         .select('player_id', F.col('shot_body_part').alias('body_part'))\
                         .filter(F.col('body_part').isin('Right Foot', 'Left Foot'))

        bp = pass_bp.union(shot_bp)

        bp_mapped = bp.withColumn('left_foot',
                                  (F.col('body_part')=='Left Foot').cast('int'))\
                      .withColumn('right_foot',
                                  (F.col('body_part')=='Right Foot').cast('int'))\
                      .drop('body_part')

        foot_counts = bp_mapped.groupBy('player_id')\
                               .sum('left_foot', 'right_foot')\
                               .withColumnRenamed('sum(left_foot)', 'left_foot')\
                               .withColumnRenamed('sum(right_foot)', 'right_foot')

        foot_counts = foot_counts.withColumn("total_actions",
                                             F.col("left_foot") + F.col("right_foot"))

        foot_counts = foot_counts.withColumn("preferred_foot",
                                             F.when((F.col("left_foot") / F.col("total_actions")) >= 0.66, "Left Foot")\
                                              .when((F.col("right_foot") / F.col("total_actions")) >= 0.66, "Right Foot")\
                                              .otherwise("Two-Footed"))

        return foot_counts.drop('left_foot', 'right_foot', 'total_actions')

    def shot_preferred_foot(self):
        df = self.preferred_foot()
        df = df.withColumnRenamed('player_id','f_player_id')
        
        self.df = self.df.join(df, self.df.player_id == df.f_player_id, how='left')\
                         .drop('f_player_id')

        self.df = self.df.withColumn('preferred_foot_shot',
                                     F.when((F.col('preferred_foot')=='Two-Footed') &
                                            (F.col('shot_body_part').isin('Right Foot', 'Left Foot')), True)\
                                      .otherwise(F.col('preferred_foot') == F.col('shot_body_part')))

    def shot_freeze_frame(self) -> DataFrame:
        
        return self.df.filter(F.col('shot_type')!='Penalty')\
                      .select('id', 'shot_freeze_frame')\
                      .dropna(subset=['shot_freeze_frame'])

    def shot_frame_df(self) -> pd.DataFrame:
        import ast
        
        df = self.shot_freeze_frame()
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

    @staticmethod
    def check_point_inside(coordinates : list[float],
                           shot_x : float, shot_y : float,
                           GOAL_X : float = 120, GOAL_Y1 : float = 36,
                           GOAL_Y2 : float = 44) -> int:
        
        def calculate_area(x1 : float, y1 : float,
                           x2 : float, y2 : float,
                           x3 : float, y3 : float) -> float:
        
            return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0

        def point_inside_triangle(player_x : float, player_y : float,
                                  shot_x : float, shot_y : float,
                                  GOAL_X : float, GOAL_Y1 : float,
                                  GOAL_Y2 : float) -> bool:
            area_abc = calculate_area(shot_x, shot_y,
                                      GOAL_X, GOAL_Y1,
                                      GOAL_X, GOAL_Y2)
            
            area_abp = calculate_area(shot_x, shot_y,
                                      GOAL_X, GOAL_Y1,
                                      player_x, player_y)
            
            area_bcp = calculate_area(shot_x, shot_y,
                                      GOAL_X, GOAL_Y2,
                                      player_x, player_y)
            
            area_cap = calculate_area(player_x, player_y,
                                      GOAL_X, GOAL_Y2,
                                      GOAL_X, GOAL_Y1)
            
            return np.isclose(area_abc,
                              (area_abp + area_bcp + area_cap),rtol=1e-10)

        count = 0
        for coord in coordinates:
            player_x, player_y = coord
            if point_inside_triangle(player_x, player_y, shot_x, shot_y, GOAL_X, GOAL_Y1, GOAL_Y2):
                count += 1
        return count

    def number_of_players(self):
        frames = self.shot_frame_df()

        frames = (frames.groupby('Shot_id')\
                        .apply(lambda group: group[['x','y']].values.tolist(), include_groups=False)\
                        .reset_index(name='coordinates'))
        
        frames = self.spark.createDataFrame(frames)
        
        frames = frames.join(self.df.select('id','shot_location_x','shot_location_y'),
                             frames.Shot_id == self.df.id)\
                       .drop('id')

        frames = frames.withColumn('players_inside_area',
                                   self.check_point_udf(F.col('coordinates'), F.col('shot_location_x'), F.col('shot_location_y')))

        self.df = self.df.join(frames.select('Shot_id','players_inside_area'), self.df.id == frames.Shot_id, how='left')\
                         .drop('Shot_id')\
                         .withColumn('players_inside_area',
                                     F.when(F.col('shot_type')=='Penalty', 1)\
                         .otherwise(F.col('players_inside_area')))
        
    def create_dummies(self):
        for col_name, mapping in self.DUMMIES_dict.items():
            
            for value, dummy_col in mapping.items():
                
                self.df = self.df.withColumn(dummy_col,
                                             F.when(F.col(col_name) == value, 1)\
                                 .otherwise(0))
                
            self.df = self.df.drop(col_name)
            
        self.df = self.df.withColumn('goal',
                                     F.col('shot_outcome') == 'Goal')\
                         .withColumn('header',
                                     F.when((F.col('shot_body_part')=='Head'), 1)\
                         .otherwise(0))\
                         .drop('shot_body_part')

    def bool_to_int(self):
        for col_name in self.BOOL_TO_INT:
            self.df = self.df.withColumn(col_name,
                                         F.when(F.col(col_name).isNull(), 0)\
                             .otherwise(F.col(col_name).cast('int')))

        self.df = self.df.withColumn('shot_one_on_one',
                                     F.when(F.col('pk_type')==1, 1)\
                                     .otherwise(F.col('shot_one_on_one')))

    def get_assist_data(self):
        pass_events = [ev for ev in self.events if 'pass_' in ev]
        pass_events.remove('pass_body_part')
        df_p = self.df.select(pass_events)
        df_p = df_p.withColumn('pass_height',
                               F.when(F.col('pass_height')=='Ground Pass', 0)\
                                .when(F.col('pass_height')=='Low Pass', 1)
                                .otherwise(2))

        df_p = df_p.withColumnRenamed('under_pressure','pass_under_pressure')

        for col in self.df.columns:
            if col in df_p.columns:
                self.df = self.df.drop(col)

        self.df = self.df.join(df_p, self.df.id == df_p.pass_assisted_shot_id, how='left')

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
        self.spatial_data()
        self.shot_preferred_foot()
        self.number_of_players()
        self.get_assist_data()
        self.create_dummies()
        self.bool_to_int()
        self.df = self.df.filter(self.df.type=='Shot').select(self.variables)

    def data_split(self, train_size : float = 0.8, seed : int = 42) -> tuple[DataFrame,DataFrame]:
        feature_assembler = VectorAssembler(inputCols=self.features,
                                            outputCol="features_vector")
        assembled_data = feature_assembler.transform(self.df)
        train_data, test_data = assembled_data.randomSplit([train_size, 1-train_size], seed=seed)

        return train_data, test_data
