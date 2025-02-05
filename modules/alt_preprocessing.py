import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import DataFrame

FEATURES = []
EVENTS = [
    'id','period','duration','location','player_id','position', # Event info
    'play_pattern','shot_body_part','shot_technique','shot_type', # Shot info
    'shot_freeze_frame', 'shot_key_pass_id', # Complicated info
    'under_pressure','shot_aerial_won','shot_first_time','shot_one_on_one',
    'shot_open_goal', 'shot_follows_dribble', # Boolean
    'shot_statsbomb_xg','shot_outcome',# Target
    'pass_body_part','type']

class Preprocessing:
    def __init__(self,
                 df : DataFrame,
                 EVENTS=EVENTS,
                 full_pp=True,
                 GOAL_X=120,
                 GOAL_Y1=36,
                 GOAL_Y2=44):
        
        self.df = df.filter((df.type =='Shot') | (df.type=='Pass')).select(EVENTS)
        self.full_pp = full_pp
        self.GOAL_X = GOAL_X
        self.GOAL_Y1 = GOAL_Y1
        self.GOAL_Y2 = GOAL_Y2
        
        self.shot_angle_udf = F.udf(Preprocessing.shot_angle, T.FloatType())
        
        if self.full_pp:
            self.df == self.preprocess()
           
    def __getattr__(self, attr):
        """Forward calls to self.df"""
        return getattr(self.df, attr)
    
    @staticmethod
    def shot_angle(shot_x, shot_y):
        import math
        u_x = 120 - shot_x
        u_y = 36 - shot_y
        v_y = 44 - shot_y
        dot_product = u_x ** 2 + u_y * v_y
        magnitude_u = math.sqrt(u_x **2 + u_y ** 2)
        magnitude_v = math.sqrt(u_x **2 + v_y ** 2)
        if magnitude_u == 0 or magnitude_v == 0:
            return 0.0
        angle_radians = math.acos(dot_product / (magnitude_u * magnitude_v))
        return math.degrees(angle_radians)

    def split_location(self):
        self.df = self.df.withColumn("shot_location_x", F.regexp_extract(F.col("location"), r'\[(.*?),', 1).cast("float")) \
             .withColumn("shot_location_y", F.regexp_extract(F.col("location"), r', (.*?)\]', 1).cast("float")).drop('location')
    
    def distance_to_goal(self):
        self.df = self.df.withColumn("distance_to_goal",
                                     F.round(F.sqrt(F.pow(self.df.shot_location_x - F.lit(self.GOAL_X), 2) +
                                                    F.pow(self.df.shot_location_y - F.lit(self.GOAL_Y1 + self.GOAL_Y2)/2, 2)), 2))
    
    def get_shot_angle(self):        
        self.df = self.df.withColumn("shot_angle",
                                     self.shot_angle_udf(
                                         self.df.shot_location_x, self.df.shot_location_y))
        
    def spatial_data(self):
        self.split_location()
        self.distance_to_goal()
        self.get_shot_angle()

    def preferred_foot(self):
        pass_bp = self.df.filter(self.df.type == 'Pass')\
            .select('player_id', F.col('pass_body_part').alias('body_part'))\
                .filter(F.col('body_part').isin('Right Foot', 'Left Foot'))

        shot_bp = self.df.filter(self.df.type == 'Shot')\
            .select('player_id', F.col('pass_body_part').alias('body_part'))\
                .filter(F.col('body_part').isin('Right Foot', 'Left Foot'))
        
        bp = pass_bp.union(shot_bp)

        bp_mapped = bp.withColumn(
            'left_foot',
            (F.col('body_part')=='Left Foot').cast('int'))\
                .withColumn(
                    'right_foot',
                    (F.col('body_part')=='Right Foot').cast('int'))\
                        .drop('body_part')
                        
        foot_counts = bp_mapped.groupBy('player_id')\
            .sum('left_foot', 'right_foot')\
            .withColumnRenamed('sum(left_foot)', 'left_foot')\
            .withColumnRenamed('sum(right_foot)', 'right_foot')

        foot_counts = foot_counts.withColumn(
            "total_actions",
            F.col("left_foot") + F.col("right_foot"))

        foot_counts = foot_counts.withColumn(
            "preferred_foot",
            F.when((F.col("left_foot") / F.col("total_actions")) >= 0.66, "Left Foot")
            .when((F.col("right_foot") / F.col("total_actions")) >= 0.66, "Right Foot")
            .otherwise("Two-Footed"))
        
        return foot_counts.drop('left_foot', 'right_foot', 'total_actions')
    
    def shot_preferred_foot(self):
        df = self.preferred_foot()
        self.df = self.df.join(df, self.df.player_id == df.player_id, how='left')\
            .drop(df.player_id)
            
        self.df = self.df.withColumn(
            'preferred_foot_shot',
            F.when(
                (F.col('preferred_foot')=='Two-Footed') &
                (F.col('shot_body_part').isin('Right Foot', 'Left Foot')), True
            ).otherwise(F.col('preferred_foot') == F.col('shot_body_part')))

    def preprocess(self):
        self.spatial_data()
        self.shot_preferred_foot()
        return self.df