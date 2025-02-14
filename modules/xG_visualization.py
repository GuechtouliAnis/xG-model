import pyspark.sql.functions as F
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.feature import VectorAssembler
import pyspark
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
from .xG_constants import *

# - data distribution (bins)
# - xG heat map, shots heatmap

# - feature importance
# - xG distribution
# - xG vs actual goal scatter
# - ROC-AUC curve
# - interactive vis
## - my xg vs sb xg
## - learning curve
## - conf matrix

class Visualization:
    def __init__(self,
                 data : object,
                 features : list[str] = FEATURES,
                 include_target : bool = True):
        
        self.df = data.df.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
        self.df.foreach(lambda row: None)
        self.shot_frame = data.shot_frame
        self.features = features.copy()
        self.include_target = include_target
        
        if self.include_target:
            self.features.extend(['shot_statsbomb_xg','goal'])
            
    def Correlation(self):
        assembler = VectorAssembler(inputCols=self.features,
                                    outputCol="features")
        df_vec = assembler.transform(self.df.select(*self.features))
        correlation_matrix = Correlation.corr(df_vec,
                                              "features").head()[0]
        corr_matrix = correlation_matrix.toArray()

        plt.figure(figsize=(30, 30))
        sns.heatmap(corr_matrix,
                    annot=True,
                    cmap="coolwarm",
                    fmt=".2f",
                    xticklabels=self.features,
                    yticklabels=self.features,
                    vmax=1,
                    vmin=-1)
        plt.title("Correlation Matrix")
        plt.show()
            
    def ShotFrame(self, shot_id):
        row = self.df.filter(self.df.id == shot_id).collect()[0]
        
        shot_data = self.shot_frame[self.shot_frame['Shot_id'] == row.id]
        teammates = shot_data[shot_data['teammate'] == 'True']
        opponents = shot_data[shot_data['teammate'] != 'True']

        fig, ax = plt.subplots(1, 1, figsize=(18, 6))

        pitch = VerticalPitch(pad_bottom=0.5, half=True, corner_arcs=True, goal_type='box', pitch_type='statsbomb')
        pitch.draw(ax=ax)

        pitch.goal_angle(x=row['shot_location_x'], y=row['shot_location_y'], goal='right',
                        color='blue', alpha=0.3, zorder=1, ax=ax)
        pitch.scatter(teammates['x'], teammates['y'], color='green', s=30, label='teammate', zorder=2, ax=ax)
        pitch.scatter(opponents['x'], opponents['y'], color='red', s=30, label='opponent', zorder=2, ax=ax)
        pitch.scatter(row['shot_location_x'], row['shot_location_y'], color='orange', ax=ax)
        # ax.legend()
        plt.show()
