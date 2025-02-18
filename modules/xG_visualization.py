import pyspark.sql.functions as F
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.feature import VectorAssembler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mplsoccer import VerticalPitch
from .xG_constants import *

# - data distribution if continuous -> bins else bars
# - feature importance
# - xG distribution
# - ROC-AUC curve
## - my xg vs sb xg
## - RMSE distribution
# - Match xG evolution vs final result (Mine vs sb)

class Visualization:
    def __init__(self,
                 data : object,
                 features : list[str] = FEATURES,
                 include_target : bool = True):
        
        self.df = data.df
        self.shot_frame = data.shot_frame
        self.features = features.copy()
        self.include_target = include_target
        
        if self.include_target:
            self.features.extend(['shot_statsbomb_xg','goal'])
            
    def Correlation(self,
                    features : list[str] = None):
        if features is None:
            features = self.features
        else:
            features = features
        assembler = VectorAssembler(inputCols=features,
                                    outputCol="features")
        df_vec = assembler.transform(self.df.select(*features))
        correlation_matrix = Correlation.corr(df_vec,
                                              "features").head()[0]
        corr_matrix = correlation_matrix.toArray()

        num_features = len(features)
        fig_size = max(8, num_features * 0.6)
        
        plt.figure(figsize=(fig_size, fig_size))
        sns.heatmap(corr_matrix,
                    annot=True,
                    cmap="coolwarm",
                    fmt=".2f",
                    xticklabels=features,
                    yticklabels=features,
                    vmax=1,
                    vmin=-1)
        plt.title("Correlation Matrix")
        plt.show()
            
    def ShotFrame(self,
                  shot_id : str,
                  show_angle : bool = False,
                  show_players : bool = True,
                  show_info : bool = True):
        
        row = self.df.filter(self.df.id == shot_id).collect()[0]
        color = '#ad993c' if row['goal'] == 1 else '#ba4f45'        
        shot_data = self.shot_frame[self.shot_frame['Shot_id'] == row.id]

        fig, ax = plt.subplots(1, 1, figsize=(18, 6))
        pitch = VerticalPitch(pad_bottom=0.5, half=True, corner_arcs=True,
                              goal_type='box', pitch_type='statsbomb')
        pitch.draw(ax=ax)

        pitch.scatter(row['shot_location_x'], row['shot_location_y'],
                      color='orange', ax=ax, zorder=4)
        pitch.arrows(row['shot_location_x'], row['shot_location_y'],
                     row['shot_end_x'], row['shot_end_y'],
                     headwidth=3, headlength=2, width = 2,
                     color=color, ax=ax)
        
        if show_angle:            
            pitch.goal_angle(x=row['shot_location_x'], y=row['shot_location_y'],
                             goal='right', color='blue', alpha=0.3, zorder=1, ax=ax)
            
        if show_players:
            teammates = shot_data[shot_data['teammate'] == 'True']
            opponents = shot_data[shot_data['teammate'] != 'True']
            
            pitch.scatter(teammates['x'], teammates['y'],
                          color='green', s=30, zorder=2, ax=ax)
            pitch.scatter(opponents['x'], opponents['y'],
                          color='red', s=30, zorder=2, ax=ax)

        if show_info:
            shot_info = [f"xG: {row['shot_statsbomb_xg']:.2f}",
                         f"Distance: {row['distance_to_goal']:.2f}",
                         f"Angle: {row['shot_angle']:.2f}°",
                         f"Players in area: {row['players_inside_area']}",
                         f"Outcome: {row['shot_outcome']}"]
            
            legend_text = "\n".join(shot_info)
            legend_handle = mlines.Line2D([], [], color='none', label=legend_text)
            ax.legend(handles=[legend_handle], loc='upper left',
                    handlelength=0, handletextpad=0, frameon=True,
                    borderpad=0.5, labelspacing=0.3)
            
        plt.show()
        
    def ConfusionMatrix(self,
                        actual : str = 'goal',
                        predicted : str = 'prediction',
                        cmap : str = 'Reds'):
        
        conf = self.df.crosstab(actual, predicted)    
        conf_pd = conf.toPandas().set_index(actual+'_'+predicted)
        conf_pd.columns = conf_pd.columns.astype(int)

        sns.heatmap(conf_pd, annot=True, fmt="d", cmap=cmap, vmin=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
        
    def ShotGoalHeatMap(self,
                        x : str = 'shot_location_x',
                        y : str ='shot_location_y',
                        target : str = 'goal'):
        
        df = self.df.select(x,y,target).toPandas()
        pitch = VerticalPitch(line_color='black', half=True, pitch_type='statsbomb', line_zorder=2)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        pitch.draw(ax=ax1)
        pitch.draw(ax=ax2)

        bin_statistic_shots = pitch.bin_statistic(df[x],
                                                  df[y],
                                                  bins=50)
        bin_statistic_goals = pitch.bin_statistic(df[df[target] == 1][x],
                                                  df[df[target] == 1][y],
                                                  bins=50)

        pcm1 = pitch.heatmap(bin_statistic_shots, ax=ax1, cmap='Reds', edgecolor='white', linewidth=0.01)
        pcm2 = pitch.heatmap(bin_statistic_goals, ax=ax2, cmap='Reds', edgecolor='white', linewidth=0.01)

        ax_cbar1 = fig.add_axes([0.46, 0.09, 0.02, 0.8])
        plt.colorbar(pcm1, cax=ax_cbar1)

        ax_cbar2 = fig.add_axes([0.88, 0.09, 0.02, 0.8])
        plt.colorbar(pcm2, cax=ax_cbar2)

        ax1.set_title("Shots Heatmap")
        ax2.set_title("Goals Heatmap")
        
        fig.suptitle("Comparison of Shots and Goals Heatmaps", fontsize=16)
        plt.show()