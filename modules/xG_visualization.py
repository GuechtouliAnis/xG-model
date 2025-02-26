import pyspark.sql.functions as F
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mplsoccer import VerticalPitch
from .xG_constants import *

# - data distribution if continuous -> bins else bars
# - feature importance
# - xG distribution
# - ROC-AUC curve
## - my xg vs sb xg

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
        
    # Create another class for post prediction
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

    @staticmethod
    def xGTimeline(predictions : DataFrame,
                   match_id : int,
                   columns : list[str] = CUMULATIVE_XG_COLUMNS):

        # possibility of match_id being None, if so, chose a random match
        
        df = predictions.filter(F.col('match_id') == match_id)

        window_spec = Window.partitionBy('match_id', 'team') \
                            .orderBy('minute', 'second') \
                            .rowsBetween(Window.unboundedPreceding, Window.currentRow)

        df = df.withColumn('sb_CxG', F.sum('shot_statsbomb_xg').over(window_spec)) \
               .withColumn('CxG', F.sum('goal_probability').over(window_spec))

        df_p = df.select(columns).orderBy('minute', 'second').toPandas()
        teams = df_p[~df_p['team'].isna()]['team'].unique()

        i = 91 if df_p['minute'].max() < 90 else df_p['minute'].max() + 1

        mins_range = np.arange(0, i)
        sec_range = np.arange(0, 60)
        ft = pd.DataFrame([(m, s) for m in mins_range for s in sec_range], columns=['minute', 'second'])

        max_sb = max_cxg = 1

        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(14, 7))

        for team in teams:
            df_team = df_p[df_p['team'] == team]
            df_team = ft.merge(df_team, on=('minute', 'second'), how='left')
            df_team['sb_CxG'] = df_team['sb_CxG'].ffill().fillna(0)
            df_team['CxG'] = df_team['CxG'].ffill().fillna(0)
            df_team['goal'] = df_team['goal'].fillna(0).astype(int)
            
            df_team['time'] = df_team['minute'] + round(df_team['second'] / 60, 2)
            
            max_sb = max(max_sb, df_team['sb_CxG'].max())
            max_cxg = max(max_cxg, df_team['CxG'].max())
            
            ax1.plot(df_team['time'], df_team['sb_CxG'], label=team)
            ax1.scatter(df_team[df_team['goal'] == 1]['time'], df_team[df_team['goal'] == 1]['sb_CxG'])
            
            ax2.plot(df_team['time'], df_team['CxG'], label=team)
            ax2.scatter(df_team[df_team['goal'] == 1]['time'], df_team[df_team['goal'] == 1]['CxG'])

        max_y = max(max_sb, max_cxg)

        for ax in [ax1, ax2]:
            ax.axvline(x=45, color='black', linestyle='--')
            ax.set_xticks([15, 30, 45, 60, 75, 90])
            ax.set_xlim(0, i - 1)
            ax.set_ylim(0, max_y + 0.3)
            ax.legend(loc='upper left')
            ax.set_xlabel('Minutes')

        ax1.set_ylabel('statsbomb xG')
        ax1.set_title('Statsbomb xG Cumulative Timeline')
        ax2.set_ylabel('xG')
        ax2.set_title('Predicted xG Cumulative Timeline')

        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def error_dist(predictions : DataFrame,
                actual : str ='shot_statsbomb_xg',
                predicted : str = 'goal_probability',
                bins : int = 20):
        
        predictions = predictions.withColumn(
            "Absolute_Error",
            F.round(F.abs(F.col(actual) - F.col(predicted)),5))
        rmse_pd = predictions.select("Absolute_Error").toPandas()

        plt.hist(rmse_pd["Absolute_Error"], bins=bins, edgecolor="black")
        plt.xlabel("Absolute Error")
        plt.ylabel("Frequency")
        plt.title("Distribution of Absolute Error")
        plt.show()

    @staticmethod
    def GxG_scatter(predictions : DataFrame):

        GxG = predictions.groupBy("player",'team')\
            .agg(F.sum("goal").alias("goals"),
                    F.round(F.sum("shot_statsbomb_xg"),3).alias("xG"))\
            .filter((F.col('goals')>1) & (F.col('xG')>1))\
            .withColumn('G-xG', F.round(F.col('goals') - F.col('xG'),5))\
            .toPandas()

        GxG['Performance'] = np.where(GxG['G-xG'] < 0, 'Underperformer', 'Overperformer')

        fig = px.scatter(
            GxG,
            x="goals",
            y="xG",
            color="Performance",
            hover_data=["player"],
            title="Goals vs. xG Scatter Plot",
            labels={"player": "Player", "xG": "xG", "goals": "Goals"},
            color_discrete_map={'Underperformer': 'red',
                                'Overperformer': 'green'})

        fig.update_layout(
            height=700,
            width=700,
            xaxis_range=[0,GxG['goals'].max()+2],
            yaxis_range=[0,GxG['xG'].max()+2])

        fig.show()