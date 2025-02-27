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

class Visualization:
    def __init__(self,
                 data : object,
                 features : list[str] = FEATURES,
                 include_target : bool = True):
        
        """
        Initialize a Visualization object for analyzing and plotting shot data.

        This constructor extracts the primary Spark DataFrame and feature list from the provided
        data object. If the 'include_target' flag is set to True, target columns (i.e., 'shot_statsbomb_xg'
        and 'goal') are appended to the feature list if they are not already included. Additionally,
        if the input data object contains shot frame data (as an attribute named 'shot_frame'), it is
        assigned to the instance; otherwise, the 'shot_frame_df' attribute is set to None.

        Parameters
        ----------
        data : object
            An object that must contain a Spark DataFrame accessible via the attribute 'df'. Optionally,
            it may also have a 'shot_frame' attribute for shot freeze frame data.
        features : list[str], optional
            A list of feature column names to be used for visualization. Defaults to the FEATURES constant.
        include_target : bool, optional
            Flag indicating whether to include target columns (['shot_statsbomb_xg', 'goal']) in the
            feature list if they are not already present. Default is True.

        Attributes
        ----------
        df : DataFrame
            The Spark DataFrame extracted from the input data object.
        features : list[str]
            The list of feature column names used for visualization, extended with target columns
            if 'include_target' is True.
        include_target : bool
            Indicates if target columns have been added to the feature list.
        shot_frame_df : DataFrame or None
            Shot freeze frame data if available in the input data object; otherwise, None.
        """
        
        self.df = data.df
        self.features = features.copy()
        self.include_target = include_target
        
        target = ['shot_statsbomb_xg', 'goal']
        if self.include_target:
            self.features.extend(t for t in target if t not in self.features)

        if hasattr(data, 'shot_frame'):
            self.shot_frame_df = data.shot_frame
        else:
            self.shot_frame_df = None
            
    def correlation(self,
                    features : list[str] = None):
        
        """
        Compute and visualize the correlation matrix for the specified feature columns.

        This method assembles the provided feature columns into a single vector using Spark's 
        VectorAssembler and computes their correlation matrix with pyspark.ml.stat.Correlation. 
        The resulting correlation matrix is converted to a numpy array and visualized as a heatmap 
        using seaborn and matplotlib. If no feature list is provided, the method defaults to using 
        the object's 'features' attribute.

        Parameters
        ----------
        features : list[str], optional
            A list of feature column names for which to compute the correlation matrix. 
            Defaults to None, in which case self.features is used.

        Returns
        -------
        None
            This method displays a heatmap of the correlation matrix and does not return any value.
        """
        
        if features is None:
            features = self.features

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
        
    def shot_frame(self,
                   shot_id : str,
                   show_angle : bool = False,
                   show_players : bool = True,
                   show_info : bool = True):
        
        """
        Visualize a shot event's spatial details and freeze frame on a football pitch.

        This method retrieves details for the specified shot event using the provided shot_id
        from the primary DataFrame. It then uses the associated shot freeze frame data (if available)
        to plot the shot's starting and ending locations, draw the shot's trajectory, and optionally
        overlay the goal angle and positions of players (teammates and opponents) on a football pitch
        using mplsoccer's VerticalPitch. Additionally, key shot metrics such as xG, distance to goal,
        shot angle, number of players inside the shot area, and shot outcome are displayed as a legend.

        Parameters
        ----------
        shot_id : str
            The unique identifier of the shot event to be visualized.
        show_angle : bool, optional
            Flag to indicate whether the goal angle should be overlaid on the pitch. Default is False.
        show_players : bool, optional
            Flag to indicate whether to plot the positions of players (teammates and opponents) from the freeze frame data.
            Default is True.
        show_info : bool, optional
            Flag to indicate whether to display shot information (e.g., xG, distance, angle, players in area, outcome)
            as a legend on the plot. Default is True.

        Raises
        ------
        ValueError
            If the 'shot_frame_df' attribute is not available in the object, indicating that freeze frame data is missing.

        Returns
        -------
        None
            Displays the plot without returning any value.
        """
        
        if self.shot_frame_df is None:
            raise ValueError('Object has no attribute shot_frame_df')
        
        row = self.df.filter(self.df.id == shot_id).collect()[0]
        color = '#ad993c' if row['goal'] == 1 else '#ba4f45'        
        shot_data = self.shot_frame_df[self.shot_frame_df['Shot_id'] == row.id]

        fig, ax = plt.subplots(1, 1, figsize=(18, 6))
        pitch = VerticalPitch(pad_bottom=0.5,
                              half=True,
                              corner_arcs=True,
                              goal_type='box',
                              pitch_type='statsbomb')
        pitch.draw(ax=ax)

        pitch.scatter(row['shot_location_x'], row['shot_location_y'],
                      color='orange',
                      ax=ax,
                      zorder=4)
        
        pitch.arrows(row['shot_location_x'], row['shot_location_y'],
                     row['shot_end_x'], row['shot_end_y'],
                     headwidth=3,
                     headlength=2,
                     width = 2,
                     color=color,
                     ax=ax)
        
        if show_angle:            
            pitch.goal_angle(x=row['shot_location_x'], y=row['shot_location_y'],
                             goal='right',
                             color='blue',
                             alpha=0.3,
                             zorder=1,
                             ax=ax)
            
        if show_players:
            teammates = shot_data[shot_data['teammate'] == 'True']
            opponents = shot_data[shot_data['teammate'] != 'True']
            
            pitch.scatter(teammates['x'], teammates['y'],
                          color='green',
                          s=30,
                          zorder=2,
                          ax=ax)
            pitch.scatter(opponents['x'], opponents['y'],
                          color='red',
                          s=30,
                          zorder=2,
                          ax=ax)

        if show_info:
            shot_info = [f"xG: {row['shot_statsbomb_xg']:.2f}",
                         f"Distance: {row['distance_to_goal']:.2f}",
                         f"Angle: {row['shot_angle']:.2f}°",
                         f"Players in area: {row['players_inside_area']}",
                         f"Outcome: {row['shot_outcome']}"]
            
            legend_text = "\n".join(shot_info)
            legend_handle = mlines.Line2D([], [],
                                          color='none',
                                          label=legend_text)

            ax.legend(handles=[legend_handle],
                      loc='upper left',
                      handlelength=0,
                      handletextpad=0,
                      frameon=True,
                      borderpad=0.5,
                      labelspacing=0.3)
            
        plt.show()
        
    def shot_goal_heatmap(self,
                          x : str = 'shot_location_x',
                          y : str ='shot_location_y',
                          target : str = 'goal'):
        
        """
        Generate side-by-side heatmaps comparing the spatial distribution of all shots and goals.

        This method converts the relevant columns from the Spark DataFrame to a pandas DataFrame and uses
        mplsoccer's VerticalPitch to create two heatmaps:
        - One heatmap displays the distribution of all shot events.
        - The other heatmap shows the distribution of goals (where the target column equals 1).

        Parameters
        ----------
        x : str, optional
            The name of the column representing the x-coordinate of the shot location. Default is 'shot_location_x'.
        y : str, optional
            The name of the column representing the y-coordinate of the shot location. Default is 'shot_location_y'.
        target : str, optional
            The name of the column indicating the target variable (e.g., goal outcome). Default is 'goal'.

        Returns
        -------
        None
            The method displays the heatmaps and does not return any value.
        """

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

        pcm1 = pitch.heatmap(bin_statistic_shots,
                             ax=ax1,
                             cmap='Reds',
                             edgecolor='white',
                             linewidth=0.01)
        pcm2 = pitch.heatmap(bin_statistic_goals,
                             ax=ax2,
                             cmap='Reds',
                             edgecolor='white',
                             linewidth=0.01)

        ax_cbar1 = fig.add_axes([0.46, 0.09, 0.02, 0.8])
        plt.colorbar(pcm1, cax=ax_cbar1)

        ax_cbar2 = fig.add_axes([0.88, 0.09, 0.02, 0.8])
        plt.colorbar(pcm2, cax=ax_cbar2)

        ax1.set_title("Shots Heatmap")
        ax2.set_title("Goals Heatmap")
        
        fig.suptitle("Comparison of Shots and Goals Heatmaps", fontsize=16)
        plt.show()

    def confusion_matrix(self,
                         actual : str = 'goal',
                         predicted : str = 'prediction',
                         cmap : str = 'Reds'):
        
        """
        Compute and visualize a confusion matrix based on actual and predicted values.

        This method creates a confusion matrix by performing a crosstab on the specified actual and predicted
        columns from the Spark DataFrame. The resulting matrix is then converted to a pandas DataFrame and visualized
        as a heatmap using seaborn. The heatmap includes annotations for the count of instances, with axis labels set to
        "Predicted" and "Actual" and an overall title "Confusion Matrix".

        Parameters
        ----------
        actual : str, optional
            The column name representing the true labels (default is 'goal').
        predicted : str, optional
            The column name representing the predicted labels (default is 'prediction').
        cmap : str, optional
            The colormap used for the heatmap visualization (default is 'Reds').

        Returns
        -------
        None
            The method displays the confusion matrix heatmap and does not return any value.
        """

        conf = self.df.crosstab(actual, predicted)

        conf_pd = conf.toPandas().set_index(actual + '_' + predicted)

        conf_pd.columns = conf_pd.columns.astype(int)

        sns.heatmap(conf_pd,
                    annot=True,
                    fmt="d",
                    cmap=cmap,
                    vmin=0)

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    def xG_timeline(self,
                    match_id : int,
                    columns : list[str] = CUMULATIVE_XG_COLUMNS):
        
        """
        Plot cumulative xG timelines for a given match.

        This method filters the shot event data for the specified match_id and computes cumulative sums for both
        Statsbomb xG and predicted xG using a window function partitioned by match and team, ordered by minute and second.
        The cumulative values are merged with a complete timeline of minutes and seconds to fill in any gaps via forward
        filling. For each team, the method plots two subplots: one for the cumulative Statsbomb xG and another for the
        cumulative predicted xG, highlighting moments when goals occurred. A vertical line at minute 45 is added to both
        plots to denote halftime.

        Parameters
        ----------
        match_id : int
            The unique identifier for the match to be visualized.
        columns : list[str], optional
            A list of column names used in the timeline visualization, defaulting to CUMULATIVE_XG_COLUMNS.

        Returns
        -------
        None
            This method displays the cumulative xG timeline plots and does not return any value.
        """

        df = self.df.filter(F.col('match_id') == match_id)

        window_spec = Window.partitionBy('match_id', 'team') \
                            .orderBy('minute', 'second') \
                            .rowsBetween(Window.unboundedPreceding,
                                         Window.currentRow)

        df = df.withColumn('sb_CxG',
                           F.sum('shot_statsbomb_xg').over(window_spec)) \
               .withColumn('CxG',
                           F.sum('xG').over(window_spec))

        df_p = df.select(columns).orderBy('minute', 'second').toPandas()

        teams = df_p[~df_p['team'].isna()]['team'].unique()

        i = 91 if df_p['minute'].max() < 90 else df_p['minute'].max() + 1

        mins_range = np.arange(0, i)
        sec_range = np.arange(0, 60)

        ft = pd.DataFrame([(m, s) for m in mins_range for s in sec_range], columns=['minute', 'second'])

        max_sb = max_cxg = 1

        fig, (ax1, ax2) = plt.subplots(ncols=2,
                                       nrows=1,
                                       figsize=(14, 7))

        for team in teams:
            df_team = df_p[df_p['team'] == team]

            df_team = ft.merge(df_team,
                               on=('minute', 'second'),
                               how='left')

            df_team['sb_CxG'] = df_team['sb_CxG'].ffill().fillna(0)
            df_team['CxG'] = df_team['CxG'].ffill().fillna(0)
            df_team['goal'] = df_team['goal'].fillna(0).astype(int)
            
            df_team['time'] = df_team['minute'] + round(df_team['second'] / 60, 2)
            
            max_sb = max(max_sb, df_team['sb_CxG'].max())
            max_cxg = max(max_cxg, df_team['CxG'].max())
            
            ax1.plot(df_team['time'],
                     df_team['sb_CxG'],
                     label=team)
            ax1.scatter(df_team[df_team['goal'] == 1]['time'],
                        df_team[df_team['goal'] == 1]['sb_CxG'])
            
            ax2.plot(df_team['time'],
                     df_team['CxG'],
                     label=team)
            ax2.scatter(df_team[df_team['goal'] == 1]['time'],
                        df_team[df_team['goal'] == 1]['CxG'])

        max_y = max(max_sb, max_cxg)

        for ax in [ax1, ax2]:
            ax.axvline(x=45,
                       color='black',
                       linestyle='--')
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

    def error_dist(self,
                   actual : str ='shot_statsbomb_xg',
                   predicted : str = 'xG',
                   bins : int = 20):
        
        """
        Plot the distribution of absolute errors between actual and predicted xG values.

        This method calculates the absolute error by subtracting the predicted xG from the actual xG values,
        rounding the result to five decimal places. It then converts the computed absolute errors to a pandas
        DataFrame and plots a histogram to visualize the frequency distribution of these errors.

        Parameters
        ----------
        actual : str, optional
            The column name for the actual xG values (default is 'shot_statsbomb_xg').
        predicted : str, optional
            The column name for the predicted xG values (default is 'xG').
        bins : int, optional
            The number of bins to use in the histogram (default is 20).

        Returns
        -------
        None
            The function displays the histogram plot and does not return any value.
        """

        df = self.df.withColumn(
            "Absolute_Error",
            F.round(F.abs(F.col(actual) - F.col(predicted)),5))
        
        rmse_pd = df.select("Absolute_Error").toPandas()

        plt.hist(rmse_pd["Absolute_Error"],
                 bins=bins,
                 edgecolor="black")
        
        plt.xlabel("Absolute Error")
        plt.ylabel("Frequency")
        plt.title("Distribution of Absolute Error")
        plt.show()

    def gxg_scatter(self,
                    xg_column : str = 'shot_statsbomb_xg',
                    goal_column : str = 'goal',
                    min_goal : int = 1,
                    min_xg : float = 1):
        
        """
        Generate an interactive scatter plot comparing goals and xG for players.

        This method aggregates shot event data by player and team from the underlying Spark DataFrame,
        summing goals and xG values. It filters out players whose cumulative goals or xG do not exceed the
        specified minimum thresholds (min_goal and min_xg) and computes the difference between goals and xG ('G-xG').
        Players are then classified as 'Underperformer' if the 'G-xG' value is negative and 'Overperformer' otherwise.
        The method creates an interactive scatter plot using Plotly Express, plotting goals on the x-axis and xG on
        the y-axis, with points colored by the performance classification. Hover data displays the player's name.

        Parameters
        ----------
        xg_column : str, optional
            The column name for xG values (default is 'shot_statsbomb_xg').
        goal_column : str, optional
            The column name for goal outcomes (default is 'goal').
        min_goal : int, optional
            The minimum number of goals required for a player to be included in the plot (default is 1).
        min_xg : float, optional
            The minimum cumulative xG required for a player to be included in the plot (default is 1).

        Returns
        -------
        None
            The function displays an interactive scatter plot and does not return any value.
        """

        GxG = self.df.groupBy("player",'team')\
            .agg(F.sum(goal_column).alias("goals"),
                 F.round(
                     F.sum(xg_column),3).alias("xG"))\
            .filter((F.col('goals') > min_goal) & (F.col('xG') > min_xg))\
            .withColumn('G-xG',
                        F.round(F.col('goals') - F.col('xG'),5))\
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
    
    def shot_distribution(self,
                          columns : list[str] = SHOT_DIST_COLUMNS,
                          val : str = 'shot_statsbomb_xg',
                          t : str = 'xg',
                          size : int = 5,
                          team_id : int | None = None,
                          match_id : int | None = None,
                          player_id : int | None = None,
                          sample : int | None = None):
        
        """
        Plot the spatial distribution of shots on a football pitch based on xG or goal outcomes.

        This method filters the shot data from the Spark DataFrame according to specified criteria such as team_id, 
        match_id, and player_id. Optionally, a random sample of the data can be selected. The filtered data is converted 
        to a pandas DataFrame and visualized on a football pitch using mplsoccer's VerticalPitch. Depending on the type 
        parameter (t), the method plots either the xG distribution (using predefined bins and colors) or the goal distribution. 
        A legend is generated to explain the color coding of the plotted points.

        Parameters
        ----------
        columns : list[str], optional
            List of column names to select from the DataFrame for plotting. Defaults to SHOT_DIST_COLUMNS.
        val : str, optional
            The column name used for determining the distribution values (e.g., 'shot_statsbomb_xg'). Default is 'shot_statsbomb_xg'.
        t : str, optional
            The type of distribution to visualize: 'xg' for xG distribution or 'goal' for goal distribution. Default is 'xg'.
        size : int, optional
            Marker size for the scatter plot points. Default is 5.
        team_id : int or None, optional
            If provided, filters the data to include only shots by the specified team. Default is None.
        match_id : int or None, optional
            If provided, filters the data to include only shots from the specified match. Default is None.
        player_id : int or None, optional
            If provided, filters the data to include only shots taken by the specified player. Default is None.
        sample : int or None, optional
            If provided, randomly selects the specified number of shot records for plotting. Default is None.

        Raises
        ------
        ValueError
            If no shot data is available after applying the filter criteria.

        Returns
        -------
        None
            The method displays a plot of the shot distribution on a football pitch and does not return any value.
        """

        df = self.df

        if team_id is not None:
            df = df.filter(df.team_id == team_id)

        if match_id is not None:
            df = df.filter(df.match_id  == match_id)
            
        if player_id is not None:
            df = df.filter(df.player_id == player_id)
        
        if sample is not None:
            df = df.orderBy(F.rand()).limit(sample)

        if df.limit(1).count() == 0:
            raise ValueError("No shot data available for the given filter criteria.")

        df = df.select(columns).toPandas()

        fig, ax = plt.subplots(1, 1, figsize=(18, 6))
        pitch = VerticalPitch(pad_bottom=0.5,
                              half=True,
                              corner_arcs=True,
                              goal_type='box',
                              pitch_type='statsbomb')
        pitch.draw(ax=ax)

        if t =='xg':
            colors = {(0.00, 0.25): '#fcc4ad',
                      (0.25, 0.50): '#fb694a',
                      (0.50, 0.75): '#b11218',
                      (0.75, 1.00): '#67000d'}
            title = 'xG Distribution'
        elif t == 'goal':
            colors = {0 : '#fb694a',
                      1 : '#67000d'}
            title = 'Goal Distribution'
        else:
            raise ValueError("Unknown type. Choose from ['xg', 'goal']")

        legend_elements = []
        for key, value in colors.items():

            if t == 'xg':
                df_1 = df[df[val].between(key[0], key[1])]
                label = f"{key[0]:.2f} - {key[1]:.2f}"
            else:
                df_1 = df[df[val] == key]
                label = "Goal" if key == 1 else "No goal"
                
            pitch.scatter(df_1['shot_location_x'],
                          df_1['shot_location_y'],
                          ax=ax,
                          s=size,
                          c=value)

            handle = mlines.Line2D([], [],
                                    marker='o',
                                    color='w',
                                    markerfacecolor=value,
                                    markersize=8,
                                    label=label)

            legend_elements.append(handle)

        ax.legend(handles=legend_elements,
                  title=title,
                  loc='lower right')
        plt.show()
