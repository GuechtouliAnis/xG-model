import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
from .xG_constants import *

# - data distribution (bins)
# - shot map, using shot_freeze_frame
# - xG heat map, shots heatmap
# - corr

# - feature importance
# - xG distribution
# - xG vs actual goal scatter
# - ROC-AUC curve
# - interactive vis
## - my xg vs sb xg
## - learning curve
## - conf matrix

class Visualisations:
    def __init__(self,
                 spark : SparkSession,
                 df : DataFrame):
        
        self.spark = spark
        self.df = df
        
    