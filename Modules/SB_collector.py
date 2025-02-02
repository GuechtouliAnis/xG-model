import warnings
import pandas as pd
import numpy as np
from statsbombpy import sb
from statsbombpy.api_client import NoAuthWarning

warnings.filterwarnings("ignore", category=NoAuthWarning)

class Scraper:
    def __init__(self):
        pass
    
    def competitions(self,
                     comps=None,
                     start=None,
                     end=None,
                     to_drop=None,
                     gender=None,
                     international=None,
                     save_return=None):
        
        self.competition = sb.competitions()
    
        # try, if no access return error
    
    def matches(self):
        pass
    
    def lineups(self):
        pass
    
    def events(self):
        pass
    
    def frames(self):
        pass



