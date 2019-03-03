#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Needed modules :

## Standard :
# Full imports :
import os
import gc
import copy
import sklearn

# Aliased imports :
import numpy as np
import pandas as pd
import seaborn as sb
import datetime as dt
import matplotlib.pyplot as plt

# Partial Imports :
from sys import getsizeof
from typing import List, Tuple
from scipy import stats
from functools import reduce

## User-defined :
import Preprocessing as pre
import Drive


# In[2]:


sb.set_style("darkgrid")


# In[3]:


# Function definitions :

def data_event_filter(df: pd.core.frame.DataFrame, 
                      undesired_columns: list) -> pd.core.frame.DataFrame:
    """
    """
    _tmp = copy.deepcopy(df)
    _tmp = _tmp.drop(undesired_columns, axis=1)
    _tmp = _tmp.loc[ (_tmp['type'] == 'data') | (_tmp['type'] == 'event') ]
    
    return _tmp
    
def time_indexed_df(df1: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """ Take a return a time-indexed dataframe.
    df1 paramater should contain a column called 'dateTime',
    which contains entries of type pandas._libs.tslibs.timestamps.Timestamp
    """
    _tmp = copy.deepcopy(df1)
    _tmp.index = df1.dateTime
    _tmp.drop('dateTime', axis=1, inplace=True)
    _tmp = _tmp.sort_index()
    gc.collect()
    return _tmp


def fill_nas(df1: pd.core.frame.DataFrame, 
             col_names: List[str] = ['activeInsulin', 'carbs', 'insulin', 'trend'],
             fill_with=0) -> pd.core.frame.DataFrame:
    """  Return a new dataframe, replacing all occurrencies of NaNs with 
    the parameter 'fill_with'.
    """
    _tmp = copy.deepcopy(df1)
    for name in col_names:
        _tmp[name] = df1[name].fillna(0)
    gc.collect()
    return _tmp


def classify(value: float, limits: Tuple[float, float] = (70, 140)) -> str:
    """ value: Numerical value
        limits: Tuple (lower_boundary, high_boundary)
        
        Returns:
            'hypo'  if value < lower_boundary
            'normo' if lower_boundary <= value <= high_boundary
            'hyper' if value > high_boundary
    """
    if value < limits[0]:
        return 'hypo'
    elif value > limits[1]:
        return 'hyper'
    else:
        return 'normo'


def set_hour(df:  pd.core.frame.DataFrame) ->  pd.core.frame.DataFrame:
    """ Return a copy of 'df', 
    adding a column called 'hour'.
    The dataframe 'df' should be time-indexed.
    """
    _tmp = copy.deepcopy(df)
    _tmp.loc[:,'hour'] = list(
                             map(
                                lambda x: x.hour, df.index
                                )
                             )
    gc.collect()
    return _tmp
        

def set_postprandial(df:  pd.core.frame.DataFrame) ->  pd.core.frame.DataFrame:
    """ Return a copy of 'df', 
    adding a column called 'hour'.
    The dataframe 'df' should be time-indexed.
    """
    _tmp = copy.deepcopy(df)
    _tmp.loc[:,'postprandial'] = list(
                                     map(
                                        lambda x: x.hour + dt.timedelta(hours=2), df.index
                                        )
                                     )
    gc.collect()
    return _tmp


def tag_glycaemiae(df:  pd.core.frame.DataFrame, 
                   column_name: str = 'glycaemia') ->  pd.core.frame.DataFrame:
    """ Return a copy of 'df', 
    adding a column called 'tag'.
    See help(classify).
    Optional param:
        column_name - containing the name of the column
                        of glycaemic values
        default : 'glycaemia'
    """
    _tmp = copy.deepcopy(df)
    _tmp.loc[:,'tag'] = list(
                             map(
                                classify, df[column_name]
                                )
                            )
    gc.collect()
    return _tmp

def get_paired_measurements(df:  pd.core.frame.DataFrame, 
                           column_names: Tuple[str, str] = ('IG', 'BG') ) -> List[Tuple]:
    """ Given a dataframe containing two columns of paired measurements, 
    return a list of tuples containing the corresponding entries.
    This function filters out any measurements which may have a missing observation, 
    as in (np.nan, 35) or (79.456, np.nan).
    """
    # f := Get paired observations as tuples, or return a tuple of (False, False)
    #      to be filtered afterwards
    f = lambda x, y: (x, y) if not (np.isnan(x) or np.isnan(y)) else (False, False)
    
    # g := Return a the given tuple 'x', if and only if both are True
    #      which in this context translates to them being numerical values (nonzero).
    g = lambda x: x if x[0] and x[1] else False
    
    _tagged = list(
                    map(f, df[column_names[0]], df[column_names[1]])
                  )
    _paired = list(
                    filter(g , _tagged)
                  )
    return _paired

def merge_glycaemic_values(df:  pd.core.frame.DataFrame, drop_na=True,
                           column_names: Tuple[str, str] = ('IG', 'BG')) -> pd.core.frame.DataFrame:
        """ Merge the glycaemic values from two specified columns into
        a sinlge column 'glycaemia'.
        """
        _tmp = copy.deepcopy(df)
        
        _tmp[column_names[1]] = list(map(
                                    lambda x, y: x if not np.isnan(x) else y, 
                                    _tmp[column_names[0]], _tmp[column_names[1]]
                                ))
        _tmp[column_names[0]] = list(map(
                                    lambda x, y: x if not np.isnan(x) else y,
                                    _tmp[column_names[0]], _tmp[column_names[1]]
                                ))  
        
        _tmp['glycaemia'] = _tmp['IG']
        _tmp = _tmp.drop(['IG', 'BG'], axis=1)
        if drop_na:
            _tmp = _tmp.dropna()
        
        gc.collect()
        return _tmp


# In[4]:


d = Drive.Drive()


# In[5]:


file_name = 'journal.jl'
file_path = os.path.join('data', file_name)
d.download(file_name   = file_name, 
           target_name = file_path
          )


# In[6]:


pre.file_filter(file_path)


# In[7]:


ls data/


# In[8]:


_raw = pd.read_json(file_path, lines=True)


# In[9]:


undesired_columns = [ 
    'LOT',
    'REF', 
    'initSuccess', 
    'secondRound',
    'food'
]


# In[10]:


_tmp = _raw.drop(undesired_columns, axis=1)
_tmp = _tmp.loc[ (_tmp['type'] == 'data') | (_tmp['type'] == 'event') ]


# In[11]:


"""
_tmp = (
        _raw.drop(undesired_columns, axis=1)
       ).loc[
                (_raw['type'] == 'data') |
                (_raw['type'] == 'event')
            ]
"""


# In[12]:


_t_data = time_indexed_df(_tmp)


# In[13]:


data = fill_nas(_t_data)


# In[14]:


data2 = copy.deepcopy(data)


# In[15]:


data3 = merge_glycaemic_values(data2)
data3 = set_hour(data3)
data3 = tag_glycaemiae(data3)


# In[16]:


postp = data3[ data3['details'] == 'Postprandial']


# In[17]:


meals = data3[ data3.carbs != 0 ]


# In[18]:


postp.head(10)


# In[19]:


meals.head(10)


# In[27]:


real_pairs = []
for i in meals.index:
    for j in postp.index:
        #print(f'{i + dt.timedelta(hours=1)}', f'{j}', f'{i + dt.timedelta(hours=3)}')
        if (i + dt.timedelta(hours=1) < j) and (i + dt.timedelta(hours=3) > j):
            real_pairs.append((i, j))
        #elif i + dt.timedelta(hours=3) < j


# In[28]:


len(real_pairs)


# In[39]:


print(data3.index[0] - dt.timedelta(hours=2))
data3.index[0] + dt.timedelta(hours=2)


# In[28]:


dt.timedelta


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




