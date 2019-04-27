#!/usr/bin/env python
# coding: utf-8

# In[147]:


## Standard :

# Name imports :
import os
import gc
import copy
import sklearn
import itertools
import time
import typing

# Aliased imports :
import multiprocessing as mp
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sb
import matplotlib.pyplot as plt

# Full imports :
from plotnine import *


# In[148]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 6)


# In[149]:


def split(arr: list, count: int) -> typing.List[list]:
    return [arr[i::count] for i in range(count)]


def time_indexed_df(df1: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """ Cast into a time-indexed dataframe.
    df1 paramater should contain a column called 'dateTime',
    which contains entries of type pandas._libs.tslibs.timestamps.Timestamp
    """
    _tmp = copy.deepcopy(df1)
    _tmp.index = df1.dateTime
    _tmp.drop('dateTime', axis=1, inplace=True)
    _tmp = _tmp.sort_index()
    gc.collect()
    return _tmp

def merge_date_time(df1: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """ Take a dataframe containing columns:
        'Date'
        'Time'
        
        And return one containing a single column:
         'dateTime' = 'Date' + 'Time'
        
        For each entry as seen below:
         '2019-03-21 17:34:05' <- '2019-03-21' + '17:34:05'
    """
    _tmp = copy.deepcopy(df1)
    _tmp['dateTime'] = _tmp['Date'] + ' ' + _tmp['Time']
    _tmp.drop(['Date', 'Time'], axis=1, inplace=True)
    gc.collect()
    return _tmp
    


# In[150]:


raw = pd.read_csv('data/carelink2.csv')


# In[151]:


raw.columns, len(raw.index)


# In[152]:


#raw['Bolus Source'].value_counts()
#raw['Bolus Number'].value_counts()
#len(raw['dateTime'])
#raw['BWZ Unabsorbed Insulin Total (U)'].value_counts()
#raw['Bolus Cancellation Reason'].value_counts()
raw['Bolus Number'] = raw['Bolus Number'].apply(lambda x: int(x) if type(x) is str else x)


# In[153]:


# Check if the list contains other thing than integers.
list(
    filter(
        lambda x: False if type(x) is int else True, raw['Final Bolus Estimate'].value_counts()
    )
)


# In[154]:


type(5) is int


# In[155]:


raw = merge_date_time(raw)


# In[156]:


# Remove ['MiniMed 640G MMT-1512/1712 Sensor', 'Date Time'] from the column, 
# as they impede coercing the values into timestamps.
for row in filter(lambda x: False if ':' in x else True, raw['dateTime'] ):
    raw = raw[raw.dateTime != row]


# In[157]:


pool = mp.Pool(processes=4)
start = time.clock()
raw.dateTime = pool.map(pd.to_datetime, raw.dateTime)
elapsed = time.clock()
print(f'{elapsed - start}')


# In[143]:


start = time.clock()
raw.dateTime = list(map(pd.to_datetime, raw.dateTime))
elapsed = time.clock()
print(f'{elapsed - start}')


# In[73]:


time


# In[20]:


start = time.clock()
raw.dateTime = raw.dateTime.apply(pd.to_datetime)
elapsed = time.clock()
print(f'{elapsed - start}')


# In[121]:


type(raw.dateTime)


# In[133]:


undesired_columns = [
    'Index',
    'New Device Time',
    'Prime Type', 
    'Prime Volume Delivered (U)',
    'Alarm', 
    'Suspend', 
    'Rewind',
    'Linked BG Meter ID',
    'Bolus Cancellation Reason',
    'Scroll Step Size',
    'Sensor Calibration Rejected Reason',
    'Network Device Associated Reason',
    'Network Device Disassociated Reason',
    'Network Device Disconnected Reason',
    'Sensor Exception',
    'Preset Temp Basal Name',
    'Preset Bolus', 
    'Bolus Source'
]


# In[134]:


raw = raw.drop(undesired_columns, axis=1)


# In[128]:


raw['New Device Time'].value_counts()


# In[129]:


proc1 = time_indexed_df(raw)


# In[132]:


proc1.loc['2019/04/01 12', :]


# In[107]:


proc1.iloc[0, :], proc1.iloc[len(proc1.index)-1, :]


# In[29]:


with open('dateTime.txt', 'w') as f:
    for i in raw['dateTime']:
        f.write(f'{i}\n')


# In[30]:


raw.index, raw['Index']


# In[31]:


raw['New Device Time'].count()


# In[32]:


raw[ raw['New Device Time'].notnull() ]['New Device Time']


# In[33]:


list(
    map(
        type, list(raw[ raw['Sensor Glucose (mg/dL)'].notnull() ]['Sensor Glucose (mg/dL)'])
    )
)


# In[34]:


glucosas = raw[ raw['Sensor Glucose (mg/dL)'].notnull() ]['Sensor Glucose (mg/dL)']


# In[35]:


glucosas = pd.to_numeric(glucosas, 'coerce')


# In[ ]:





# In[36]:


glucosas = glucosas[glucosas == glucosas // 1]


# In[37]:


sb.distplot(glucosas)


# In[38]:


glucosas.plot()


# In[46]:


proporciones = (lambda borne_inf, borne_sup: 
                { 
                    'hypo':  100 * glucosas[glucosas < borne_inf].count() / glucosas.count(),
                    'normo': 100 * glucosas[(glucosas >= borne_inf) & (glucosas <=borne_sup)].count() / glucosas.count(),
                    'hyper': 100 * glucosas[glucosas > borne_sup].count() / glucosas.count()
                }
)(70, 160)


# In[48]:


proporciones


# In[49]:


glucosas.mean(), glucosas.std()


# In[50]:


raw['BWZ Estimate (U)'].count()/90


# In[51]:


raw.iloc[0,:], raw.iloc[len(raw.index)-1, :]


# In[29]:


x = pd.core.frame.DataFrame()


# In[96]:


'''
glucosas = filter(lambda x: x if not np.isnan(x) else False,
                  list(
                    map(int, glucosas)
                    )
            )
#Value error: cannot convert float NaN to integer
'''    


# In[ ]:




