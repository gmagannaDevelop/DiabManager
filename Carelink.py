#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Standard :

# Name imports :
import os
import gc
import copy
import sklearn
import itertools

# Aliased imports :
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sb

# Full imports :
from plotnine import *


# In[2]:


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


# In[3]:


raw = pd.read_csv('data/carelink.csv')


# In[21]:


x = raw['ISIG Value']


# In[22]:


dir(x)


# In[4]:


raw.columns


# In[54]:


#raw['Bolus Source'].value_counts()
#raw['Bolus Number'].value_counts()
len(raw['dateTime'])


# In[55]:


len(list(
    filter(lambda x: True if ':' in x else False, raw['dateTime'] )
))


# In[56]:


list(
    filter(lambda x: False if ':' in x else True, raw['dateTime'] )
)


# In[57]:


for row in filter(lambda x: False if ':' in x else True, raw['dateTime'] ):
    raw = raw[raw.dateTime != row]


# In[58]:


list(map(pd.to_datetime, raw.dateTime))


# In[59]:


raw.dateTime = raw.dateTime.apply(pd.to_datetime)


# In[60]:





# In[46]:


undesired_columns = [
    'Scroll Step Size'
    'Sensor Calibration Rejected Reason'
    'Network Device Associated Reason',
    'Network Device Disassociated Reason',
    'Network Device Disconnected Reason',
    'Sensor Exception',
    'Preset Temp Basal Name',
    'Preset Bolus', 
    'Bolus Source'
]


# In[8]:


for date, time, i in zip(raw['Date'], raw['Time'], range(4)):
    print(date + ' ' +time)


# In[47]:


raw['dateTime'] = raw['Date'] + ' ' +raw['Time']


# In[51]:


with open('dateTime.txt', 'w') as f:
    for i in raw['dateTime']:
        f.write(f'{i}\n')


# In[11]:


raw.index, raw['Index']


# In[10]:


raw['New Device Time'].count()


# In[19]:


raw[ raw['New Device Time'].notnull() ]['New Device Time']


# In[31]:


list(
    map(
        type, list(raw[ raw['Sensor Glucose (mg/dL)'].notnull() ]['Sensor Glucose (mg/dL)'])
    )
)


# In[33]:


glucosas = raw[ raw['Sensor Glucose (mg/dL)'].notnull() ]['Sensor Glucose (mg/dL)']


# In[41]:


glucosas = pd.to_numeric(glucosas, 'coerce')


# In[70]:


'''
glucosas = filter(lambda x: x if not np.isnan(x) else False,
                  list(
                    map(int, glucosas)
                    )
            )
Value error: cannot convert float NaN to integer
'''    


# In[50]:


glucosas = glucosas[glucosas == glucosas // 1]


# In[51]:


sb.distplot(glucosas)


# In[79]:


sb.


# In[65]:


proporciones = { 
    'hypo':  100 * glucosas[glucosas < 70].count() / glucosas.count(),
    'normo': 100 * glucosas[(glucosas >= 70) & (glucosas <=180)].count() / glucosas.count(),
    'hyper': 100 * glucosas[glucosas > 180].count() / glucosas.count()
}


# In[66]:


proporciones


# In[69]:


glucosas.mean(), glucosas.std()


# In[77]:


raw['BWZ Estimate (U)'].count()/90


# In[76]:


raw.iloc[0,:], raw.iloc[len(raw.index)-1, :]


# In[29]:


x = pd.core.frame.DataFrame()


# In[31]:





# In[ ]:




