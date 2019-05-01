#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reset', '')


# In[2]:


### To measure execution time (this method might not be very accurate, use with precaution)
#start = time.clock()
### "Instructions"
#elapsed = time.clock()
#print(f'{elapsed - start}')


# In[208]:


## Standard :

# Name imports :
import os
import gc
import copy
import sklearn
import itertools
import time
import typing
import random

# Aliased imports :
import multiprocessing as mp
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sb
import matplotlib.pyplot as plt

# Partial imports :
from scipy.integrate import quad as integrate
from sklearn.neighbors import KernelDensity
from sklearn.svm import SVR

# Full imports :
from plotnine import *


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 8)
sb.set_style("dark")


# In[5]:


def split(arr: list, count: int) -> typing.List[list]:
    return [arr[i::count] for i in range(count)]

def overlapping_histograms(df: pd.core.frame.DataFrame, 
                           columns: typing.List[str], 
                           names=None, 
                           colors: typing.List[str]=None,
                           labels: typing.Tuple[str]=None
                          ) -> bool:
    """ Create a figure with overlapping histograms and KDEs from 
    a dataframe's specified columns.
    
    df : A pandas.core.frame.DataFrame
    columns : Names of the columns that will be used to construct the histograms.
    names :  Used to label each histogram and KDE, defaults to 'columns'.
    colors :  A list of colors for the histograms. See sb.xkcd_rgb for available colors.
    labels : A tuple containing ('Plot Title', 'xlabel', 'ylabel' )
    
    Returns: 
        True uppon success
        False uppon an error i.e. 
                One of the specified columns isn't found
                on df.columns
    """
    for col in columns:
        if not (col in df.columns):
            return False
    
    if not names:
        names = columns
    
    if not colors:
        colors = [random.choice(list(sb.xkcd_rgb.values())) for i in range(len(columns))]
    
    for column, name, color in zip(columns, names, colors):
        sb.distplot(
            raw[column].dropna(), 
            kde_kws={"color":color,"lw":2,"label":name,"alpha":0.6}, 
            hist_kws={"color":color,"alpha":0.25}
        )
    
    if labels:
        plt.title(labels[0])
        plt.xlabel(labels[1])
        plt.ylabel(labels[2])
    
    return True
    

def select_date_range(df: pd.core.frame.DataFrame, 
                     start_date: str, 
                     end_date: str) -> pd.core.frame.DataFrame:
    """
    """
    mask = (df.index >= start_date) & (df.index <= end_date)
    
    

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

def hybrid_interpolator(data: pd.core.series.Series,
                        mean: float = None,
                        limit: float = None,
                        methods: typing.List[str] = ['linear', 'spline'], 
                        weights: typing.List[float] = [0.65, 0.35],
                        direction: str = 'forward',
                        order: int = 2
                       ) -> pd.core.series.Series:
    """
    Return a pandas.core.series.Series instance resulting of the weighted average
    of two interpolation methods.
    
    Model:
        φ = β1*method1 + β2*method2
        
    Default:
        β1, β2 = 0.6, 0.4
        method1, method2 = linear, spline
    
    Weights are meant to be numbers from the interval (0, 1)
    which add up to one, to keep the weighted sum consistent.
    
    limit_direction : {‘forward’, ‘backward’, ‘both’}, default ‘forward’
    If limit is specified, consecutive NaNs will be filled in this direction.
    
    If the predicted φ_i value is outside of the the interval
    ( (mean - limit), (mean + limit) )
    it will be replaced by the linear interpolation approximation.
    
    If not set, mean and limit will default to:
        mean = data.mean()
        limit = 2 * data.std()
    
    This function should have support for keyword arguments, but is yet to be implemented.
    """
    predictions: typing.List[float] = [] 
    
    if not np.isclose(sum(weight for weight in weights), 1):
        raise Exception('Sum of weights must be equal to one!')
    
    for met in methods:
        if (met == 'spline') or (met == 'polynomial'):
            predictions.append(data.interpolate(method=met, order=order, limit_direction=direction))
        else:
            predictions.append(data.interpolate(method=met, limit_direction=direction))

    linear: pd.core.series.Series = predictions[0]
    spline: pd.core.series.Series = predictions[1]
    hybrid: pd.core.series.Series = weights[0]*predictions[0] + weights[1]*predictions[1]
    
    corrected: pd.core.series.Series = copy.deepcopy(hybrid) 
    
    if not mean:
        mean = data.mean()
    if not limit:
        limit = 2 * data.std()
    
    for idx, val in zip(hybrid[ np.isnan(data) ].index, hybrid[ np.isnan(data) ]):
        if (val > mean + limit) or (val < mean - limit):
            corrected[idx] = linear[idx]
    
    #df = copy.deepcopy(interpolated)
    #print(df.isnull().astype(int).groupby(df.notnull().astype(int).cumsum()).sum())
    
    return corrected
    
    
def naive_hybrid_interpolator(data: pd.core.series.Series, 
                        methods: typing.List[str] = ['linear', 'spline'], 
                        weights: typing.List[float] = [0.85, 0.15],
                        direction: str = 'both',
                        order: int = 2
                       ) -> pd.core.series.Series:
    """
    Return a pandas.core.series.Series instance resulting of the weighted average
    of two interpolation methods.
    
    Model:
        φ = β1*method1 + β2*method2
        
    Default:
        β1, β2 = 0.6, 0.4
        method1, method2 = linear, spline
    
    limit_direction : {‘forward’, ‘backward’, ‘both’}, default ‘forward’
    If limit is specified, consecutive NaNs will be filled in this direction.
    
    This function should have support for keyword arguments, but is yet to be implemented.
    """
    predictions: typing.List[float] = [] 
    
    if sum(weight for weight in weights) > 1:
        raise Exception('Sum of weights must be equal to one!')
    
    for met in methods:
        if (met == 'spline') or (met == 'polynomial'):
            predictions.append(data.interpolate(method=met, order=order, limit_direction=direction))
        else:
            predictions.append(data.interpolate(method=met, limit_direction=direction))

    #linear = predictions[0]
    #spline = predictions[1]
    
    #print(linear[ np.isnan(data) ])
    
    # working version:
    interpolated = weights[0]*predictions[0] + weights[1]*predictions[1]
    
    #df = copy.deepcopy(interpolated)
    #print(df.isnull().astype(int).groupby(df.notnull().astype(int).cumsum()).sum())
    
    return interpolated
    

def probability_estimate(data: pd.core.series.Series, 
                         start: float, 
                         end: float, 
                         N: int = 150,
                         show_plots=False) -> float:
    """
    """
    
    # Plot the data using a normalized histogram
    dev = copy.deepcopy(data)
    dev = dev.dropna()
    x = np.linspace(dev.min(), min(data), max(data))[:, np.newaxis]

    # Do kernel density estimation
    kd = KernelDensity(kernel='gaussian', bandwidth=0.85).fit(np.array(dev).reshape(-1, 1))

    # Plot the estimated densty
    kd_vals = np.exp(kd.score_samples(x))

    # Show the plots
    if show_plots:
        plt.plot(x, kd_vals)
        plt.hist(dev, 50, normed=True)
        plt.xlabel('Concentration mg/dl')
        plt.ylabel('Density')
        plt.title('Probability Density Esimation')
        plt.show()

    #probability = integrate(lambda x: np.exp(kd.score_samples(x.reshape(-1, 1))), start, end)[0]
    
    # Integration :
    step = (end - start) / (N - 1)  # Step size
    x = np.linspace(start, end, N)[:, np.newaxis]  # Generate values in the range
    kd_vals = np.exp(kd.score_samples(x))  # Get PDF values for each x
    probability = np.sum(kd_vals * step)  # Approximate the integral of the PDF
    
    return probability


def dev_from_mean(data: pd.core.series.Series) -> typing.Tuple[float, pd.core.series.Series, float]:
    """
    Returns (mean, abs_deviations, avg_of_devs)
    
        mean: float = the mean of the sample
        
        abs_deviations: pandas.core.series.Series = absoulute value of deviations from the mean
        
        avg_of_devs: float = mean of the absolute value of deviations.
    """
    _mean: float = data.mean()
    _std: float = data.std()
    _devs: pd.core.series.Series = np.abs(data - _mean)
    _avg_dev_mean: float = _devs.mean()
        
    return _mean, _std, _devs, _avg_dev_mean


def porcentage_original(serie: pd.core.series.Series, start: str, stop: str) -> float:
    return serie[start:stop].count() / len(serie.loc[start:stop])
    
    
def porcentage_interpolated(serie: pd.core.series.Series, start: str, stop: str) -> float:
    return 1 - porcentage_original(serie, start, stop)


# In[6]:


raw = pd.read_csv('data/carelink2.csv')


# In[7]:


raw.columns, len(raw.index)


# In[8]:


#raw['Bolus Source'].value_counts()
#raw['Bolus Number'].value_counts()
#len(raw['dateTime'])
#raw['BWZ Unabsorbed Insulin Total (U)'].value_counts()
#raw['Bolus Cancellation Reason'].value_counts()
#raw['Bolus Number'] = raw['Bolus Number'].apply(lambda x: int(x) if type(x) is str else x)


# In[9]:


# Check if the list contains other thing than integers.
list(
    filter(
        lambda x: False if type(x) is int else True, raw['Final Bolus Estimate'].value_counts()
    )
)


# In[10]:


raw = merge_date_time(raw)


# In[11]:


# Remove ['MiniMed 640G MMT-1512/1712 Sensor', 'Date Time'] from the column, 
# as they impede coercing the values into timestamps.
for row in filter(lambda x: False if ':' in x else True, raw['dateTime'] ):
    raw = raw[raw.dateTime != row]


# In[12]:


pool = mp.Pool() # processes parameter can be set manually, 
                 # but this is suposed to spawn the same number as the system has cores.

raw.dateTime = pool.map(pd.to_datetime, raw.dateTime)

pool.close()
pool.terminate()


# In[13]:


undesired_columns = [
    'Index',
    'Temp Basal Type', 
    'Temp Basal Duration (h:mm:ss)',
    'BWZ Target High BG (mg/dL)', 
    'BWZ Target Low BG (mg/dL)',
    'Bolus Type',
    'Insulin Action Curve Time',
    'New Device Time',
    'Bolus Duration (h:mm:ss)',
    'Prime Type', 
    'Prime Volume Delivered (U)',
    'Alarm',
    'ISIG Value',
    'Event Marker',
    'Bolus Number',
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


# In[14]:


raw = raw.drop(undesired_columns, axis=1)


# In[15]:


raw.columns


# In[16]:


unsure_columns = [
    'BG Reading (mg/dL)',
    'Sensor Calibration BG (mg/dL)'
]


# In[17]:


proc1 = raw.drop(unsure_columns, axis=1)


# In[18]:


proc1 = time_indexed_df(proc1)


# In[19]:


# len(proc1)


# In[20]:


# proc1.head(3) 


# In[21]:


proc1 = proc1.iloc[2:, :]


# In[22]:


# len(proc1)


# In[23]:


# proc1.head(3)


# In[24]:


# Cutoff seconds so that measurements are not lost when interpolating.
proc1.index = proc1.index .map(lambda t: t.replace(second=0))


# In[25]:


overlapping_histograms(proc1, 
                       ['Bolus Volume Delivered (U)', 'BWZ Correction Estimate (U)', 'BWZ Food Estimate (U)'],
                       colors=['red', 'green', 'blue'], 
                       labels=('Bolus Wizard Estimation', 'Units', 'Density')
                      )


# In[81]:


proc2 = proc1.copy()


# In[79]:


# False in list(filter(lambda x: x, sorted(proc2.index) == proc2.index))


# In[80]:


# proc2['Sensor Glucose (mg/dL)'] = hybrid_interpolator(proc2['Sensor Glucose (mg/dL)'])


# In[173]:


def resample_dataframe(_df : pd.core.frame.DataFrame,
         interpolation : bool = False,
         resample_freq : str  = '1T') -> pd.core.frame.DataFrame:
    '''
        Resamples
    '''

    df = _df.copy()
    df = df.resample(resample_freq).mean()
    #_index = df.index
    
    if interpolation:
        df['Sensor Glucose (mg/dL)'] = df['Sensor Glucose (mg/dL)'].interpolate(method='linear')
        #df['Basal Rate (U/h)'] = df['Basal Rate (U/h)'].interoplate(method='pad')
  
    return df


# In[58]:


#proc1.loc[proc1.index[0], proc1.index[-1]].max()


# In[59]:


len(proc1.index), len(proc1.index.get_duplicates())


# In[179]:


dummy = proc1.copy()


# In[180]:


dummy['_grouper'] = dummy.index


# In[181]:


dummy = dummy.groupby('_grouper').max().reset_index()


# In[182]:


dummy.index = dummy['_grouper']


# In[183]:


dummy = dummy.drop('_grouper', axis=1)


# In[184]:


dummy['Sensor Glucose (mg/dL)'] = hybrid_interpolator(dummy['Sensor Glucose (mg/dL)'])


# In[153]:


# proc1['Sensor Glucose (mg/dL)']


# In[154]:


# dummy['Sensor Glucose (mg/dL)']


# In[107]:


dummy = dummy.loc['2018/06/24':'2019/04/23']


# In[116]:


#help(dummy.dropna)
dummy['Sensor Glucose (mg/dL)'].count() / len(dummy['Sensor Glucose (mg/dL)'])


# In[123]:


data = dummy.dropna(axis='columns')


# In[124]:


X = np.array(data.index)


# In[138]:


X = X.reshape(-1, 1)


# In[126]:


y = np.array(data)


# In[146]:


y = y.flatten()


# In[149]:


X


# In[150]:


y


# In[147]:


np.savetxt('data/X.np', X)


# In[148]:


np.savetxt('data/y.np', y)


# In[151]:


X


# In[152]:


y


# In[155]:


dummy.loc['2019/02/02 15':'2019/02/04 20'].plot()


# In[165]:


first_half = data.loc['2019/01/01':'2019/02/02 15']


# In[166]:


second_half = data.loc['2019/02/04 20':data.index[-1]]


# In[167]:


first_half.iloc[len(first_half)-100:len(first_half)-3, :].plot()


# In[168]:


proc1.iloc[0:1000].plot()


# In[185]:


first_half_resampled = resample_dataframe(first_half, interpolation=True)


# In[187]:


second_half_resampled = resample_dataframe(second_half, interpolation=True)


# In[189]:


#second_half_resampled


# In[191]:


def nn_format_df(    df : pd.core.frame.DataFrame,                  order : str = 'first'           ) -> pd.core.frame.DataFrame:
  ''' Take a DataFrame with n columns and return a DataFrame with 
  m*n columns containing the information of those three original columns
  within the next 0:30, 1, 1:30, 2, 2:30, 2:45, and 3 hours (depending on the order).
    order:
      'first'  = 1, 2, and 3 hours.
      'second' = 1, 2, 2:45, and 3 hours.
      'third'  = 0:30, 1, 1:30, 2, 2:30, 2:45, and 3 hours.
  '''
  _df : pd.core.frame.DataFrame = df.copy()
  if order == 'first':
    _df = pd.concat([_df,                     _df.shift( -60).rename(columns=dict((elem, elem+'1') for elem in df.columns)),                     _df.shift(-120).rename(columns=dict((elem, elem+'2') for elem in df.columns)),                     _df.shift(-180).rename(columns=dict((elem, elem+'3') for elem in df.columns))],                      axis=1)
  elif order == 'second':
    _df = pd.concat([_df,                   _df.shift( -60).rename(columns=dict((elem, elem+'1') for elem in df.columns)),                   _df.shift(-120).rename(columns=dict((elem, elem+'2') for elem in df.columns)),                   _df.shift(-165).rename(columns=dict((elem, elem+'2:45') for elem in df.columns)),                   _df.shift(-180).rename(columns=dict((elem, elem+'3') for elem in df.columns))],                    axis=1)
  elif order == 'third':
    _df = pd.concat([_df,                   _df.shift( -15).rename(columns=dict((elem, elem+'+ 0:15') for elem in df.columns)),                   _df.shift( -30).rename(columns=dict((elem, elem+'+ 0:30') for elem in df.columns)),                   _df.shift( -45).rename(columns=dict((elem, elem+'+ 0:45') for elem in df.columns)),                   _df.shift( -60).rename(columns=dict((elem, elem+'+ 1:00') for elem in df.columns)),                   _df.shift( -75).rename(columns=dict((elem, elem+'+ 1:15') for elem in df.columns)),                   _df.shift( -90).rename(columns=dict((elem, elem+'+ 1:30') for elem in df.columns)),                   _df.shift(-105).rename(columns=dict((elem, elem+'+ 1:45') for elem in df.columns)),                   _df.shift(-120).rename(columns=dict((elem, elem+'+ 2:00') for elem in df.columns)),                   _df.shift(-180).rename(columns=dict((elem, elem+'+ 3:00') for elem in df.columns))],                    axis=1)
  elif order == 'naive':
    _df = pd.concat([_df,                   _df.shift( -30).rename(columns=dict((elem, elem+'0:30') for elem in df.columns)),                   _df.shift( -60).rename(columns=dict((elem, elem+'1') for elem in df.columns)),                   _df.shift( -90).rename(columns=dict((elem, elem+'1:30') for elem in df.columns)),                   _df.shift(-120).rename(columns=dict((elem, elem+'2') for elem in df.columns)),                   _df.shift(-150).rename(columns=dict((elem, elem+'2:30') for elem in df.columns)),                   _df.shift(-165).rename(columns=dict((elem, elem+'2:45') for elem in df.columns)),                   _df.shift(-180).rename(columns=dict((elem, elem+'3') for elem in df.columns))],                    axis=1)
  else:
    printf('Error, order {order} is not valid. \n Options are: \'first\', \'second\', or \'third\' \n')
    _df = None

  return _df


# In[200]:


hola = nn_format_df(first_half_resampled).dropna(how='any')


# In[204]:


adios = nn_format_df(second_half_resampled).dropna(how='any')


# In[205]:


hola.to_csv('binaries/first_half_resampled.csv')


# In[206]:


adios.to_csv('binaries/second_half_resampled.csv')


# In[207]:


adios


# In[209]:


def predict_values(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))
    
    svrs = {
        'linear': SVR(kernel='linear', C=1e3), 
        'poly': SVR(kernel='poly', C=1e3, degree=2), 
        'rbf': SVR(kernel='rbf', C=1e3, gamma=0.1)
    }
    
    [ svrs[key].fit(dates, prices) for key in svrs.keys() ]
    


# In[215]:


str(dummy.index[0]).split('-')[2]


# In[ ]:


def predict_prices(dates, prices, x):
    dates = np.reshape(dates,(len(dates), 1)) # convert to 1xn dimension
    x = np.reshape(x,(len(x), 1))
    
    svr_lin  = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    
    # Fit regression model
    svr_lin .fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)
    
    plt.scatter(dates, prices, c='k', label='Data')
    plt.plot(dates, svr_lin.predict(dates), c='g', label='Linear model')
    plt.plot(dates, svr_rbf.predict(dates), c='r', label='RBF model')    
    plt.plot(dates, svr_poly.predict(dates), c='b', label='Polynomial model')
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    
    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]


# In[231]:


#def predict_prices(dates, prices, x):
#    dates = np.reshape(dates,(len(dates), 1)) # convert to 1xn dimension
#    x = np.reshape(x,(len(x), 1))

class SVRegressor(object):
    
    def __init__(self, 
                 C: float = 1e3, 
                 degree: int = 2, 
                 gamma: float = 0.1,
                 features: str = 'X',
                 labels: str = 'y'
                ):
        self._keys = ['linear', 'poly', 'rbf']   
        self._svrs = {
            'linear': SVR(kernel='linear', C=C), 
            'poly': SVR(kernel='poly', C=C, degree=degree), 
            'rbf': SVR(kernel='rbf', C=C, gamma=gamma)
        }
        self._labels = {
            'features': features, 
            'labels': labels
        }
    
    def __getitem__(self, key):
        if key in self.keys:
            return self._svrs[position]
        else:
            raise Exception(f'{key} not found in keys. Possible values are: {self.keys}')
            
    @property
    def keys(self):
        return self._keys
    
    @property
    def kernels(self):
        return self._keys
    
    def set_training_data(self, X, y):
        if type(X) is np.ndarray and type(y) is np.ndarray:
            self._X = X
            self._y = y
        else:
            raise Exception('type() X and y should be numpy.ndarray')
        
    @property
    def training_data(self):
        ''' Returns self._X (features), self._y (labels)'''
        return self._X, self._y
        
    def fit(self, kernel: str = 'all'):
        if kernel == 'all':
            [ self._svrs[i].fit(self._X, self._y) for i in self.keys ]
        elif kernel in self.kernels:
            self._svrs[kernel].fit(self._X, self._y)
        else:
            raise Exception(f'Invalid kernel name, available kernels are {self.kernels}')

    def training_plot(self, xlabel: str = 'X', y_label: str = 'y'):
        plt.scatter(_X, y, c='k', label='Data')
        
        if len(self._X.shape) == 1:
            for key, color in zip(svrs.keys(), ['g', 'r', 'n']):
                plt.plot(self._X, svrs[key].predict(self._X), c=color, label=key)
        else:
            for i, key, color in enumerate(zip(svrs.keys(), ['g', 'r', 'n'])):
                plt.plot(i, svrs[key].predict(self._X), c=color, label=key)
    
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('Support Vector Regression')
        plt.legend()
        plt.show()
        
    def predict(self, kernel: str = 'all',  X: np.ndarray = None):
        """ Acutal predictions are the first element, to access them:
            SVRegressor.predict()[0]
        """
        if not X:
            X = self._X
        elif type(X) is not np.ndarray:
            raise Exception('Input type  type(X), shoud be numpy.ndarray')
        
        if kernel == 'all':
            return  { 
                kernel: self._svrs[kern].predict(X) for kern in self.kernels 
            }
        elif kernel not in self.kernels:
                raise Exception(f'Kernel not found. Possible values are: all, {self.kernels}')
        else:
            return self._svrs[kernel].predict(X) 


# In[238]:


data = pd.read_csv('binaries/t03.csv')
data.columns


# In[253]:


X = data.loc[:, 'Sensor Glucose (mg/dL)':'Sensor Glucose (mg/dL)2'].values
X


# In[252]:


y = data['Sensor Glucose (mg/dL)3'].values
y


# In[254]:


R = SVRegressor()


# In[255]:


R.set_training_data(X, y)


# In[ ]:


R.fit()


# In[229]:


type(np.array([[1, 2], [3, 4]])) is not np.ndarray


# In[227]:


len(np.array([1, 2, 3, 4]).shape)


# In[109]:


proc1['Basal Rate (U/h)'].interpolate(method='pad').count() / len(proc1['Basal Rate (U/h)'].interpolate(method='pad'))


# (proc1.loc['2019/02/14 12':'2019/02/15 12']['Bolus Volume Delivered (U)'].dropna()*10).plot()
# (proc1.loc['2019/02/14 12':'2019/02/15 12']['BWZ Carb Input (grams)'].dropna()).plot()
# (proc1.loc['2019/02/14 12':'2019/02/15 12']['Basal Rate (U/h)'].interpolate(method='pad')*100).plot()
# proc1.loc['2019/02/14 12':'2019/02/15 12']['Sensor Glucose (mg/dL)'].interpolate(method='linear').plot()
# proc1.loc['2019/02/14 12':'2019/02/15 12']['Sensor Glucose (mg/dL)'].interpolate(method='slinear').plot()
# proc1.loc['2019/02/14 12':'2019/02/15 12']['Sensor Glucose (mg/dL)'].interpolate(method='quadratic').plot()
# proc1.loc['2019/02/14 12':'2019/02/15 12']['Sensor Glucose (mg/dL)'].interpolate(method='cubic').plot()
# proc1.loc['2019/02/14 12':'2019/02/15 12']['Sensor Glucose (mg/dL)'].interpolate(method='spline', order=2).plot()
# [
#     hybrid_interpolator(proc1.loc['2019/02/14 12':'2019/02/15 12']['Sensor Glucose (mg/dL)'], weights=w).plot()
#     for w in weights_set
# ]
# proc1.loc['2019/02/14 12':'2019/02/15 12']['Sensor Glucose (mg/dL)']
# plt.axhline(200, color='red')
# plt.axhline(70, color='green')
# plt.legend(['Bolus', 'Carbs', 'Basal', 'Linear', 'Slinear', 'Quadratic', 'Cubic', 'spline', *labs, 'Data'])

# In[65]:


#overlapping_histograms(proc1, 
 #                      ['Bolus Volume Delivered (U)', 'BWZ Correction Estimate (U)', 'BWZ Food Estimate (U)'],
  #                     colors=['red', 'green', 'blue'], 
   #                    labels=('Bolus Wizard Estimation', 'Units', 'Density')
    #                  )


# In[66]:


#sb.distplot(dummy['Sensor Glucose (mg/dL)'].dropna())
#sb.distplot(resample_dataframe(dummy)['Sensor Glucose (mg/dL)'].dropna())


# In[67]:


dummy2 = resample_dataframe(dummy)


# In[69]:


dummy2['2019']


# In[40]:


resample_dataframe(dummy).count()


# In[41]:


proc1.count()


# In[37]:


proc1[ proc1['Sensor Glucose (mg/dL)'] == dummy.loc[proc1.index]['Sensor Glucose (mg/dL)'] ]['Sensor Glucose (mg/dL)']


# In[477]:


proc1[ proc1['Sensor Glucose (mg/dL)'] != dummy.loc[proc1.index]['Sensor Glucose (mg/dL)'] ]['Sensor Glucose (mg/dL)'].dropna()


# In[38]:


dummy[ proc1['Sensor Glucose (mg/dL)'] != dummy.loc[proc1.index]['Sensor Glucose (mg/dL)'] ]['Sensor Glucose (mg/dL)'].dropna()


# In[472]:


proc1.loc['2018-06-23 12:08:00']['Sensor Glucose (mg/dL)'], dummy.loc['2018-06-23 12:08:00']['Sensor Glucose (mg/dL)']


# In[426]:


dummy['Sensor Glucose (mg/dL)'].plot()


# In[428]:


proc1.duplicated().head()


# In[432]:


proc1.loc['2018/06/23 17:27:00']


# In[357]:


f"Original index: {len(proc1.index)},\n  Original duplicates: {len(proc1.index.get_duplicates())},\n  index - duplicates: {len(proc1.index) - len(proc1.index.get_duplicates())},\n  merged_index: {len(dummy.index)}, merged_index_duplicates: {len(dummy.index.get_duplicates())}"


# In[360]:


proc1.duplicated()


# In[349]:


'2019-04-23 16:27:00' in map(lambda x: str(x), proc1.index.get_duplicates())


# In[340]:


pd.core.series.Series(sorted(set(proc1.index))) == proc1.index


# In[332]:


df  = proc1.replace(np.nan, '').ffill().bfill()
df_ = df.replace('', np.nan).ffill().bfill()


# In[333]:


pd.concat([
        df_[df_.duplicated()],
        df.loc[df_.drop_duplicates(keep=False).index]
    ])


# In[334]:


proc1.resample('1T').asfreq()


# In[328]:


resampled1 = resample_dataframe(proc1)


# In[322]:


proc1.index


# In[323]:


proc1.index.map(lambda t: t.replace(second=0))


# In[324]:


proc1.iloc[0:20, :]


# In[326]:


resampled1.iloc[25:40, :]


# In[19]:


len(proc1['Basal Rate (U/h)']), proc1['Basal Rate (U/h)'].count()


# In[20]:


#with sb.axes_style('white'):
 #   sb.jointplot('Bolus Volume Delivered (U)', 'BWZ Correction Estimate (U)', raw[['Bolus Volume Delivered (U)', 'BWZ Correction Estimate (U)']].dropna(), kind='hex')


# In[21]:


len(proc1['2019']['Basal Rate (U/h)']), proc1['2019']['Basal Rate (U/h)'].interpolate(method='pad').count()


# In[262]:


mean, std, dev, _avg_dev_mean = dev_from_mean(proc1['Sensor Glucose (mg/dL)'])
mean, std, _avg_dev_mean


# In[260]:


x = proc1['Sensor Glucose (mg/dL)']
probability_estimate(x, 
                     x.mean() - 2*x.std(), 
                     x.mean() + 2*x.std(),
                     N=500, 
                     show_plots=True
                    )


# In[263]:


probability_estimate(dev, 0, 80, N=500, show_plots=True)


# In[235]:


weights_set = [ 
    [0.1, 0.9], 
    [0.2, 0.8],
    [0.3, 0.7],
    [0.4, 0.6],
    [0.5, 0.5],
    [0.6, 0.4],
    [0.7, 0.3],
    [0.8, 0.2],
    [0.9, 0.1]
]
labs = [f'Corrected hybrid {weight}' for weight in weights_set]


# In[265]:


(proc1.loc['2019/02/14 12':'2019/02/15 12']['Bolus Volume Delivered (U)'].dropna()*10).plot()
(proc1.loc['2019/02/14 12':'2019/02/15 12']['BWZ Carb Input (grams)'].dropna()).plot()
(proc1.loc['2019/02/14 12':'2019/02/15 12']['Basal Rate (U/h)'].interpolate(method='pad')*100).plot()
proc1.loc['2019/02/14 12':'2019/02/15 12']['Sensor Glucose (mg/dL)'].interpolate(method='linear').plot()
proc1.loc['2019/02/14 12':'2019/02/15 12']['Sensor Glucose (mg/dL)'].interpolate(method='slinear').plot()
proc1.loc['2019/02/14 12':'2019/02/15 12']['Sensor Glucose (mg/dL)'].interpolate(method='quadratic').plot()
proc1.loc['2019/02/14 12':'2019/02/15 12']['Sensor Glucose (mg/dL)'].interpolate(method='cubic').plot()
proc1.loc['2019/02/14 12':'2019/02/15 12']['Sensor Glucose (mg/dL)'].interpolate(method='spline', order=2).plot()
[
    hybrid_interpolator(proc1.loc['2019/02/14 12':'2019/02/15 12']['Sensor Glucose (mg/dL)'], weights=w).plot()
    for w in weights_set
]
proc1.loc['2019/02/14 12':'2019/02/15 12']['Sensor Glucose (mg/dL)']
plt.axhline(200, color='red')
plt.axhline(70, color='green')
plt.legend(['Bolus', 'Carbs', 'Basal', 'Linear', 'Slinear', 'Quadratic', 'Cubic', 'spline', *labs, 'Data'])


# In[237]:


test_day = copy.deepcopy(proc1.loc['2019/02/14 12':'2019/02/14 22']['Sensor Glucose (mg/dL)'])
gap1 = copy.deepcopy(test_day)
gap2 = copy.deepcopy(test_day)


# In[266]:


proc2 = proc1.loc['2019/02/05':'2019/04/23']


# In[132]:


hybrid_interpolator(proc2.loc['2019/02/10':'2019/02/15']['Sensor Glucose (mg/dL)']).plot()
proc2.loc['2019/02/10':'2019/02/15']['Sensor Glucose (mg/dL)'].interpolate(method='polynomial', order=2).plot()
proc2.loc['2019/02/10':'2019/02/15']['Sensor Glucose (mg/dL)'].plot()


# In[133]:


list(filter(lambda x: not x, sorted(proc2.index) == proc2.index))


# In[134]:


df = copy.deepcopy(test_day)
x = df.isnull().astype(int).groupby(df.notnull().astype(int).cumsum()).sum()
#x


# In[238]:


gap1.loc['2019/02/14 20:00':'2019/02/14 20:30'] = np.nan
gap2.loc['2019/02/14 13:45':'2019/02/14 14:25'] = np.nan
#gap1.loc['2019/02/14 17:45':'2019/02/14 18:15'] = np.nan


# In[ ]:





# In[ ]:





# In[137]:


hybrid_proc1 = hybrid_interpolator(proc1.loc['2019/01':'2019/03']['Sensor Glucose (mg/dL)'])
linear_proc1 = proc1.loc['2019/01':'2019/03']['Sensor Glucose (mg/dL)'].interpolate(method='linear', limit_direction='both')
#proc1.loc['2019/01':'2019/03']['Sensor Glucose (mg/dL)'].rolling(2).mean().plot()
#proc1.loc['2019/01':'2019/03']['Sensor Glucose (mg/dL)']
#plt.legend(['Hybrid interpolator', 'Linear', 'Data'])


# In[143]:


dates: list = []
for i in range(1, 4):
    for j in range(1, 28):
        if j < 10:
            dates.append(f'2019/0{i}/0{j}')
        else:
            dates.append(f'2019/0{i}/{j}')


# In[233]:





# In[234]:


naive_hybrid_interpolator(proc1.loc['2019/02/01 10':'2019/02/05 23']['Sensor Glucose (mg/dL)']).plot()
[
    hybrid_interpolator(proc1.loc['2019/02/01 10':'2019/02/05 23']['Sensor Glucose (mg/dL)'], weights=w).plot()
    for w in weights_set
]
proc1.loc['2019/02/01 10':'2019/02/05 23']['Sensor Glucose (mg/dL)'].interpolate(method='linear').plot()
proc1.loc['2019/02/01 10':'2019/02/05 23']['Sensor Glucose (mg/dL)'].plot()
plt.legend(['Naive hybrid', *labs, 'Linear', 'Original Data'])


# In[282]:


proc1.loc['2019/03/01':'2019/03/08']['Sensor Glucose (mg/dL)'].count() / len(proc1.loc['2019/03/01':'2019/03/08']['Sensor Glucose (mg/dL)'])


# In[286]:





# In[288]:


porcentage_original(proc1['Sensor Glucose (mg/dL)'], proc1.index[0], proc1.index[-1]), porcentage_interpolated(proc1['Sensor Glucose (mg/dL)'], proc1.index[0], proc1.index[-1])


# In[148]:


for date in dates:
    hybrid_proc1[date].plot()
    proc1.loc[date]['Sensor Glucose (mg/dL)'].interpolate(method='linear').plot()
    proc1.loc[date]['Sensor Glucose (mg/dL)'].plot()
    plt.legend(['Hybrid interpolator', 'Linear', 'Data'])
    plt.show()


# In[ ]:





# In[149]:


#for i in pd.date_range(pd.to_datetime('2019/01/10'), periods=60).tolist():
 #   hybrid_interpolator(proc1.loc[i]['Sensor Glucose (mg/dL)']).plot()
  #  proc1.loc[i]['Sensor Glucose (mg/dL)'].plot()


# In[151]:


#hybrid_interpolator(proc1.loc['2019/02/05']['Sensor Glucose (mg/dL)']).plot()
#proc1.loc['2019/02/05']['Sensor Glucose (mg/dL)'].plot()


# In[152]:


#proc1.loc['2019/03/05 03':'2019/03/05 07']['Sensor Glucose (mg/dL)'].interpolate(method='linear').plot()
#hybrid_interpolator(proc1.loc['2019/03/05 03':'2019/03/05 07']['Sensor Glucose (mg/dL)']).plot()
#proc1.loc['2019/03/05 03':'2019/03/05 07']['Sensor Glucose (mg/dL)'].plot()


# In[153]:


#proc1.iloc[0, :], proc1.iloc[len(proc1.index)-1, :]


# In[154]:


#with open('dateTime.txt', 'w') as f:
#    for i in raw['dateTime']:
#        f.write(f'{i}\n')


# In[156]:


'''
list(
    map(
        type, list(raw[ raw['Sensor Glucose (mg/dL)'].notnull() ]['Sensor Glucose (mg/dL)'])
    )
)
'''


# In[157]:


#glucosas = raw[ raw['Sensor Glucose (mg/dL)'].notnull() ]['Sensor Glucose (mg/dL)']


# In[158]:


#glucosas = pd.to_numeric(glucosas, 'coerce')


# In[ ]:


#glucosas = glucosas[glucosas == glucosas // 1]


# In[160]:


#sb.distplot(glucosas)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[176]:


proporciones = (lambda glucosas, borne_inf, borne_sup1, borne_sup2: 
                { 
                    'hypo': (f'x <= {borne_inf}', 100 * glucosas[glucosas <= borne_inf].count() / glucosas.count()),
                    'normo': (f'{borne_inf} < x <= {borne_sup1}', 100 * glucosas[(glucosas > borne_inf) & (glucosas <=borne_sup1)].count() / glucosas.count()),
                    'hyper': (f'x > {borne_sup1}', 100 * glucosas[glucosas > borne_sup1].count() / glucosas.count()),
                    'hyper2': (f'x > {borne_sup2}', 100 * glucosas[glucosas > borne_sup2].count() / glucosas.count()),
                }
)(proc1['Sensor Glucose (mg/dL)'], 70, 140, 180)


# In[177]:


proporciones


# In[ ]:





# In[ ]:





# In[168]:


#raw.iloc[0,:], raw.iloc[len(raw.index)-1, :]
#data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
#data = pd.DataFrame(data, columns=['x', 'y'])
#data


# In[ ]:





# In[96]:


'''
glucosas = filter(lambda x: x if not np.isnan(x) else False,
                  list(
                    map(int, glucosas)
                    )
            )
#Value error: cannot convert float NaN to integer
'''    


# In[279]:


len(proc2.loc['2019/02/02':'2019/02/05']), len(proc2.loc['2019/02/02':'2019/02/05'].resample('1T').mean())


# In[ ]:





# In[254]:


proc1[ proc1['Sensor Glucose (mg/dL)'] == 140 ]


# In[ ]:




