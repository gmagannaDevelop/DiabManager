#!/usr/bin/env python
# coding: utf-8

# In[18]:


from functools import wraps
from time import time, sleep
import datetime as dt


# In[13]:


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' %               (f.__name__, args, kw, te-ts)
        )
        return result
    return wrap


# In[21]:


@timing
def lol(s):
    print(s)
    
@timing
def lal(n, times):
    for i in range(times):
        sleep(n)
    print(f'Slept for {n} seconds')


# In[22]:


lol(3)


# In[24]:


lal(3, 3)


# In[ ]:




