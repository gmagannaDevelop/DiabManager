#!/usr/bin/env python
# coding: utf-8

# In[2]:


import Preprocessing as pre


# In[3]:


# Code to download the dataset and preprocess it 
# should be placed here.


# In[ ]:


x  = get_paired_measurements(data)
ig = [ i[0] for i in x ]
bg = [ i[1] for i in x ] 


# In[ ]:


t_test = stats.ttest_rel(ig, bg)
t_test


# In[ ]:


correlation = stats.pearsonr(ig, bg)
correlation


# In[ ]:


difs = list(map(lambda x, y: x - y, ig, bg))


# In[ ]:


sb.distplot(difs)


# In[ ]:


stats.shapiro(difs)


# In[ ]:


stats.wilcoxon(difs)


# In[ ]:


ax = sb.distplot(ig, color="Red")
ax = sb.distplot(bg, color="Blue")

