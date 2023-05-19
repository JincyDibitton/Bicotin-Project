#!/usr/bin/env python
# coding: utf-8

# # Bicotin Project

# In[95]:


# import libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[52]:


#Load dataset
df=pd.read_csv('BitcoinPrice.csv')


# In[53]:


df.head()


# In[54]:


df.tail()


# In[55]:


df.info()


# In[56]:


df.drop(['Date'],1,inplace=True)


# In[57]:


p_days=30


# In[58]:


df['Prediction']=df[['Price']].shift(-p_days)


# In[59]:


df.head()


# In[60]:


df.tail()


# In[61]:


#seperate x and y


# In[62]:


X=np.array(df.drop(['Prediction'],1))


# In[63]:


X=X[:len(df)-p_days]


# In[64]:


X.shape


# In[65]:


X


# In[66]:


Y=np.array(df['Prediction'])


# In[67]:


Y=Y[:-p_days]


# In[68]:


Y.shape


# In[69]:


Y


# In[70]:


#split the data


# In[71]:


from sklearn.model_selection import train_test_split


# In[72]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)


# In[85]:


p_days_array=np.array(df.drop(['Prediction'],1))[-p_days]
print(p_days_array)


# In[86]:


#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor


# In[87]:


Rf=RandomForestRegressor(n_estimators=1000,random_state=1)
Rf.fit(X_train,Y_train)


# In[88]:


print('Random Forest Accuracy:{:.2f} %'.format(Rf.score(X_test,Y_test)*100))


# In[89]:


#prediction
Rf_predict=Rf.predict(X_test)
print(Rf_predict)


# In[90]:


print(Y_test)


# In[93]:


#model prediction for 30 days
Rf_predict_30=Rf.predict([p_days_array])
print(Rf_predict)


# In[94]:


#orginal value
df.head()


# In[ ]:




