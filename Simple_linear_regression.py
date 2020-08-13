#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# loading dataset
df=pd.read_csv("Salary_Data.csv")


# In[3]:


# first 5 rows
df.head()


# In[4]:


# shape of Dataset
df.shape


# In[5]:


# checking missing values in dataset
df.isnull().sum()


# In[6]:


#checking datatype of columns
df.info()


# In[14]:


# spliting data into independent and dependent variables
X = df.drop(["Salary"],axis=1)
y = df.Salary


# In[15]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.03,random_state=355)


# In[16]:


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)


# In[21]:


# Predicting the Test set results
y_pred=lr.predict(X_test)


# In[22]:


# Visualising the Training set results
plt.scatter(X_train,y_train,color="red")
plt.plot(X_train, lr.predict(X_train), color = 'blue')


# In[27]:


# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, lr.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:




