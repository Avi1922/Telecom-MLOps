#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[7]:


import pandas as pd


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


import joblib


# In[10]:


import json


# ### Dataset

# In[11]:


df=pd.read_csv("Telecom_Tower_Failure_Dataset_10000-1.csv")
df.head()


# In[12]:


X=df[['Temperature_C','Battery_Voltage','Power_Consumption_W','Signal_Strength_Percent','Fan_Speed_RPM','Humidity_Percent','Traffic_Load','Tower_Age_Years']]
y=df['Failure_Within_48Hrs']


# In[13]:


print(X)


# In[14]:


print(y)


# ### Train - Test Split

# In[15]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


# In[16]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ### Accuracy

# In[17]:


model.fit(X_train,y_train)
accuracy=model.score(X_test,y_test)
print("Accuracy of random forest",accuracy)


# ### Saving the model

# In[18]:


joblib.dump(model,"telecom_tower_model.pkl")


# In[19]:


metrics={"accuracy":accuracy}
with open("metrics.json","w") as f:
    json.dump(metrics,f,indent=4)
print("Training Completed Successfully")


# In[ ]:




