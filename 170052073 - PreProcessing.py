#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


# 1. Read the stroke csv file from my folder

# In[41]:


df = pd.read_csv("stroke.csv")


# In[42]:


df


# # Exploratory data analysis

# 2. Get the information --> we can see that BMI have some missin values

# In[43]:


df.info()


# In[ ]:





# 3. See how many missing values, and which columnn have/has missing values

# In[44]:


df.isnull().sum()


# In[ ]:





# # Handling the Missing Values --> BMI

# 4. This shows the number of people who has BMI --> Looking at this most of the people (41) have a BMI of 28.7

# In[45]:


df["bmi"].value_counts()


# In[ ]:





# 5. Describe function let's me see the values for BMI --> STD, mean, min, max etc. 

# In[46]:


df["bmi"].describe()


# In[ ]:





# 6. Fill the null values with the mean instead of removing them as BMI is a factor for stroke

# In[47]:


df["bmi"].fillna(df["bmi"].mean(), inplace=True)


# In[48]:


df["bmi"].describe()


# In[ ]:





# 7. Double checking to see if all the null values have replaced. 

# In[49]:


df.isnull().sum()


# In[ ]:





# In[ ]:





# # Feature Selection

# 8. Id column is not necessary, hence removing the ID column

# In[50]:


df.drop("id", axis = 1, inplace = True) 


# In[51]:


df


# In[ ]:





# In[ ]:





# # Outlier Removal

# 9. Values that are far away and not related as well as the values that are out of the box will be the outliers.The outliers will not be removed as both BMI and glucose level will have an impact on the stroke prediction 
# 

# In[52]:


plt.figure(figsize=(30,20))
df.boxplot()


# In[ ]:





# # Label Encoding

# 10. This is the section where all the classes will turn into binary. 

# In[53]:


df.head()


# In[ ]:





# 11. Checking how many work types there are

# In[54]:


df["work_type"].unique()


# In[ ]:





# 12. Importing LabelEncoder from sklearn's preprocessing. 

# In[55]:


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()


# In[ ]:





# 13. Labelling each column that has a classification. 

# In[56]:


gender = enc.fit_transform(df["gender"])
smoking_status = enc.fit_transform(df["smoking_status"])
work_type = enc.fit_transform(df["work_type"])
Residence_type = enc.fit_transform(df["Residence_type"])
ever_married = enc.fit_transform(df["ever_married"])


# In[ ]:





# 14. Replacing the labelling 

# In[57]:


# replacing the labelling
df["work_type"] = work_type
df["gender"] = gender
df["smoking_status"] = smoking_status
df["Residence_type"] = Residence_type
df["ever_married"] = ever_married


# In[58]:


df.to_csv("stroke_prediction.csv")


# In[ ]:





# # Partitioning --> Splitting the data for train and test

# X --- X_train, X_test 75(training)/25(testing)
# 
# y --- y_train, y_test

# In[59]:


X = df.drop("stroke", axis = 1)
y= df["stroke"]


# In[60]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 101)


# In[61]:


X_train


# In[62]:


X_test


# In[ ]:





# In[ ]:





# # Normalisation 

# In[63]:


df.describe()


# In[64]:


from sklearn.preprocessing import StandardScaler
std = StandardScaler()


# In[65]:


X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)


# In[66]:


X_train


# In[67]:


X_test


# In[ ]:





# # Training

# # Decision Tree

# In[69]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()


# In[70]:


dt.fit(X_train, y_train)


# In[71]:


dt.feature_importances_


# In[72]:


y_pred = dt.predict(X_test)


# In[73]:


from sklearn.metrics import accuracy_score


# In[74]:


dt_ac = accuracy_score(y_test, y_pred)


# In[75]:


dt_ac


# # Logistic Regression

# In[76]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[83]:


lr.fit(X_train, y_train)


# In[84]:


y_pred_lr = lr.predict(X_test)


# In[85]:


y_test


# In[87]:


lr_ac = accuracy_score(y_test, y_pred_lr)


# In[88]:


lr_ac

