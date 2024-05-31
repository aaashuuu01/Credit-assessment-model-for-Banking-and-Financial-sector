#!/usr/bin/env python
# coding: utf-8

# # Step 1: Data Loading

# In[30]:


import pandas as pd

# Load the dataset
url = "https://github.com/YBIFoundation/Dataset/raw/main/Credit%20Default.csv"
data = pd.read_csv(url)

# Display the first few rows of the dataset
print(data.head())


# # Step 2: Data Preprocessing

# In[31]:


# Check for missing values
print(data.isnull().sum())

# Handle missing values (if any)

# Encode categorical variables (if any)
# You can use techniques like one-hot encoding or label encoding

# Split data into features and target variable
X = data.drop('Default', axis=1)  # Features
y = data['Default']  # Target variable


# # Step 3: Feature Engineering

# In[36]:


# Perform feature engineering (if needed)
# This can include feature scaling, normalization, creating interaction terms, etc.


# # Step 4: Model Selection

# In[37]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier()  # You can try different models here
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))


# # Step 5: Model Evaluation and Optimization

# In[38]:


# Perform hyperparameter tuning or model optimization (if needed)
# This can be done using techniques like GridSearchCV or RandomizedSearchCV


# # Step 6: Deployment

# In[39]:


# Save the model for future use
import joblib
joblib.dump(model, 'credit_assessment_model.pkl')

# Later, you can load the model and use it for predictions
# loaded_model = joblib.load('credit_assessment_model.pkl')
# result = loaded_model.predict(new_data)


# In[ ]:




