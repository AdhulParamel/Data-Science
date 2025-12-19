#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')
import streamlit as st


# In[12]:


log=pickle.load(open('logis.pkl','rb'))


# In[13]:


st.title('Model Deployment using Logistic Regression')


# In[14]:


def user_input_parameter():
    Pregnancies= st.sidebar.number_input('Enter the pregnancies')
    Glucose= st.sidebar.number_input('Enter the Glucose')
    BloodPressure= st.sidebar.number_input('Enter the Blood Pressure')
    SkinThickness= st.sidebar.number_input('Enter the Skin Thickness')
    Insulin= st.sidebar.number_input('Enter the Insulin')
    BMI= st.sidebar.number_input('Enter the BMI')
    Diabetes= st.sidebar.number_input('Enter the Diabetes')
    Age= st.sidebar.slider('Select your Age',0,100)
    dict1= {'Pregnancies':Pregnancies,'Glucose':Glucose,'BloodPressure':BloodPressure,'SkinThickness':SkinThickness,'Insulin':Insulin,'BMI':BMI,'DiabetesPedigreeFunction':Diabetes,'Age':Age}
    features= pd.DataFrame(dict1, index=[0])
    return features
df= user_input_parameter()


# In[15]:


pred=log.predict(df)
pred_prob=log.predict_proba(df)
button=st.button('Predict')
if button is True:
    st.subheader('Predicted')
    st.write('Diabetes' if pred_prob[0][1]>=0.5 else 'Not Diabetes')
    st.subheader('Pred_Prob')
    st.write(pred_prob)


# In[16]:


df


# In[ ]:





# In[ ]:




