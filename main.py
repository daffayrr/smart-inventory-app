#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Smart Inventory Application')

# Upload file CSV
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file is not None:
    # Baca dataset
    train = pd.read_csv('train.csv', parse_dates=['date'])
    data = pd.read_csv(uploaded_file, parse_dates=['date'])
    df = pd.concat([train, data], sort=False)

    # Feature engineering
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df = df[~df.index.duplicated()]


    # Model training
    feature_cols = ['store', 'item', 'day', 'month', 'year', 'dayofweek', 'dayofyear', 'weekofyear']
    target_col = 'sales'
    lgb_train = lgb.Dataset(df[feature_cols], label=df[target_col])
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    st.write('Training model...')
    gbm = lgb.train(params, lgb_train, num_boost_round=100)

    # Prediction
    st.write('Predicting sales for next 30 days...')
    last_date = df['date'].max()
    dates = pd.date_range(last_date, periods=30, freq='D')[1:]
    pred_data = pd.DataFrame({'date': dates, 'sales': -1})
    pred_data['day'] = pred_data['date'].dt.day
    pred_data['month'] = pred_data['date'].dt.month
    pred_data['year'] = pred_data['date'].dt.year
    pred_data['dayofweek'] = pred_data['date'].dt.dayofweek
    pred_data['dayofyear'] = pred_data['date'].dt.dayofyear
    pred_data['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    pred_data['store'] = df['store'].unique()[0]
    pred_data['item'] = df['item'].unique()[0]
    pred_sales = gbm.predict(pred_data[feature_cols])


    # Plotting hasil prediksi
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, pred_sales)
    ax.set(xlabel='Date', ylabel='Sales',
           title='Predicted Sales for Next 30 Days')
    ax.grid()
    st.pyplot(fig)

