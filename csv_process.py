import pandas as pd
import numpy as np
import json
import ast

import matplotlib.pyplot as plt
import seaborn as sns
import aqi
from datetime import datetime

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import boto3
from boto3.dynamodb.conditions import Key

def fetch_api_data(starttime, endtime):
    TABLE_NAME = 'api_records'

    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(TABLE_NAME)

    response = table.scan(
                    FilterExpression = Key('timestamp').between(starttime,endtime)
                )

    recent_records = response['Items']

    API_records = pd.DataFrame()
    for item in recent_records:
        record = pd.json_normalize(item)

        iaqi = record['iaqi'].values[0]
        iaqi = json.loads(json.dumps(ast.literal_eval(iaqi)))

        record['iaqi.PM10'] = iaqi['pm10']['v']
        record['iaqi.PM25'] = iaqi['pm25']['v']
        try:
            record['iaqi.MaxPol'] = iaqi[record['dominantpol'].values[0]]['v']
        except KeyError:
            record['iaqi.MaxPol'] = float('NaN')

        if API_records.empty:
            API_records = record
        else:
            API_records = API_records.append(record, ignore_index=True)

    API_records.drop(columns=['timestamp','iaqi'], inplace=True)
    API_records = API_records.apply(pd.to_numeric, errors='ignore')
    API_records['measure_time'] = pd.to_datetime(API_records['measure_time'])
    API_records['measure_time'] = API_records['measure_time'].apply(lambda x: x.replace(tzinfo=None))
    API_records = API_records.resample('H', on='measure_time').mean()

    API_records['AQI'] = API_records.apply(lambda row: max(row['iaqi.PM10'], row['iaqi.PM25']), axis=1)

    return API_records

def pearson_correlate(df, c1, c2):
    df = df.dropna()
    corr, _ = pearsonr(df[c1], df[c2])
    print(f"Pearson's Correlation between {c1} and {c2} is: {corr}")
    return corr

def spearman_correlate(df, c1, c2):
    df = df.dropna()

    corr, _ = spearmanr(df[c1], df[c2])
    print(f"Spearman's Correlation between {c1} and {c2} is: {corr}")
    return corr

def show_heatmap(data):
    # data['unix_time'] = data.apply(lambda row: int((row.name - datetime(1970,1,1)).total_seconds()), axis=1)
    data['hour'] = data.index.hour
    corr = data.corr(method='spearman')
    corr2 = corr[corr < 1].unstack().transpose().sort_values( ascending=False).drop_duplicates()
    print(corr2.head(10))

    fig, ax = plt.subplots()
    
    sns.heatmap(corr, cmap="Greens", annot=True, ax=ax)
    ax.set_xticklabels(data.columns, fontsize=10, rotation=90)
    fig.gca().xaxis.tick_bottom()
    ax.set_yticklabels(data.columns, fontsize=10)

    fig.tight_layout(pad=2)
    fig.suptitle("Feature Correlation Heatmap", fontsize=14)
    fig.savefig('Report/corrmap.jpg')

def plot_humtemp(df):
    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.7)

    df['humidity'].plot(ax=ax, style='b-')
    df['temp'].plot(ax=ax, style='r-', secondary_y=True)
    ax.legend([ax.get_lines()[0], ax.right_ax.get_lines()[0]], ['Humidity', 'Temp.'], loc='best')
    ax.set_ylim([0.5, 1.0])
    ax.right_ax.set_ylim([0, 15])

    ax.set_xlabel('Date/Time')
    ax.set_ylabel('Relative Humidity (%)')
    ax.right_ax.set_ylabel('Temperature (deg C)')

    fig.suptitle("Temperature & Humidity Over Time", fontsize=14)
    fig.savefig('Report/corrmap.jpg')

def prep_df(file):
    df = pd.read_csv(file)

    df2 = json.loads(df['device_data'].to_json(orient='index'))
    for key,item in df2.items():
        for nested_key, nested_item in json.loads(item).items():
            df.loc[int(key), nested_key] = float(nested_item.get("N"))
            
    df.drop(columns=['device_data'], inplace=True)
    df['sample_time'] = pd.to_datetime(df['sample_time'],unit='ms')
    print(f"{len(df)} unique datapoints collected with {max(len(df.loc[df['temp'] >= 150]), len(df.loc[df['humidity'] >= 0.999]))} out of bounds.")
    df = df.resample('H', on='sample_time').mean()

    df.loc[df['temp'] >= 50, 'temp'] = float('NaN')
    df.loc[df['humidity'] >= 0.999, 'humidity'] = float('NaN')
    #df = df.fillna(method='bfill')
    df = df.dropna()

    df['total_PM'] = df.apply(lambda row: min(row['PM10']+row['PM25'], 100), axis=1)
    df['AQI'] = df.apply(lambda row: int(aqi.to_iaqi(aqi.POLLUTANT_PM25, row['total_PM'])), axis=1)

    return df

def plot_loss(history):
    fig, ax = plt.subplots()

    ax.plot(history.history['loss'], label='loss')
    ax.plot(history.history['val_loss'], label='val_loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    ax.legend(['Loss', 'Val Loss'])
    ax.grid(True)

def lin_model(norm, train_features):
    linear_model = tf.keras.Sequential([norm, layers.Dense(units=1)])
    linear_model.predict(train_features)
    linear_model.compile(optimizer = tf.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error')
    return linear_model

def deep_model(norm, train_features):
    model = tf.keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)])
    model.predict(train_features)
    model.compile(optimizer = tf.optimizers.Adam(learning_rate=0.0005), loss='mean_absolute_error')
    return model

def regression(df, y_col, mode, plot=True):
    df = df.dropna()

    train_features = df.sample(frac=0.8, random_state=0)
    test_features = df.drop(train_features .index)
    train_labels = train_features.pop(y_col)
    test_labels = test_features.pop(y_col)

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))

    if mode.lower() == 'linear': model = lin_model(normalizer, train_features)
    elif mode.lower() == 'deep': model = deep_model(normalizer, train_features)
    else: return None

    history = model.fit(
        train_features,
        train_labels,
        epochs=200,
        verbose=0,
        validation_split = 0.2)

    if plot: plot_loss(history)

    test_predictions = model.predict(test_features).flatten()
    error = test_predictions - test_labels
    fig, ax = plt.subplots()
    ax.hist(error, bins=25)
    ax.set_xlabel(mode+' NN Prediction Error')
    ax.set_ylabel('Count')

    if mode.lower() == 'linear': print(model.layers[1].get_weights())

    return model.evaluate(test_features, test_labels, verbose=0)

def norm_col(df, col):
    return (df[col]-df[col].min())/(df[col].max()-df[col].min())

df = prep_df('Results/6-12_to_13-12_FINAL.csv')

df['AQI_rolling'] = df['AQI'].rolling(8).mean().shift(-4) #rolling average window in hours

starttime = int((df.iloc[0].name - datetime(1970,1,1)).total_seconds())
endtime = int((df.iloc[-1].name - datetime(1970,1,1)).total_seconds())
API_records = fetch_api_data(starttime*1000, endtime*1000)
df['API_AQI'] = API_records['iaqi.MaxPol']
df['API_AQI'] = df['API_AQI'].fillna(method='ffill').fillna(method='bfill')
df['hour'] = df.index.hour

print(f"{len(df)} hours of data from {df.iloc[0].name} to {df.iloc[-1].name}.")

# pearson_correlate(df, 'AQI_rolling', 'API_AQI')
# pearson_correlate(df, 'AQI_rolling', 'AQI')
# pearson_correlate(df, 'AQI', 'API_AQI')

# spearman_correlate(df, 'AQI', 'API_AQI')
# spearman_correlate(df, 'AQI_rolling', 'AQI')
# spearman_correlate(df, 'AQI_rolling', 'API_AQI')

show_heatmap(df[['temp','humidity','PM25','PM10']])
show_heatmap(df[['AQI','AQI_rolling','API_AQI']])
plot_humtemp(df)

# ----- HISTOGRAMS
# df[['temp','humidity','PM25','PM10','AQI']].hist(bins=40)

plot = sns.distplot(df['PM10'], kde=True, hist_kws=dict(edgecolor="k", linewidth=0))
plot.set_xlabel('PM10, ug/m3')
plot.set_ylabel('Count')

fig, ax = plt.subplots()
sns.distplot(df['PM25'], kde=True, hist_kws=dict(edgecolor="k", linewidth=0), ax=ax)
sns.distplot(df['PM10'], kde=True, hist_kws=dict(edgecolor="k", linewidth=0), ax=ax)
ax.legend(['PM2.5', 'PM10'])


# ----- CORREL & DISTRIBUTIONS
sns.pairplot(df[['temp','humidity','PM25','PM10']], diag_kind='kde')

# ----- MODEL
learn_results = {}

learn_df = df[['temp','humidity','PM25','PM10','AQI','AQI_rolling','API_AQI']]
learn_df = df[['temp','humidity','PM25','PM10','AQI','API_AQI']]
learn_df['hour'] = learn_df.index.hour

learn_results['linear'] = 1 - (regression(learn_df, 'API_AQI', 'Linear') / np.mean(df['API_AQI']))
learn_results['deep'] = 1 - (regression(learn_df, 'API_AQI', 'Deep') / np.mean(df['API_AQI']))

print(learn_results)

# ----- AUTOCORRELATIONS
plot_acf(df['temp'].dropna(), lags=24, title='Temperature Autocorrelation')
plot_acf(df['AQI'], lags=24, title='Measured AQI Autocorrelation')
plot_acf(df['AQI_rolling'].dropna(), lags=24, title='Rolling Average AQI Autocorrelation')
plot_acf(df['API_AQI'].dropna(), lags=24, title='API AQI Autocorrelation')

# ----- SEASONAL DECOMPOSITION
res = sm.tsa.seasonal_decompose(df['AQI_rolling'].dropna(), period=24)
res.plot()

# ----- DISTRIBUTION CHARACTERISTICS
print(f"AQI_rolling mean: {df['AQI_rolling'].mean()}")
print(f"AQI_rolling stdev: {df['AQI_rolling'].std()}")
print(f"AQI mean: {df['AQI'].mean()}")
print(f"AQI stdev: {df['AQI'].std()}")

plt.show()

