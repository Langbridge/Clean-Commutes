import requests
import json
import ast

from datetime import datetime, timedelta, tzinfo, timezone
import time
from pprint import pprint
from decouple import config

from decimal import Decimal
import aqi

import pandas as pd
import matplotlib.pyplot as plt

import boto3
from boto3.dynamodb.conditions import Key

# """
# FETCHING API & PREPPING FOR DDB PUSH
# """
# key = config('WAQI_ID')

# def get_info():
#     id = '@11767'
#     r = requests.get(f"https://api.waqi.info/feed/{id}/?token={key}")

#     if r.status_code != 200 or r.json()['status'] != 'ok':
#         raise RuntimeError('The API request failed.')
#     else:
#         return r.json()

# data = get_info()['data']
# pprint(data)
# mydict = {}
# mydict['timestamp'] = data['time']['iso']
# mydict['dominantpol'] = data['dominentpol']
# mydict['iaqi'] = data['iaqi']

# pprint(mydict)

# """
# FETCHING DYNAMODB RECORDS LIVE USING KEYS
# """
# TABLE_NAME = 'aqi_data'
# SAMPLE_TIME = 30

# dynamodb = boto3.resource('dynamodb')
# table = dynamodb.Table(TABLE_NAME)

# now             = int(time.time() * 1000)
# past_time       = int(now - SAMPLE_TIME*60000)

# fe       = Key('sample_time').between(past_time,now)
# response = table.scan(
#                 FilterExpression=fe
#             )

# recent_record = response['Items'][0]
# record = pd.json_normalize(recent_record)
# record.drop(columns=['device_data.time'], inplace=True)
# record['sample_time'] = pd.to_datetime(int(record['sample_time']),unit='ms')

# print(record.head())


"""
FETCHING LAST 20 RECORDS OF ESP32 DATA
"""
TABLE_NAME = 'aqi_data'
SAMPLE_TIME = 30
NUM_SAMPLES = 100

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(TABLE_NAME)

now             = int(time.time() * 1000)
past_time       = int(now - NUM_SAMPLES*(SAMPLE_TIME+5)*60000)

response = table.scan(
                FilterExpression = Key('sample_time').between(past_time,now)
            )

recent_records = response['Items']

records = pd.DataFrame()
for item in recent_records:
    record = pd.json_normalize(item)
    if records.empty:
        records = record
    else:
        records = records.append(record, ignore_index=True)


#records.drop(columns=['device_data.time'], inplace=True)
records = records.apply(pd.to_numeric, errors='ignore')
records['sample_time'] = pd.to_datetime(records['sample_time'],unit='ms')
records.sort_values(by='sample_time', ascending=False, inplace=True, ignore_index=True)

# print(records.head(10))

print(f"Number of out-of-bounds records: {max(len(records.loc[records['device_data.temp'] >= 150]), len(records.loc[records['device_data.humidity'] >= 0.999]))}")
records.loc[records['device_data.temp'] >= 150, 'device_data.temp'] = float('NaN')
records.loc[records['device_data.humidity'] >= 0.999, 'device_data.humidity'] = float('NaN')
records = records.fillna(method='bfill')

records['AQI_PM10'] = records.apply(lambda row: int(aqi.to_iaqi(aqi.POLLUTANT_PM10, row['device_data.PM10'])), axis=1)
records['AQI_PM25'] = records.apply(lambda row: int(aqi.to_iaqi(aqi.POLLUTANT_PM25, row['device_data.PM25'])), axis=1)

records['AQI'] = records.apply(lambda row: max(row['AQI_PM10'], row['AQI_PM25']), axis=1)
records['max_AQI'] = records.apply(lambda row: int(aqi.to_iaqi(aqi.POLLUTANT_PM25, row['device_data.PM25']+row['device_data.PM10'])), axis=1)


# records.plot(x='sample_time', y=['device_data.temp','device_data.PM25'])
# records.plot(x='sample_time', y=['AQI'])
# plt.show()



"""
FETCHING LAST X RECORDS OF API DATA - TO MATCH ABOVE
"""
TABLE_NAME = 'api_records'

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(TABLE_NAME)

now             = int(time.time() * 1000)
past_time       = int(now - NUM_SAMPLES*(SAMPLE_TIME+5)*60000)

response = table.scan(
                FilterExpression = Key('timestamp').between(past_time,now)
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
API_records['AQI'] = API_records.apply(lambda row: max(row['iaqi.PM10'], row['iaqi.PM25']), axis=1)

API_records.drop_duplicates(inplace=True, ignore_index=True)
API_records.sort_values(by='measure_time', ascending=False, inplace=True, ignore_index=True)

# print(API_records.head())
# API_records.plot(x='measure_time', y=['AQI'])
# plt.show()

# """
# PLOTTING MEASURED AGAINST API AQI
# """
df1 = API_records.resample('H', on='measure_time').mean().fillna(method='ffill')
df2 = records.resample('H', on='sample_time').mean()#.fillna(method='ffill')
df2.rename(columns={'AQI': 'measure_AQI'}, inplace=True)

#df2['AQI'] = df1['AQI']
df2['AQI'] = df1['iaqi.MaxPol']
print(df2)

df2['AQI_rolling'] = df2['max_AQI'].rolling(4).mean() #rolling average window in hours

df2.plot(y=['AQI','max_AQI','measure_AQI','AQI_rolling'])

fig, ax = plt.subplots()
fig.subplots_adjust(right=0.7)
df2['device_data.humidity'].plot(ax=ax, style='b-')
df2['device_data.temp'].plot(ax=ax, style='r-', secondary_y=True)
ax.legend([ax.get_lines()[0], ax.right_ax.get_lines()[0]], ['humidity', 'temp'])
ax.set_ylim([0.5, 1.0])
ax.right_ax.set_ylim([0, 15])

plt.show()



# """
# PARSING, READING & PROCESSING TEST CSV DATA DOWNLOADED MANUALLY
# """
# df = pd.read_csv('testdb_results.csv')

# df2 = json.loads(df['device_data'].to_json(orient='index'))
# for key,item in df2.items():
#     for nested_key, nested_item in json.loads(item).items():
#         df.loc[int(key), nested_key] = float(nested_item.get("N"))
        
# df.drop(columns=['device_data','time'], inplace=True)
# df['sample_time'] = pd.to_datetime(df['sample_time'],unit='ms')
# df['total_PM'] = df['PM25'] + df['PM10']

# # print(df.head())

# df.plot(x='sample_time', y=['PM25','PM10'])
# df.plot(x='sample_time', y=['temp','total_PM'])

# plt.show()