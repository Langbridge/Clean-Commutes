import time
import json
import ast

from flask import Flask, jsonify, request
from journey import RouteFactory
from datetime import datetime
import aqi

import boto3
from boto3.dynamodb.conditions import Key

import pandas as pd

OBJECTIVES = {0: {'attr': 'duration', 'rev': False}, 1: {'attr': 'rdd', 'rev': False}, 2: {'attr': 'fare', 'rev': False}}

app = Flask(__name__)
app.config["DEBUG"] = True

dynamodb = boto3.resource('dynamodb')

@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The requested resource could not be found.</p>", 404

@app.route('/', methods=['GET'])
def home():
    return "<h1>Real-Time Routing</h1><p>This site is a prototype API for real-time routing.</p>"

def api_all(routes):
    results = [route.__dict__ for route in routes.values()]
    return jsonify(results)

@app.route('/routes', methods=['GET'])
def api_mode():
    start = request.args.get('start', None)
    end = request.args.get('end', None)
    mode = request.args.get('mode', None)

    A = RouteFactory(start, end)
    routes = A.get_routes()

    if mode:
        mode = str(request.args['mode']).lower()
    else:
        return api_all(routes)

    results = []
    for route in routes.values():
        if mode in route.txt_mode:
            results.append(route.__dict__)

    return jsonify(results)

@app.route('/AQI_local', methods=['GET'])
def get_background_exposure(mode=0):
    num  = request.args.get('num', 24)

    NUM_SAMPLES = int(num)
    TABLE_NAME = 'aqi_data'
    SAMPLE_TIME = 30

    table = dynamodb.Table(TABLE_NAME)

    now             = int(time.time() * 1000)
    past_time       = int(now - NUM_SAMPLES*(SAMPLE_TIME*3)*60000)

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

    if records.empty and mode:
        API_AQI = get_api_aqi(mode=1)
        print("USING API DATA:", API_AQI)
        return API_AQI

    records = records.apply(pd.to_numeric, errors='ignore')
    records['sample_time'] = pd.to_datetime(records['sample_time'],unit='ms')
    records.sort_values(by='sample_time', ascending=False, inplace=True, ignore_index=True)

    records['AQI'] = records.apply(lambda row: int(aqi.to_iaqi(aqi.POLLUTANT_PM25, row['device_data.PM25']+row['device_data.PM10'])), axis=1)

    if mode:
        print("USING ESP DATA: ", records['AQI'].iloc[0])
        return records['AQI'].iloc[0]

    return records.head(2*NUM_SAMPLES).to_json(orient='records', date_format='iso')

@app.route('/routes/suggest', methods=['GET'])
def api_suggest():
    start = request.args.get('start', None)
    end = request.args.get('end', None)

    age = request.args.get('age', 35)
    sex = request.args.get('sex', 'M')
    mass = request.args.get('mass', 70)

    str_obj = str(request.args.get('objective', 0))
    obj = [int(val) for val in str_obj]

    # OBJECTIVE DICT:
    # 0: TIME
    # 1: EXPOSURE
    # 2: COST

    A = RouteFactory(start, end, get_background_exposure(mode=1), user={'age': int(age), 'sex': sex, 'mass': int(mass)})

    routes = A.get_routes()
    route_list = list(routes.values())

    rank = dict.fromkeys(route_list, 0)
    for objective in obj:
        sorted_routes = sorted(route_list, key=lambda x: getattr(x, OBJECTIVES[objective]['attr']), reverse=OBJECTIVES[objective]['rev'])
        for i, sorted_route in enumerate(sorted_routes):
            rank[sorted_route] += i

    ranked_routes = sorted(rank.items(), key=lambda item: item[1])

    results = {'routes': [route[0].__dict__ for route in ranked_routes]}
    return jsonify(results)

@app.route('/AQI', methods=['GET'])
def get_api_aqi(mode=0):
    num  = request.args.get('num', 24)
    NUM_SAMPLES = int(num)

    TABLE_NAME = 'api_records'
    SAMPLE_TIME = 60
    table = dynamodb.Table(TABLE_NAME)

    now             = int(time.time() * 1000)
    past_time       = int(now - NUM_SAMPLES*(SAMPLE_TIME*1.5)*60000)

    response = table.scan(
                    FilterExpression = Key('timestamp').between(past_time,now)
                )

    recent_records = response['Items']

    API_records = pd.DataFrame()
    for item in recent_records:
        record = pd.json_normalize(item)

        iaqi = record['iaqi'].values[0]
        iaqi = json.loads(json.dumps(ast.literal_eval(iaqi)))

        record['PM10'] = iaqi['pm10']['v']
        record['PM25'] = iaqi['pm25']['v']

        if API_records.empty:
            API_records = record
        else:
            API_records = API_records.append(record, ignore_index=True)

    API_records.drop(columns=['timestamp','iaqi','dominantpol'], inplace=True)
    API_records = API_records.apply(pd.to_numeric, errors='ignore')

    API_records['measure_time'] = pd.to_datetime(API_records['measure_time'])
    API_records['measure_time'] = API_records['measure_time'].apply(lambda x: x.replace(tzinfo=None))

    API_records.drop_duplicates(inplace=True, ignore_index=True)
    API_records.sort_values(by='measure_time', ascending=False, inplace=True, ignore_index=True)

    if mode:
        return API_records['AQI'].iloc[0]
    
    return API_records.head(NUM_SAMPLES).to_json(orient='records', date_format='iso')

app.run(port=5001)