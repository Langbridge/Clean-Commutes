from flask import Flask, jsonify, request
from flask import render_template
from datetime import time
import random
import json
import requests
import pandas as pd
import aqi
 
app = Flask(__name__)

colours = [('rgba(255, 99, 132, 0.2)', 'rgba(255, 99, 132, 1)'),
           ('rgba(255, 159, 64, 0.2)', 'rgba(255, 159, 64, 1)'),
           ('rgba(255, 205, 86, 0.2)', 'rgba(255, 205, 86, 1)'),
           ('rgba(75, 192, 192, 0.2)', 'rgba(75, 192, 192, 1)'),
           ('rgba(54, 162, 235, 0.2)', 'rgba(54, 162, 235, 1)'),
           ('rgba(153, 102, 255, 0.2)', 'rgba(153, 102, 255, 1)'),
           ('rgba(201, 203, 207, 0.2)', 'rgba(201, 203, 207, 1)')]

def get_routes(start, end, user, objective):
    route = {}

    payload = {'start': start, 'end': end, 'objective': '01', 'age': user['age'], 'sex': user['sex'], 'mass': user['mass'], 'objective': objective}
    r = requests.get(f"http://127.0.0.1:5001/routes/suggest", params=payload)

    if r.status_code != 200:
        raise RuntimeError('The API request failed.')

    data = r.json()['routes']

    for i in range(len(data) - 1):
        route[i] = {}
        route[i]['data'] = [data[i]['duration'], data[i]['fare']/100, round(data[i]['rdd'],3)]
        route[i]['label'] = data[i]['txt_mode']
        route[i]['arrive'] = data[i]['arrive_time']

        if i==0:
            route[i]['follow'] = [leg[1] for leg in data[i]['legs']]
   
    return route

def get_AQI():
    route = {}

    payload = {'num': 24}
    r = requests.get(f"http://127.0.0.1:5001/AQI_local", params=payload)
    r2 = requests.get(f"http://127.0.0.1:5001/AQI", params=payload)

    if r.status_code != 200:
        raise RuntimeError('The API request failed.')

    local_data = pd.DataFrame(r.json())
    API_data = pd.DataFrame(r2.json())

    print(len(local_data), len(API_data))

    local_data['sample_time'] = pd.to_datetime(local_data['sample_time'])
    local_data = local_data.resample('H', on='sample_time').mean()
    local_data.loc[local_data['device_data.temp'] >= 50, 'device_data.temp'] = float('NaN')
    local_data.loc[local_data['device_data.humidity'] >= 0.999, 'device_data.humidity'] = float('NaN')
    local_data = local_data.fillna(method='bfill')
    local_data['total_PM'] = local_data.apply(lambda row: min(row['device_data.PM10']+row['device_data.PM25'], 100), axis=1)
    local_data['AQI'] = local_data.apply(lambda row: int(aqi.to_iaqi(aqi.POLLUTANT_PM25, row['total_PM'])), axis=1)

    API_data['measure_time'] = pd.to_datetime(API_data['measure_time'])
    API_data = API_data.resample('H', on='measure_time').mean()
    API_data = API_data.fillna(method='ffill')

    route_df = API_data[['PM25']]
    route_df['Measured PM25'] = local_data['AQI']
    route_df = route_df.dropna()

    route_df['AQI_rolling'] = route_df['Measured PM25'].rolling(4).mean()

    route[0] = {}
    route[0]['data'] = route_df['PM25'].tolist()
    route[0]['label'] = 'PM2.5 AQI @ Hounslow Chiswick'

    route[1] = {}
    route[1]['data'] = route_df['Measured PM25'].tolist()
    route[1]['label'] = 'Measured PM2.5 AQI'

    route[2] = {}
    route[2]['data'] = route_df['AQI_rolling'].tolist()
    route[2]['label'] = 'Measured PM2.5 AQI (4h Rolling Avg)'

    labels = route_df.index.tolist()

    return labels, route
 
@app.route("/")
def homepage():        
    return render_template('home.html', bar_data=plot_routes(0, labels=True), line_data=plot_AQI())

@app.route("/test", methods=['GET'])
def plot_routes(num=3, labels=False):
    start = request.args.get('start')
    end = request.args.get('end')
    user = {}
    user['age'] = request.args.get('age')
    user['sex'] = request.args.get('sex')
    user['mass'] = request.args.get('mass')
    objective = request.args.get('obj')

    dict = {'datasets': []}

    if num == 0 and labels:
        dict['labels'] = ['Duration (min)', 'Fare (£)', 'RDD (ug)']
        return dict

    count = 0
    route = get_routes(start, end, user, objective)
    local_cols = colours.copy()


    for key, val in route.items():
        if count >= num: break
        col = random.choice(local_cols)
        local_cols.remove(col)

        data = {
                'label': val['label'],
                'borderColor': col[1],
                'borderWidth': 1,
                'backgroundColor': col[0],
                'data': val['data']
                }

        dict['datasets'].append(data)
        count += 1

    follow_instr = ""
    for leg in route[0]['follow']:
        follow_instr += leg+", "

    dict['follow'] = follow_instr[:-2]

    if labels: dict['labels'] = ['Duration (min)', 'Fare (£)', 'RDD (ug)']
    return(dict)

def plot_AQI():
    count = 0
    labels, route = get_AQI()
    local_cols = colours.copy()

    AQI_dict = {'datasets': []}
    for key, val in route.items():
        col = random.choice(local_cols)
        local_cols.remove(col)

        data = {
                'label': val['label'],
                'borderColor': col[1],
                'borderWidth': 1,
                'backgroundColor': col[0],
                'data': val['data']
                }

        AQI_dict['datasets'].append(data)
        count += 1

    AQI_dict['labels'] = labels
    return(AQI_dict)

if __name__ == "__main__":
    app.run(debug=True)