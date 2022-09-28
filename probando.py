

from flask import Flask, request, jsonify
import os
import pickle
import pandas as pd
import sqlite3
import numpy as np
from sklearn.model_selection import cross_val_score


os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/", methods=['GET'])
def hello():
    return "<h2>Bienvenido a la API advertising de Enrique Rubio<h2>"


# extra
@app.route('/v1/table', methods=['GET'])
def extra():
    model = pickle.load(open('advertising_model','rb'))

    # tv = request.args.get('tv', None)
    # radio = request.args.get('radio', None)
    # newspaper = request.args.get('newspaper', None)
    connection = sqlite3.connect('advertising_sales2.db')
    cursor = connection.cursor()
    query= 'SELECT TV, radio, newspaper from s'
    result = cursor.execute(query,).fetchall()
    
    predict = model.predict(result)
    predict = np.round(predict,2)
    df = pd.DataFrame(predict, columns=['sales_predict'])
    df[['TV','radio', 'newspaper']] = result
    df2 = df.to_html()
    connection.close()
    return df2

# primero

@app.route('/v1/predict', methods=['GET'])
def predict():
    model = pickle.load(open('advertising_model_v3','rb'))
    tv = request.args['tv']
    radio = request.args['radio']
    newspaper = request.args['newspaper']
    
    if tv is None or radio is None or newspaper is None:
        return "Missing args, the input values are needed to predict"
    else:
        prediction = model.predict([[tv,radio,newspaper]])
        return "<h3>The prediction of sales investing that amount of money in TV, radio and newspaper is:<h3>" + str(round(prediction[0],2)) + ' €'


# segundo
    
@app.route('/v2/insertdata', methods=['GET','PUT'])
def re():
    tv = request.args['tv']
    radio = request.args['radio']
    newspaper = request.args['newspaper']
    sales = request.args['sales']
    
    connection = sqlite3.connect('advertising_sales3.db')
    cursor = connection.cursor()
    query = "INSERT INTO r (TV, radio, newspaper,sales) VALUES (?, ?, ?, ?)"
    
    result = cursor.execute(query, (tv, radio, newspaper,sales,)) #.fetchall()
    connection.commit()
    connection.close()
    return '<h2>guardado<h2>'
    

# tercero
    
@app.route('/v3/retrain', methods=['GET'])
def retrain():
    model = pickle.load(open('advertising_model_v3','rb'))

    connection = sqlite3.connect('advertising_sales3.db')
    cursor = connection.cursor()
    query= 'SELECT TV, radio, newspaper, sales from r'
    result = cursor.execute(query,).fetchall()

    df = pd.DataFrame(result, columns=['TV', 'radio', 'newspaper', 'sales'])
    df['TV'] =df['TV'].apply(lambda x: float(x))
    df['radio'] =df['radio'].apply(lambda x: float(x))
    df['newspaper'] =df['newspaper'].apply(lambda x: float(x))
    df['sales'] =df['sales'].apply(lambda x: float(x))

    X = df.drop(columns=['sales'])
    y = df['sales']

    model.fit(X,y)
    pickle.dump(model, open('advertising_model_v3','wb'))

    scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_absolute_error')

    connection.close()
    return "<h3>New model retrained and saved as advertising_model_v1. The results of MAE with cross validation of 10 folds is:<h3> " + str(abs(round(scores.mean(),2)))

app.run()