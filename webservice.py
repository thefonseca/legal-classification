import sys
import traceback

from flask import Flask, request, jsonify
import pandas as pd
from sklearn.externals import joblib

import inference


app = Flask(__name__)


def load_model():
    model_filename, labels_filename = inference.download_model()
    print('Loading model...')
    model_ = joblib.load(model_filename)
    labels_ = pd.read_csv(labels_filename)
    print('Modelo treinado para {} temas'.format(labels_.shape[0]))
    return model_, labels_


labels = None
model = None


@app.route('/temas', methods=['POST'])
def predict():
    try:
        query = request.get_json()
        prediction = inference.predict_labels(model, labels, query['text'])
        result = list(zip(prediction['LABEL'], prediction['PROBABILITY']))
        #print('\n', result)
        return jsonify({'prediction': result}), 200

    except Exception as ex:
        return jsonify({'error': str(ex), 'trace': traceback.format_exc()})


@app.route('/temas/url', methods=['POST'])
def predict_url():
    try:
        query = request.get_json()
        prediction, text = inference.predict_labels_url(model, labels, query['url'])
        result = list(zip(prediction['LABEL'], prediction['PROBABILITY']))
        return jsonify({'text': text, 'prediction': result}), 200

    except Exception as ex:
        return jsonify({'error': str(ex), 'trace': traceback.format_exc()})


try:
    port = int(sys.argv[1])
except:
    port = 8080

try:
    model, labels = load_model()

except Exception as ex:
    print('Could not load model.')
    print(str(ex))
    model = None
