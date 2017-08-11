import sys
import os
import shutil
import time
import traceback

from flask import Flask, request, jsonify
import pandas as pd
from sklearn.externals import joblib

import inference

app = Flask(__name__)


model_filename = os.path.join('models', 'bag-of-words-all-classes.pkl')
labels_filename = os.path.join('models', 'temas.csv')
labels = None
clf = None

@app.route('/predict', methods=['POST'])
def predict():
    
    if clf:
        try:
            #json_ = request.json
            #query = pd.get_dummies(pd.DataFrame(json_))
            query = request.get_json()

            # https://github.com/amirziai/sklearnflask/issues/3
            # Thanks to @lorenzori
            #query = query.reindex(columns=model_columns, fill_value=0)
            print(query)

            prediction = list(inference.predict_labels(clf, labels, query.text))

            return jsonify({'prediction': prediction})

        except Exception as e:
            return jsonify({'error': str(e), 'trace': traceback.format_exc()})

    else:
        print('train first')
        return 'no model here'



try:
    port = int(sys.argv[1])
except:
    port = 8080

try:
    clf = joblib.load(model_filename)
    print('model loaded')
    labels = pd.read_csv(labels_filename)
    print('model labels loaded')

except Exception as e:
    print('No model here')
    print('Train first')
    print(str(e))
    clf = None
