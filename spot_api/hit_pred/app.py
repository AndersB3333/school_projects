import pickle
import numpy as np
from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify
import pandas as pd
from textblob import TextBlob

app = Flask(__name__)
CORS(app)

model = pickle.load(open('song_pred.pkl','rb'))

@app.route('/', methods=['POST'])
@cross_origin()

def post():
    data = request.get_json()
    print(data)
    song_df = pd.DataFrame.from_dict(data, orient="index").T.rename(columns={"explicit-input": "explicit"})
    song_df['sentiment'] = TextBlob(str(song_df['song-name-input'].values)[2:-2]).sentiment[0]
    song_df = song_df.iloc[:, 2:]
    prediction = model.predict_proba(song_df)
    response = jsonify(prediction[0, 1])
    return response

if __name__ == '__main__':
    app.run(port=5000, debug=True)