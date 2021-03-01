# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import librosa
from flask import Flask, request, jsonify
import os
import random
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


app = Flask(__name__)

# Quick Test 
@app.route("/hello")
def hello():
    return "<h1 style='color:blue'>Hello There!</h1>"

@app.route("/predict", methods=["POST"])
def siren_sense_predict():
    
    max_pad_len = 174

    num_rows = 40
    num_columns = 174
    num_channels = 1

    # Convert into a Panda dataframe 
    # featuresdf = pd.read_csv("server/flask/model/featuresdf.csv")
    featuresdf = pd.read_csv("model/featuresdf.csv")

    y = np.array(featuresdf.class_label.tolist())
    # print(y)
    
    # Encode the classification labels
    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y))


    # model = tf.keras.models.load_model('server/flask/model/siren_sense_model.hdf5')
    model = tf.keras.models.load_model('model/siren_sense_model.hdf5')
    # model.summary()

    audio_file = request.files["file"]
    file_name = str(random.randint(0,1000000))
    audio_file.save(file_name)

    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    prediction_feature = mfccs
    prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)

   
    predicted_vector = model.predict_classes(prediction_feature)
    predicted_classes = le.inverse_transform(predicted_vector) 
    # print("The predicted class is:", predicted_classes[0], '\n') 

    predicted_class = predicted_classes[0]

    data = {"predicted_class": predicted_class}

    os.remove(file_name)
    return jsonify(data)
    

if __name__ == "__main__":
    # app.run(debug=False)
    app.run(host='0.0.0.0')
# %%
