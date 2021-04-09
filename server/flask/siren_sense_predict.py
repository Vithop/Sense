# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# import tensorflow as tf
import soundfile as sf
import pandas as pd
import numpy as np
import librosa
import pprint
import random
import sys
import os
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
# from pydub import AudioSegment
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.utils import to_categorical
# sys.path.append(".../model")
# from ...model.predictor import Model
from predictor import Model
app = Flask(__name__)
CORS(app)

# class Sense:
#     max_pad_len = 174

#     num_rows = 40
#     num_columns = 174
#     num_channels = 1
    
#     def __init__(self):

#         # Convert into a Panda dataframe 
#         self.featuresdf = pd.read_csv("server/flask/model/featuresdf.csv")
#         # self.featuresdf = pd.read_csv("model/featuresdf.csv")

#         y = np.array(self.featuresdf.class_label.tolist())
#         # print(y)
        
#         # Encode the classification labels
#         self.le = LabelEncoder()
#         to_categorical(self.le.fit_transform(y))

#         # self.model = tf.keras.models.load_model('server/flask/model/siren_sense_model.hdf5')
#         # self.model = tf.keras.models.load_model('model/siren_sense_model.hdf5')
#         self.model = tf.keras.models.load_model('model/saved_models/new_model_4_class')
#         # self.sess = tf.compat.v1.keras.backend.get_session()
#         # self.graph = tf.Graph
#     def predict(self, file_name):
#         try:
#             audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
#             # audio, sample_rate = sf.read(file_name.stream()) 
#             mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
#             pad_width = self.max_pad_len - mfccs.shape[1]
#             mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
#         except Exception as e:
#             print("Error encountered while parsing file: ", file_name)
#             return None 
        
#         prediction_feature = mfccs.reshape(1, self.num_rows, self.num_columns, self.num_channels)


#         predicted_vector = np.argmax(self.model.predict(prediction_feature), axis=-1)
#         predicted_classes = self.le.inverse_transform(predicted_vector) 
#         # print("The predicted class is:", predicted_classes[0], '\n') 

#         return predicted_classes[0]



sense = Model()

# Quick Test 
@app.route("/hello")
def hello():
    # data = "<h1 style='color:blue'>Hello There!</h1>"
    # return jsonify(data)
    return "<h1 style='color:blue'>Hello There!</h1>"

@app.route("/predict_wav", methods=["POST"])
def siren_sense_predict_wav():
    
    

    audio_file = request.files["file"]
    file_name = str(random.randint(0,1000000))+".wave"
    audio_file.save(file_name)

    predicted_class = sense.predict(file_name)

    data = {"predicted_class": predicted_class}

    os.remove(file_name)
    return jsonify(data)

@app.route("/predict_mp4", methods=["POST"])
def siren_sense_predict_mp4():
        
    audio_file = request.files["file"]
    file_name = str(random.randint(0,1000000))+".mp4"
    audio_file.save(file_name)
    print("saved file")
    # audio, sample_rate = sf.read(audio_file.stream()) 
    data = {"predicted_class": sense.predict(file_name)}

    os.remove(file_name)
    return jsonify(data)
    

if __name__ == "__main__":
    # app.run(debug=False)
    app.run(host='0.0.0.0', debug=True)
# %%
