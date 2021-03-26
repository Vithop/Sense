# Class predictor class :)

# IMPORTS
import numpy as np
import librosa
import os

# PROCESSING
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# MODEL
from keras.models import load_model

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
RELATIVE_MODEL_LOCATION = 'saved_models/new_model_4_class'
MODEL_PATH = os.path.join(CURRENT_PATH, RELATIVE_MODEL_LOCATION)
LABELS = ["car_horn", "construction", "noise", "siren"]


class Model:
    def __init__(self):
        self.model = load_model(MODEL_PATH)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(LABELS)
        self.rows = 40
        self.columns = 174
        self.channels = 1
        self.max = 174

    def extract_features(self, file_name):
        try:
            # Loading audio file
            audio, sr = librosa.load(file_name, res_type='kaiser_fast')
            # Extracting MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

            print(mfccs.shape)
            # If we have to many features, cut them, not needed
            if (mfccs.shape[1] <= self.max):
                pad_width = self.max - mfccs.shape[1]
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:  # Limit the number of features we use to 174
                mfccs = mfccs[:, :174]

        except Exception as e:
            print("Error parsing: ", file_name)
            return None

        return mfccs

    def predict(self, file_name):
        print(file_name)
        features = self.extract_features(file_name)
        features = features.reshape(1, self.rows, self.columns, self.channels)

        predicted_vector = self.model.predict_classes(features)
        predicted_class = self.label_encoder.inverse_transform(predicted_vector)
        print("Predicted class is:", predicted_class[0], '\n')

        # Returning Prediction
        return predicted_class[0]

        # Uncomment below for access to percentages and
        # percentages of  categories not chosen.
        '''
        prob_list = self.model.predict(features)

        # List that contains each probability of each category
        probability = prob_list[0]

        for i in range(len(probability)):
            category = self.label_encoder.inverse_transform(np.array([i]))
            # Prints prob of each category
            print(category[0], "\t\t : ", format(probability[i], '.8f'))
        '''


model = Model()
audio_path = 'audio/dog.wav'
prediction = model.predict(os.path.join(CURRENT_PATH, audio_path))
