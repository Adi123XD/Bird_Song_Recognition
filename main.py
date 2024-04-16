# load the libraries
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle as pickle 
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder



audio_dataset_path='data\wavfiles'
metadata_path =r'data\bird_songs_metadata.csv'



class AudioFeatureExtractor:
    def __init__(self, n_mfcc=40):
        self.n_mfcc = n_mfcc  # Number of Mel-frequency cepstral coefficients

    def extract_features(self, file_name):
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=self.n_mfcc)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        return mfccs_scaled_features
class BirdSongData:
    def __init__(self):
        self.audio_dataset_path = audio_dataset_path
        self.metadata = pd.read_csv(metadata_path)
        self.labelencoder = LabelEncoder()

    def prepare_data(self):
        extracted_features = []
        for index_num, row in tqdm(self.metadata.iterrows()):
            file_name = os.path.join(os.path.abspath(self.audio_dataset_path), str(row["filename"]))
            final_class_labels = row["species"]
            data = AudioFeatureExtractor().extract_features(file_name)
            extracted_features.append([data, final_class_labels])

        extracted_features_df = pd.DataFrame(extracted_features, columns=['feature', 'class'])
        X = np.array(extracted_features_df['feature'].tolist())
        y = np.array(extracted_features_df['class'].tolist())
        y = to_categorical(self.labelencoder.fit_transform(y))
        # Save data to a pickle file (optional)
        with open('features.pickle', 'wb') as f:
            pickle.dump((X, y, self.labelencoder), f)
        return X, y

    # Getter methods (optional)
    def get_labelencoder(self):
        return self.labelencoder


class BirdSongPredictor:
    def __init__(self):
        self.model = load_model(r'saved_models\audio_classification.hdf5')
        self.audio_dataset_path = 'data\wavfiles'
        self.metadata = pd.read_csv(r'data\bird_songs_metadata.csv')
        self.audio_feature_extractor = AudioFeatureExtractor()
        try:
            # Load label encoder from pickle (optional)
            with open('features.pickle', 'rb') as f:
                _, _, self.labelencoder = pickle.load(f)
                print("Loaded label encoder from pickle file.")
        except FileNotFoundError:
            print("Pickle file not found. Prediction might not be accurate.")
            # Optionally: Create a new label encoder (if needed)
            # self.labelencoder = LabelEncoder()  # Uncomment if needed

    def predict(self, filename):
        prediction_feature = self.audio_feature_extractor.extract_features(filename)
        prediction_feature = prediction_feature.reshape(1, -1)
        predicted_label = self.model.predict(prediction_feature)
        predicted_class_index = np.argmax(predicted_label, axis=1)

        # Use loaded label encoder (if available)
        if hasattr(self, 'labelencoder'):
            prediction_class = self.labelencoder.inverse_transform(predicted_class_index)
        else:
            prediction_class = predicted_class_index  # Return class indices if no encoder

        return prediction_class

# Example usage
# predictor = BirdSongPredictor()
# filename = r"data\wavfiles\101308-2.wav"
# predicted_class = predictor.predict(filename)
# print("Predicted class:", predicted_class)
