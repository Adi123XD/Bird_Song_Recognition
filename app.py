import streamlit as st
import librosa
import pickle as pickle
import numpy as np
import tempfile
import os 
from main import BirdSongPredictor,AudioFeatureExtractor

audio_data_filepath ='data\wavfiles'
# Dictionary mapping original species names to Indian bird names
species_mapping = {
    'bewickii': 'Indian Peafowl (Pavo cristatus)',
    'polyglottos': 'Indian Robin (Saxicoloides fulicatus)',
    'migratorius': 'Common Kingfisher (Alcedo atthis)',
    'melodia': 'Indian Pond Heron (Ardeola grayii)',
    'cardinalis': 'Asian Koel (Eudynamys scolopaceus)'
}


def main():
    # Title and description
    st.title("Bird Song Classification")
    st.write("This application predicts the class of a bird song from a uploaded audio file.")

    # Upload file
    uploaded_file = st.file_uploader("Choose an audio file (WAV format)", type="wav")

    # Load prediction model and label encoder (if available)
    predictor = BirdSongPredictor()

    if uploaded_file is not None:
        file_name = uploaded_file.name
        audio_path = os.path.join(audio_data_filepath, file_name)

        # Extract features
        audio_feature_extractor = AudioFeatureExtractor()
        prediction_feature = audio_feature_extractor.extract_features(audio_path)
        prediction_feature = prediction_feature.reshape(1, -1)

        # Make prediction
        predicted_label = predictor.model.predict(prediction_feature)
        predicted_class_index = np.argmax(predicted_label, axis=1)

        # Display results
        st.write("**Uploaded File:**", uploaded_file.name)
        # st.write("**Predicted Class:**", predicted_class_index[0])

        # Display class label if encoder is loaded
        if hasattr(predictor, 'labelencoder'):
            prediction_class = predictor.labelencoder.inverse_transform(predicted_class_index)
            if prediction_class[0] in species_mapping:
                indian_bird_name = species_mapping[prediction_class[0]]
                st.subheader(f'This is the sound of {indian_bird_name} bird') 
            else:
                # st.write("Bird species not found in mapping.")
                st.subheader(f'This is the sound of {prediction_class[0]} bird') 
        else:
            st.write("Label encoder not found. Class labels might not be available.")


if __name__ == "__main__":
    main()
