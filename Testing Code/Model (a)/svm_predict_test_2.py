import os
import numpy as np
import pandas as pd
import joblib
import librosa
from feature_extraction import extract_features

def load_model_scaler(model_path='Dataset/svm_model_2.joblib', scaler_path='Dataset/scaler_2.joblib'):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict_genres(model, scaler, test_folder='Dataset/test/'):
    file_ids = sorted([f for f in os.listdir(test_folder) if f.endswith('.wav')])
    features_test = []

    # Extract features from test data
    for file_id in file_ids:
        file_path = os.path.join(test_folder, file_id)
        feature = extract_features(file_path)
        features_test.append(feature)

    # Scale the features
    features_test_scaled = scaler.transform(np.array(features_test))

    # Predict using the SVM model
    predicted_labels = model.predict(features_test_scaled)
    
    return file_ids, predicted_labels

def save_predictions(file_ids, predictions, encoder, submission_path='Dataset/svm_submission_2.csv'):
    # Inverse transform the encoded labels to genres
    predicted_genres = encoder.inverse_transform(predictions)

    # Create a DataFrame with IDs and predicted genres
    submission = pd.DataFrame({
        'ID': file_ids,  # Keep the '.wav' extension as required
        'Genre': predicted_genres
    })

    # Save the predictions to a new CSV file
    submission.to_csv(submission_path, index=False)
    print(f"Predictions saved to '{submission_path}'")

if __name__ == '__main__':
    # Load the trained SVM model and scaler
    svm_model, scaler = load_model_scaler()

    # Load the label encoder
    encoder = joblib.load('Dataset/label_encoder.joblib')

    # Predict genres on the test data
    file_ids, predictions = predict_genres(svm_model, scaler)

    # Save the predictions to a CSV file
    save_predictions(file_ids, predictions, encoder)
