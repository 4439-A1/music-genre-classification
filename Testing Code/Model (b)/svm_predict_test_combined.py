import os
import numpy as np
import pandas as pd
import joblib
from feature_extraction_combined import combine_features

def load_resources(model_path, scaler_path, encoder_path):
    """Load the trained model, scaler, and label encoder."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    return model, scaler, encoder

def predict_genres(test_folder, model, scaler, encoder):
    """Predict the genres of the audio files in the test folder."""
    file_ids = [f for f in os.listdir(test_folder) if f.endswith('.wav')]
    file_ids.sort()
    predictions = []

    for file_id in file_ids:
        file_path = os.path.join(test_folder, file_id)
        features = combine_features(file_path)
        features_scaled = scaler.transform([features])  # Scale the extracted features
        predicted_label = model.predict(features_scaled)
        predicted_genre = encoder.inverse_transform(predicted_label)[0]
        predictions.append((file_id, predicted_genre))

    return predictions

def save_predictions(predictions, output_file):
    """Save the predictions to a CSV file."""
    df = pd.DataFrame(predictions, columns=['ID', 'Genre'])
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    model_path = 'Dataset/svm_model_best_combined.joblib'
    scaler_path = 'Dataset/scaler_best_combined.joblib'
    encoder_path = 'Dataset/label_encoder_best_combined.joblib'

    test_folder = 'Dataset/test' 
    output_file = 'Dataset/svm_submission_combined.csv'

    model, scaler, encoder = load_resources(model_path, scaler_path, encoder_path)
    predictions = predict_genres(test_folder, model, scaler, encoder)
    save_predictions(predictions, output_file)
