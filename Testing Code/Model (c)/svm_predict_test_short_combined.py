import numpy as np
import pandas as pd
import joblib
from scipy.stats import mode

def load_resources(model_path, scaler_path, encoder_path):
    """Load the trained SVM model, scaler, and label encoder."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    return model, scaler, encoder

def predict_genres(features, model, scaler, encoder):
    """Predict the genres of the audio files using preloaded features."""
    features_scaled = scaler.transform(features)
    predicted_labels = model.predict(features_scaled)
    predicted_genres = encoder.inverse_transform(predicted_labels)
    return predicted_genres

def load_test_data(features_path):
    """Load the test data features from the specified part of the dataset."""
    features = np.load(features_path)
    return features

def majority_voting(predictions, num_clips=10):
    """Aggregate predictions by majority voting for each file."""
    file_predictions = []
    for i in range(0, len(predictions), num_clips):
        # Find the most common genre in the 10 clips of each file
        genres, counts = np.unique(predictions[i:i+num_clips], return_counts=True)
        file_genre = genres[np.argmax(counts)]
        file_predictions.append(file_genre)
    return file_predictions

if __name__ == "__main__":
    model_path = 'Dataset/svm_model_best_combined_normalized.joblib'
    scaler_path = 'Dataset/scaler_best_combined_normalized.joblib'
    encoder_path = 'Dataset/label_encoder_best_combined_normalized.joblib'
    features_path = 'Dataset/combined_testclips.npy'
    output_file = 'Dataset/svm_submission_short_combined.csv'

    model, scaler, encoder = load_resources(model_path, scaler_path, encoder_path)
    test_features = load_test_data(features_path)
    predicted_genres = predict_genres(test_features, model, scaler, encoder)

    # Get the majority vote genre for each file
    final_predictions = majority_voting(predicted_genres, num_clips=10)
    
    # Generate IDs for each original test file (200 test files)
    test_file_ids = [f"test{i:03d}.wav" for i in range(len(final_predictions))]

    # Save the final aggregated predictions to a CSV file
    df = pd.DataFrame(list(zip(test_file_ids, final_predictions)), columns=['ID', 'Genre'])
    df.to_csv(output_file, index=False)
    print(f"Majority vote predictions saved to {output_file}")
