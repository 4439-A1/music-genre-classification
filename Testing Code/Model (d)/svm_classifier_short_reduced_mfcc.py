import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data(features_path, labels_path):
    # Load features and original labels
    features = np.load(features_path)
    original_labels = np.load(labels_path)
    
    return features, original_labels

def train_svm_classifier(features, labels):
    scaler = StandardScaler()
    encoder = LabelEncoder()
    
    # Encode labels and scale features
    labels_encoded = encoder.fit_transform(labels)
    features_scaled = scaler.fit_transform(features)
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features_scaled, labels_encoded, test_size=0.2, random_state=42)
    
    # Set SVM with specified parameters
    model = SVC(C=1, kernel='rbf', gamma='scale')
    model.fit(X_train, y_train)
    
    # Predict on the validation set
    y_pred = model.predict(X_val)
    validation_accuracy = accuracy_score(y_val, y_pred)
    print(f'Validation Accuracy: {validation_accuracy}')

    # Save the best model, scaler, and encoder
    joblib.dump(model, 'Dataset/svm_model_best_reduced_mfcc.joblib')
    joblib.dump(scaler, 'Dataset/scaler_best_reduced_mfcc.joblib')
    joblib.dump(encoder, 'Dataset/label_encoder_best_reduced_mfcc.joblib')

if __name__ == "__main__":
    features, labels = load_data('Dataset/Processed(3)/train_features.npy', 'Dataset/Processed(3)/train_labels.npy')
    train_svm_classifier(features, labels)
