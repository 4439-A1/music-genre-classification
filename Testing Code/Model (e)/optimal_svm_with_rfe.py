import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import RFE
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
    
    # Determine if you should use the dual formulation
    use_dual = X_train.shape[0] > X_train.shape[1]  # True if more samples than features

    # Initialize the LinearSVC model for RFE
    linear_svc = LinearSVC(dual=use_dual, max_iter=20000, tol=1e-4)
    
    # Feature selection with RFE using LinearSVC
    rfe = RFE(estimator=linear_svc, n_features_to_select=20, step=1, verbose=3)
    rfe.fit(X_train, y_train)
    
    # Transform features using RFE
    X_train_rfe = rfe.transform(X_train)
    X_val_rfe = rfe.transform(X_val)
    
    # Train the final model using the RBF kernel with selected features
    model = SVC(kernel='rbf', C=1, decision_function_shape='ovo', gamma='scale')
    model.fit(X_train_rfe, y_train)
    
    # Predict on the validation set with the reduced feature set
    y_pred = model.predict(X_val_rfe)
    validation_accuracy = accuracy_score(y_val, y_pred)
    print(f'Validation Accuracy: {validation_accuracy}')

    # Save the best model, scaler, encoder, and RFE selector
    joblib.dump(model, 'Dataset/svm_model_optimal_rfe.joblib')
    joblib.dump(scaler, 'Dataset/scaler_optimal_rfe.joblib')
    joblib.dump(encoder, 'Dataset/label_encoder_optimal_rfe.joblib')
    joblib.dump(rfe, 'Dataset/rfe_selector_optimal.joblib')

if __name__ == "__main__":
    features, labels = load_data('Dataset/Processed(4)/train_features.npy', 'Dataset/Processed(4)/train_labels.npy')
    train_svm_classifier(features, labels)
