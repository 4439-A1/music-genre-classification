import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

def load_features_labels(feature_path='Dataset/mfcc_features.npy', label_path='Dataset/mfcc_labels.npy'):
    features = np.load(feature_path)
    labels = np.load(label_path)
    return features, labels

def train_svm(features, labels):
    # Split the data
    features_train, features_val, labels_train, labels_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_val_scaled = scaler.transform(features_val)

    # Define extended parameter grid for GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10, 50, 100],
        'gamma': [0.001, 0.01, 0.1, 0.5, 1],
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'degree': [2, 3, 4]  # Only used for 'poly' kernel
    }

    # Create a GridSearchCV object with extended parameter grid
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3, cv=5, scoring='accuracy')
    grid.fit(features_train_scaled, labels_train)

    # Print out the best parameters
    print("The best parameters are: ", grid.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))

    # Use the best estimator to make predictions
    labels_val_pred = grid.predict(features_val_scaled)

    # Print the accuracy with the best parameters
    accuracy = accuracy_score(labels_val, labels_val_pred)
    print(f'Validation Accuracy: {accuracy}')

    return grid.best_estimator_, scaler

def save_model(model, path='Dataset/svm_model_2_mfcc.joblib'):
    joblib.dump(model, path)
    print(f"Model saved to {path}")

if __name__ == '__main__':
    # Load saved features and labels
    features, labels = load_features_labels()

    # Train the SVM classifier with the extended GridSearchCV
    svm_model, scaler = train_svm(features, labels)

    # Save the trained model to a file
    save_model(svm_model)
    save_model(scaler, 'Dataset/scaler_2_mfcc.joblib')
