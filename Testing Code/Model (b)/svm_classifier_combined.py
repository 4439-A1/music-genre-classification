import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

def load_data(features_path, labels_path):
    features = np.load(features_path)
    labels = np.load(labels_path)
    return features, labels

def train_svm_classifier(features, labels):
    scaler = StandardScaler()
    encoder = LabelEncoder()
    
    labels_encoded = encoder.fit_transform(labels)
    features_scaled = scaler.fit_transform(features)
    
    X_train, X_val, y_train, y_val = train_test_split(features_scaled, labels_encoded, test_size=0.2, random_state=42)
    
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],  # Increasing penalty for misclassifying a data point
        'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],  # Kernel coefficient for 'rbf', 'poly', 'sigmoid'
        'kernel': ['rbf', 'poly', 'sigmoid', 'linear']  # Different types of kernels
    }
    
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3, cv=5)
    grid.fit(X_train, y_train)
    
    print("The best parameters are: ", grid.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))

    # Predicting on the validation set with the best estimator
    y_pred = grid.best_estimator_.predict(X_val)
    validation_accuracy = accuracy_score(y_val, y_pred)
    print(f'Validation Accuracy: {validation_accuracy}')

    # Saving the best model, scaler, and encoder
    joblib.dump(grid.best_estimator_, 'Dataset/svm_model_best_combined.joblib')
    joblib.dump(scaler, 'Dataset/scaler_best_combined.joblib')
    joblib.dump(encoder, 'Dataset/label_encoder_best_combined.joblib')

if __name__ == "__main__":
    features, labels = load_data('Dataset/combined_features.npy', 'Dataset/labels.npy')
    train_svm_classifier(features, labels)
    # Tried out robust, minmax scaling but found standard scaling is the best
    # Tried using PCA but accuracy was 0.6625 with PCA and 0.69375 without