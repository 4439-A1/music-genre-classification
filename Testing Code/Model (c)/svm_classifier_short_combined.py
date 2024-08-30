# import numpy as np
# import joblib
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
# from sklearn.metrics import accuracy_score

# def load_data(features_path, labels_path, num_train_samples):
#     # Load features and original labels
#     features = np.load(features_path)
#     original_labels = np.load(labels_path)
    
#     # Select only the training part of the dataset
#     features_train = features[:num_train_samples]  # Assuming the first 8000 are training samples
#     labels_train = np.repeat(original_labels, 10)  # Assuming each label should be replicated 10 times
    
#     return features_train, labels_train

# def train_svm_classifier(features, labels):
#     scaler = StandardScaler()
#     encoder = LabelEncoder()
    
#     # Encode labels and scale features
#     labels_encoded = encoder.fit_transform(labels)
#     features_scaled = scaler.fit_transform(features)
    
#     # Split the data into training and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(features_scaled, labels_encoded, test_size=0.2, random_state=42)
    
#     param_grid = {
#         'C': [0.001, 0.01, 0.1, 1],  # Lower C values to increase regularization
#         'kernel': ['linear', 'rbf'],  # Trying a simpler linear kernel alongside RBF
#         'gamma': ['scale', 'auto']  # Adjust the gamma parameter, which can also affect overfitting
#     }

#     grid_search = GridSearchCV(SVC(), param_grid, cv=5, verbose=3)
#     grid_search.fit(X_train, y_train)

#     print("Best parameters:", grid_search.best_params_)
#     print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

#     # Predict on the validation set with the best estimator
#     y_pred = grid_search.best_estimator_.predict(X_val)
#     validation_accuracy = accuracy_score(y_val, y_pred)
#     print(f'Validation Accuracy: {validation_accuracy}')

#     # Save the best model, scaler, and encoder
#     joblib.dump(grid_search.best_estimator_, 'Dataset/svm_model_best_combined_normalized.joblib')
#     joblib.dump(scaler, 'Dataset/scaler_best_combined_normalized.joblib')
#     joblib.dump(encoder, 'Dataset/label_encoder_best_combined_normalized.joblib')


# if __name__ == "__main__":
#     num_train_samples = 8000  # Specify the number of training samples
#     features, labels = load_data('Dataset/combined_normalized_new.npy', 'Dataset/labels.npy', num_train_samples)
#     train_svm_classifier(features, labels)


import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data(features_path, labels_path, num_train_samples):
    # Load features and original labels
    features = np.load(features_path)
    original_labels = np.load(labels_path)
    
    # Select only the training part of the dataset
    features_train = features[:num_train_samples]  # Assuming the first 8000 are training samples
    labels_train = np.repeat(original_labels, 10)  # Assuming each label should be replicated 10 times
    
    return features_train, labels_train

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
    joblib.dump(model, 'Dataset/svm_model_best_combined_normalized.joblib')
    joblib.dump(scaler, 'Dataset/scaler_best_combined_normalized.joblib')
    joblib.dump(encoder, 'Dataset/label_encoder_best_combined_normalized.joblib')

if __name__ == "__main__":
    num_train_samples = 8000  # Specify the number of training samples
    features, labels = load_data('Dataset/combined_unnormalized.npy', 'Dataset/labels.npy', num_train_samples)
    train_svm_classifier(features, labels)
