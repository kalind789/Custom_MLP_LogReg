from util import loadDataset, toHotEncoding, accuracy
from logreg import LogisticRegression
from mlp import MultilayerPerceptron
import numpy as np

def process_dataset(train_path, valid_path, dataset_name, model_type='mlp'):
    print(f"\nProcessing {dataset_name} dataset with {model_type}...")

    # Load and preprocess training data
    X_train, T_train = loadDataset(train_path)
    T_train = toHotEncoding(T_train, 2)  # Ensure T_train is one-hot encoded

    # Initialize and train the model
    if model_type == 'mlp':
        model = MultilayerPerceptron()
        model.fit(X_train, T_train)
        model.save(f"{dataset_name.lower()}.model")
    elif model_type == 'logreg':
        model = LogisticRegression()
        model.fit(X_train, T_train)

    # Load and preprocess validation data
    X_valid, T_valid = loadDataset(valid_path)
    T_valid = toHotEncoding(T_valid, 2)  # Ensure T_valid is one-hot encoded

    # Make predictions on validation data
    Y_valid = model.predict(X_valid)

    # Calculate accuracy
    accuracy_score = accuracy(T_valid, Y_valid)
    print(f'Accuracy for {dataset_name} dataset using {model_type}: {accuracy_score}')

def main():
    # Paths for the "water" dataset
    water_train = './data/water.train'
    water_valid = './data/water.val'

    # Paths for the "loans" dataset
    loans_train = './data/loans.train'
    loans_valid = './data/loans.val'

    print("Using Logistic Regression:")
    process_dataset(water_train, water_valid, "Water", model_type='logreg')
    process_dataset(loans_train, loans_valid, "Loans", model_type='logreg')

    print("\nUsing Multilayer Perceptron:")
    process_dataset(water_train, water_valid, "Water", model_type='mlp')
    process_dataset(loans_train, loans_valid, "Loans", model_type='mlp')

if __name__ == '__main__':
    main()
