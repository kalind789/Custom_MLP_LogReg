from util import loadDataset, toHotEncoding, accuracy
from logreg import LogisticRegression
import numpy as np

def process_dataset(train_path, valid_path, dataset_name):
    print(f"\nProcessing {dataset_name} dataset...")
    
    # Load training data
    X, T = loadDataset(train_path)
    T = toHotEncoding(T, 2)
        
    # Train logistic regression model
    m = LogisticRegression()
    m.fit(X, T)

    # Load validation data
    X2, T2 = loadDataset(valid_path)
    T2 = toHotEncoding(T2, 2)
    
    # Predict on validation data (return class labels, not one-hot)
    Y2 = m.predict(X2)
    
    # Calculate accuracy (T2 should be class labels, so convert it)
    accuracy_score = accuracy(np.argmax(T2, axis=1), Y2)
    print(f'Accuracy for {dataset_name} dataset: {accuracy_score}')

def main():
    # Paths for the "water" dataset
    water_train = './data/water.train'
    water_valid = './data/water.val'

    # Paths for the "loans" dataset
    loans_train = './data/loans.train'
    loans_valid = './data/loans.val'

    # Process both datasets
    process_dataset(water_train, water_valid, "Water")
    process_dataset(loans_train, loans_valid, "Loans")
    
if __name__ == '__main__':
    main()
