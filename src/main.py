from util import loadDataset, toHotEncoding, accuracy
from logreg import LogisticRegression
import numpy as np

def main():
    loans_train = './data/loans.train'
    loans_valid = './data/loans.val'

    X, T = loadDataset(loans_train)
    T = toHotEncoding(T, 2)
    
    m = LogisticRegression()
    m.fit(X, T)
    
if __name__ == '__main__':
    main()
