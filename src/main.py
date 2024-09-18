from util import *
from logreg import LogisticRegression
import numpy as np

loans_train = './data/loans.train'


X, T = loadDataset(loans_train)
T = toHotEncoding(T,2)

m = LogisticRegression()

m.fit(X,T)
Y = m.predict(X)
print(f"First Y: {Y}")
E = m.error(T, Y)
print(f"First E: {E}")
m.GradientDescent(X,T,Y)

Y = m.predict(X)
print(f"Second Y: {Y}")
E = m.error(T, Y)
print(f"Second E: {E}")

# epochs = 10000
# for i in range(epochs):
#     Y = m.predict(X)
#     m.GradientDescent(X, T, Y)

#     if i % 1000 == 0:
#         # print(f"The prediction: {Y}")
#         print(f"accuracy: {accuracy(T, Y)}")
        # print(f"weights: {m.weights}")

# for i in range(epochs):
#     Y = m.predict(X)
    
#     if i % 100 == 0:
#         print(accuracy(T, Y))



