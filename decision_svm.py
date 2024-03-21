import cv2
import numpy as np
import time
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
from data_loader import get_labeled_data
import time

with open('svm_V1.pkl', 'rb') as f:
    clf_gd = pickle.load(f)

def feature_extractor(X):
    timer_start = time.perf_counter()
    X_extracted = []
    for x in X:
        # downsample image
        dsf = 10  # down sampling factor
        x_ds = cv2.resize(x, (x.shape[1] // dsf, x.shape[0] // dsf))

        # binary classification
        x_reshaped = x_ds.copy().reshape((-1, 3))
        x_binary = clf_gd.predict(x_reshaped)  # ground detection classifier
        X_extracted.append(x_binary)

    timer_end = time.perf_counter()
    timer_result = timer_end - timer_start
    print(f"Extracting the features took {np.round(timer_result, 2)}s. That is {np.round(timer_result/len(X_extracted),2)}s per image.\n")
    return X_extracted


X, y = get_labeled_data(100, binary=True, balance=True)
X = feature_extractor(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf_decision = svm.SVC()
timer_start = time.perf_counter()
clf_decision.fit(X_train, y_train)
timer_end = time.perf_counter()
print(f"The decision making SVM took {np.round(timer_end - timer_start, 2)}s to fit.")

y_pred = clf_decision.predict(X_test)
print(f"The accuracy score is {accuracy_score(y_test, y_pred)}.")
print(f"The confusion matrix is \n {confusion_matrix(y_test, y_pred)}")
