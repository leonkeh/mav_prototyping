import os
import cv2
import time
import numpy as np
import random
from sklearn.utils import resample

def get_labeled_data(n, directory="labeled_data", binary=False, balance=False):
    timer_start = time.perf_counter()
    X = []
    y = []
    lst = os.listdir(directory)
    random.shuffle(lst)
    for img in lst[:n]:
        filename = directory + "/" + img
        x = cv2.imread(filename)
        X.append(x)
        if binary:
            if img[0] in ['l', 'r']:
                y.append(0)
            elif img[0] == 's':
                y.append(1)
        else:
            if img[0] == 'l':
                y.append(0)
            elif img[0] == 's':
                y.append(1)
            elif img[0] == 'r':
                y.append(2)

    if balance:
        n_classes = len(np.bincount(y))
        X = np.array(X)
        y = np.array(y)
        class_counts = np.bincount(y)
        print(f"Original class counts: {class_counts}")
        target_count = np.min(class_counts)
        X_balanced = []
        y_balanced = []
        for class_label in range(len(class_counts)):
            X_class = X[y == class_label]
            X_resampled = resample(X_class, replace=True, n_samples=target_count, random_state=42)
            y_resampled = [class_label for _ in range(target_count)]
            X_balanced.extend([X_resampled[i, :, :, :] for i in range(target_count)])
            y_balanced.extend(y_resampled)

        X, y = X_balanced, y_balanced

    timer_end = time.perf_counter()
    print(f"Loading the data took {np.round(timer_end - timer_start, 2)}s.\n")
    print(f"The classes are now distributed as {np.bincount(y)}")
    return X, y

"""testing"""
#X_, y_ = get_labeled_data(10)
#print(X_[:10], "\n", y_[:10])