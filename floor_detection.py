import os

from PIL import Image
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from os import listdir
import pickle

image_1 = "data/184020281.jpg"
image = Image.open('data/184020281.jpg')


def sample_data(image=image_1):
    # define ref area
    image = cv2.imread(image)
    x_min, x_max, y_min, y_max = 200, 230, 200, 300
    # plot the reference area in the image
    image_info = image.copy()
    red = np.array([255., 0., 0.])
    image_info[x_min:x_max, y_max, :] = red
    image_info[x_min:x_max, y_min, :] = red
    image_info[x_min, y_min: y_max, :] = red
    image_info[x_max, y_min:y_max, :] = red


    plt.imshow(image_info)
    #plt.show()

    data = {}

    rp = image[x_min:x_max, y_min:y_max, :]  # reference pixels
    rp_Y = rp[:, :, 0].flatten()
    rp_U = rp[:, :, 1].flatten()
    rp_V = rp[:, :, 2].flatten()
    data["rp"] = rp


    # some obstacle pixels
    # a pillar
    x_min, x_max, y_min, y_max = 50, 150, 100, 125
    op1 = image[x_min:x_max, y_min:y_max, :]
    op1_Y = op1[:, :, 0].flatten()
    op1_U = op1[:, :, 1].flatten()
    op1_V = op1[:, :, 2].flatten()
    data["o_pillar"] = op1
    # the wall
    x_min, x_max, y_min, y_max = 0, 150, 200, 400
    op2 = image[x_min:x_max, y_min:y_max, :]
    op2_Y = op2[:, :, 0].flatten()
    op2_U = op2[:, :, 1].flatten()
    op2_V = op2[:, :, 2].flatten()
    data["o_wall"] = op2


    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Y')
    ax.set_ylabel('U')
    ax.set_zlabel('V')
    ax.scatter(op1_U, op1_V, op1_Y, color='red', label="pillar obstacle")
    ax.scatter(op2_U, op2_V, op2_Y, color='black', label="wall")

    ax.scatter(rp_U, rp_V, rp_Y, color='green', label="reference area")
    ax.legend()
    #plt.show()
    return data

# get data
data = sample_data(image=image_1)
X_rp = data["rp"].reshape((-1, 3))
X_obs1 = data["o_pillar"].reshape((-1, 3))
X_obs2 = data["o_wall"].reshape((-1, 3))
X_obs = np.vstack([X_obs1, X_obs2])
y_rp = np.ones(X_rp.shape[0]).reshape((-1, 1))
y_obs = np.zeros(X_obs.shape[0]).reshape((-1, 1))

X = np.vstack([X_rp, X_obs])
y = np.vstack([y_rp, y_obs]).flatten()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = svm.SVC()
clf.fit(X_train, y_train)
print("The SVM was trained. \n")
y_pred = clf.predict(X_test)
print(f"The accuracy score of the SVM on the test data is {np.round(accuracy_score(y_test, y_pred),3)}")

SAVE_CLF = False
if SAVE_CLF:
    with open('svm_V1.pkl', 'wb') as f:
        pickle.dump(clf, f)
        print("SVM parameters saved successfully")


def visualize_classifier(clf, image):
    image = cv2.imread(image)
    plt.figure()
    green = np.array([0, 255, 0])
    black = np.array([0, 0, 0])
    image_filtered = image.copy()
    pixels = image.copy().reshape((-1, 3))
    preds = clf.predict(pixels).reshape(image.shape[:2])
    image_filtered[preds == 1] = green

    # filter out small green clusters
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(preds.astype(np.uint8), 4)
    min_size = 1500  # hyperparameter!
    for i in range(1, num_labels):  # Skip the first label, as it's the background
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            labels[labels == i] = 0
    image_filtered2 = image.copy()
    large_green_areas = (labels > 0).astype(np.uint8)
    image_filtered2[large_green_areas == 1] = green
    # If you want to visualize the result
    return image, image_filtered, image_filtered2


directory = "data3"
for img in os.listdir(directory):
    filename = directory + "/" + img
    image, image_filtered, image_filtered2 = visualize_classifier(clf, filename)
    plt.title(img + " - original image file")
    plt.imshow(image)
    plt.show()
    plt.title("Classified to obstacle pixel True False via SVM")
    plt.imshow(image_filtered)
    plt.show()
    plt.title("Small clusters removed.")
    plt.imshow(image_filtered2)
    plt.show()

