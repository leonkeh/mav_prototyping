import cv2
import numpy as np
import time
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
from data_loader import get_labeled_data
import time
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


class DecisionModel:
    def __init__(self):
        self.clf_gd = None
        self.clf_decision = None
        self.dsf = 20
        self.y_test = None
        self.y_pred = None
        self.construct()

    def construct(self):
        with open('svm_V1.pkl', 'rb') as f:
            self.clf_gd = pickle.load(f)

    def feature_extractor(self, X, visualize=False, pred_dict={}):
        timer_start = time.perf_counter()
        X_extracted = []
        for x in X:
            # downsample image
            ds_size = (x.shape[1] // self.dsf, x.shape[0] // self.dsf)
            x_ds = cv2.resize(x, ds_size)

            # binary classification
            x_reshaped = x_ds.copy().reshape((-1, 3))
            x_binary = self.clf_gd.predict(x_reshaped)  # ground detection classifier
            X_extracted.append(x_binary)
            if visualize and pred_dict:
                if pred_dict["label"] != pred_dict["pred"]:
                    x_visual_ds = x_ds
                    x_visual_gd = x_binary.reshape(tuple(reversed(ds_size)))
                    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1)
                    ax0.imshow(np.rot90(x))
                    ax0.set_ylabel("Original")
                    ax0.set_yticklabels([])
                    ax0.set_xticklabels([])
                    ax1.imshow(np.rot90(x_visual_ds))
                    ax1.set_ylabel("Downsampled")
                    ax1.set_yticklabels([])
                    ax1.set_xticklabels([])
                    ax2.imshow(np.rot90(x_visual_gd))
                    ax2.set_ylabel("Ground detection")
                    ax2.set_yticklabels([])
                    ax2.set_xticklabels([])
                    label2verbose = {0: "turning", 1: "go straight"}
                    if pred_dict:
                        if pred_dict["label"] != pred_dict["pred"]:
                            fig.suptitle("Label was " + label2verbose[pred_dict["label"]] + " predicted "+ label2verbose[pred_dict["pred"]], color="red")
                        else:
                            fig.suptitle("Correctly predicted " + label2verbose[pred_dict["label"]], color="green")
                    plt.tight_layout()
                    plt.savefig("output/" + str(time.time()) + ".png", bbox_inches='tight')
                    plt.show()

        timer_end = time.perf_counter()
        timer_result = timer_end - timer_start
        print(f"Extracting the features took {np.round(timer_result, 2)}s. That is {np.round(timer_result/len(X_extracted),2)}s per image.\n")

        return X_extracted

    def train(self, X_train, y_train):
        X_train = self.feature_extractor(X_train)
        self.clf_decision = svm.SVC(kernel='linear')
        timer_start = time.perf_counter()
        self.clf_decision.fit(X_train, y_train)
        timer_end = time.perf_counter()
        print(f"The decision making SVM took {np.round(timer_end - timer_start, 2)}s to fit.")

    def predict(self, X_test, y_test, inspect=False):
        X_test_ = self.feature_extractor(X_test)
        self.y_pred = self.clf_decision.predict(X_test_)
        conf_mat = confusion_matrix(y_test, self.y_pred)
        acc_score = accuracy_score(y_test, self.y_pred)
        print(f"The accuracy score is {acc_score}.")
        print(f"The confusion matrix is \n {conf_mat}")
        if inspect:
            for i in range(len(y_test)):
                pred_dict = {}
                pred_dict["label"] = y_test[i]
                pred_dict["pred"] = self.y_pred[i]
                self.feature_extractor([X_test[i]], visualize=True, pred_dict=pred_dict)



    def run(self, inspect=False, X=None, y=None):
        X, y = get_labeled_data(583, directory="data_labelled_big/labelled_week6", binary=True, balance=True)
        X_train, X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2)
        model.train(X_train, y_train)
        model.predict(X_test, self.y_test, inspect=inspect)

    def run_cross_val(self):
        X, y = get_labeled_data(583, directory="data_labelled_big/labelled_week6", binary=True, balance=True)
        X_extracted = self.feature_extractor(X)
        self.clf_decision = svm.SVC(kernel='linear')
        scores = cross_val_score(self.clf_decision, X_extracted, y, cv=20)
        print("%0.2f accuracy with a standard deviation of %0.2f, at worst %0.2f" % (scores.mean(), scores.std(), np.min(scores)))
        plt.hist(scores)
        plt.xlabel("Accuracy score")
        plt.ylabel("Number of splits")
        plt.title("Cross validation performance")
        plt.show()

    def make_roc(self):
        dsfs = np.arange(5, 50, 2)
        X, y = get_labeled_data(583, directory="data_labelled_big/labelled_week6", binary=True, balance=True)
        X_train, X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2)

        fpr_list = []
        tpr_list = []
        roc_auc_list = []
        for dsf in dsfs:
            self.dsf = dsf
            model.train(X_train, y_train)
            model.predict(X_test, self.y_test)
            conf_mat = confusion_matrix(self.y_test, self.y_pred)
            tpr = conf_mat[0, 0] / np.sum(conf_mat[:, 0])
            fpr = conf_mat[0, 1] / np.sum(conf_mat[:, 1])
            fpr_list.append(fpr)
            tpr_list.append(tpr)

        # Plot the ROC curves
        plt.figure() # figsize=(10, 6)


        plt.plot(fpr_list, tpr_list)
        plt.plot(np.array([0, 1]), np.array([0, 1]), "--")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()
        print(fpr_list)
        print(tpr_list)

    def export(self):
        with open('saved_svms/decision_svm.pkl', 'wb') as f:
            pickle.dump(self.clf_decision, f)
            print("decision_svm parameters saved in .pkl file")
        coefficients = self.clf_decision.coef_
        intercepts = self.clf_decision.intercept_
        print(f"The decision svm has {coefficients.shape, intercepts.shape} parameters.")
        with open("saved_svms/decision_svm_parameters_dsf" + str(self.dsf) + ".txt", "w") as f:
            for coef in coefficients:
                f.write(' '.join(map(str, coef)) + '\n')
            f.write(' '.join(map(str, intercepts)) + '\n')
        print("decision_svm parameters saved in .txt file")

        # save ground detection
        # with open('saved_svms/ground_detection_svm.pkl', 'wb') as f:
        #     pickle.dump(self.clf_gd, f)
        #     print("decision_svm parameters saved in .pkl file")
        # coefficients = self.clf_gd.coef_
        # intercepts = self.clf_gd.intercept_
        # print(f"The ground detection svm has {coefficients.shape, intercepts.shape} parameters.")
        # with open("saved_svms/ground_detection_svm_parameters.txt", "w") as f:
        #     for coef in coefficients:
        #         f.write(' '.join(map(str, coef)) + '\n')
        #     f.write(' '.join(map(str, intercepts)) + '\n')
        # print("decision_svm parameters saved in .txt file")




model = DecisionModel()
#model.run_cross_val()
model.run(inspect=True)
#model.make_roc()
model.export()