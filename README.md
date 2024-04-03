# mav_prototyping
This repository contains the Support Vector Machine (SVM) approach prototypes for both ground detection and deciding on where to go.

## Ground classification
Please run the the floor_detection.py file. The features will be extracted, the SVM fitted, and some example filtered images will be displayed. The SVM parameters are saved to "svm_V1.pkl".

## Decision classification
Please run decision_svm.py. The decision SVM will be trained and some examples from the test set will be displayed. Note, that you can change some booleans:
- "binary", if True, the SVM will only predict between go straight and turn. If False, the SVM will do multiclass classification between left, straight, and right. "balance" ensures a balanced dataset by down sampling to the minority class count. Finally, you can run the function run_cross_val, to obtain a histogram with the cross validation scores.
