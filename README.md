# mav_prototyping
please run the the decision_svm.py file

Note the setting of binary True/False in the data_loader. Binary means we only classify between go straight and turn, this seems to work much better than classifying between left/straight/right and we can use the stupid policy to always turn in one direction. Otherwise, the multiclass classification might improve a lot by gathering more data.
