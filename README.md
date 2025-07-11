# FingerSpell-An-ASL-Recognition-System

# Directory

│  app.py
│  keypoint_classification.ipynb

│  point_history_classification.ipynb

│  
├─model
│  ├─keypoint_classifier
│  │  │  keypoint.csv
│  │  │  keypoint_classifier.hdf5
│  │  │  keypoint_classifier.py
│  │  │  keypoint_classifier.tflite
│  │  └─ keypoint_classifier_label.csv
│  │          
│  └─point_history_classifier
│      │  point_history.csv
│      │  point_history_classifier.hdf5
│      │  point_history_classifier.py
│      │  point_history_classifier.tflite
│      └─ point_history_classifier_label.csv
│          
└─utils
    └─cvfpscalc.py
## app.py

Collect training data (hand coordinate history) for gesture recognition.

## keypoint_classification.ipynb

This is a model training script for hand sign recognition.

## model/keypoint_classifier

This directory stores files related to hand sign recognition.
The following files are stored.

1.Training data(keypoint.csv)
2.Trained model(keypoint_classifier.tflite)
3.Label data(keypoint_classifier_label.csv)
4.Inference module(keypoint_classifier.py)
