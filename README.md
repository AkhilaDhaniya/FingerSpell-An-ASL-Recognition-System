# FingerSpell-An-ASL-Recognition-System

# Directory

│  app.py

│  keypoint\_classification.ipynb

│  point\_history\_classification.ipynb

│  

├─model

│  ├─keypoint\_classifier

│  │  │  keypoint.csv

│  │  │  keypoint\_classifier.hdf5

│  │  │  keypoint\_classifier.py

│  │  │  keypoint\_classifier.tflite

│  │  └─ keypoint\_classifier\_label.csv

│  │          

│  └─point\_history\_classifier

│      │  point\_history.csv

│      │  point\_history\_classifier.hdf5

│      │  point\_history\_classifier.py

│      │  point\_history\_classifier.tflite

│      └─ point\_history\_classifier\_label.csv

│          

└─utils

&nbsp;   └─cvfpscalc.py

## app.py

Collect training data (hand coordinate history) for gesture recognition.

## keypoint\_classification.ipynb

This is a model training script for hand sign recognition.

## model/keypoint\_classifier

This directory stores files related to hand sign recognition.
The following files are stored.

1.Training data(keypoint.csv)
2.Trained model(keypoint\_classifier.tflite)
3.Label data(keypoint\_classifier\_label.csv)
4.Inference module(keypoint\_classifier.py)

