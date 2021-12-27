# Wrinkle_force_microscopy
## For the review of Article "Wrinkle force microscopy: a new machine learningbased approach to predict cell mechanics from images".
## A Graphic card with at least 6GB VRAM is recommended.
## Environment:
      with Python 3.5 and CUDA 9.0:
            Tensorflow 1.13.0rc2,
            numpy 1.16.4,
            matplotlib 3.0.2,
            opencv-python 4.1.0.25
## unzip data.zip will acquire training data.
    Please unzip the items to 'data/' not 'data/data'
    The data is divided into 5 parts, set the trigger '--dataset' in train.py and test.py to train or test it.
    In a set of training folder, the train folder is for training, the val folder is for testing. 
    Test data in the folder 'data/w2f_1/val' is well prepaired, if you want to test others, please prepair the data refer to it. 
## The value of the force and the visualization of it can be acquired at the same time by running test.py.
