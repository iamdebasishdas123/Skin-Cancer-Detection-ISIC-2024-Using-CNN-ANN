## Download the data From the Kaggle database
## Unzip the data
## Load the data into the dataframes
import os
os.environ["KERAS_BACKEND"] = "tensorflow" # other options: tensorflow or torch

import keras_cv
import keras
from keras import ops
import tensorflow as tf
import cv2

import pandas as pd
import numpy as np
from glob import glob
from tqdm.notebook import tqdm
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt 

# Download the data
import os
import shutil
import stat

# Create the .kaggle directory if it doesn't exist
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)

# Copy the kaggle.json file to the .kaggle directory
shutil.copy("kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json"))

# Change the permissions of the kaggle.json file
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), stat.S_IRUSR | stat.S_IWUSR)

def read_data(path):
    return pd.read_csv(path)