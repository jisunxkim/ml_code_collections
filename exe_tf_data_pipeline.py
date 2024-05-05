import os 
import subprocess

import tensorflow as tf
from tfx import v1 as tfx

# TFX libraries
import tensorflow_data_validation as tfdv
import tensorflow_transform as tft
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext

# For performing feature selection
from sklearn.feature_selection import SelectKBest, f_classif

# For feature visualization
import matplotlib.pyplot as plt 
import seaborn as sns

# Utilities
from tensorflow.python.lib.io import file_io
from tensorflow_metadata.proto.v0 import schema_pb2
from google.protobuf.json_format import MessageToDict
from  tfx.proto import example_gen_pb2
from tfx.types import standard_artifacts
from tensorflow_transform.tf_metadata import dataset_metadata, schema_utils
import tensorflow_transform.beam as tft_beam
import os
import pprint
import tempfile
import pandas as pd

# To ignore warnings from TF
tf.get_logger().setLevel('ERROR')

# For formatting print statements
pp = pprint.PrettyPrinter()

# Display versions of TF and TFX related packages
print('TensorFlow version: {}'.format(tf.__version__))
print('TFX version: {}'.format(tfx.__version__))
print('TensorFlow Data Validation version: {}'.format(tfdv.__version__))
print('TensorFlow Transform version: {}'.format(tft.__version__))

# # OPTIONAL: Just in case you want to restart the lab workspace *from scratch*, you
# # can uncomment and run this block to delete previously created files and
# # directories. 
# !rm -rf pipeline
# !rm -rf data


# data path
DATA_DIR = '../datasets/covertype/'
TRAINING_DIR = f'{DATA_DIR}/training'
TRAINING_DATA = f'{TRAINING_DIR}/dataset.csv'

# # Create the directory
# # !mkdir -p {TRAINING_DIR}
# os.makedirs(TRAINING_DIR, exist_ok=True)

# # Download data
# command = f'wget -nc https://storage.googleapis.com/mlep-public/course_2/week3/dataset.csv -P {TRAINING_DIR}'

# subprocess.run(command, shell=True)

# Load training data
# Load the dataset to a dataframe
df = pd.read_csv(TRAINING_DATA)

# Preview the dataset
print(df.head())

# Copy original dataset
df_num = df.copy()

# Categorical columns
cat_columns = ['Wilderness_Area', 'Soil_Type']

# Label column
label_column = ['Cover_Type']

# Drop the categorical and label columns
df_num.drop(cat_columns, axis=1, inplace=True)
df_num.drop(label_column, axis=1, inplace=True)

# Preview the resuls
print(df_num.head())

# Set the target values
y = df[label_column].values

# Set the input values
X = df_num.values

# Feature Selection
### START CODE HERE ###

# Create SelectKBest object using f_classif (ANOVA statistics) for 8 classes
select_k_best = SelectKBest(f_classif, k=8)

# Fit and transform the input data using select_k_best
X_new = select_k_best.fit_transform(X, y)

# Extract the features which are selected using get_support API
features_mask = select_k_best.get_support()

### END CODE HERE ###

# Print the results
reqd_cols = pd.DataFrame({'Columns': df_num.columns, 'Retain': features_mask})
print(reqd_cols)

# Set the paths to the reduced dataset
TRAINING_DIR_FSELECT = f'{TRAINING_DIR}/fselect'
TRAINING_DATA_FSELECT = f'{TRAINING_DIR_FSELECT}/dataset.csv'

# Create the directory
os.makedirs(TRAINING_DIR_FSELECT, exist_ok=True)

# Get the feature names from SelectKBest
feature_names = list(df_num.columns[features_mask])

# Append the categorical and label columns
feature_names = feature_names + cat_columns + label_column

# Select the selected subset of columns
df_select = df[feature_names]

# Write CSV to the created directory
df_select.to_csv(TRAINING_DATA_FSELECT, index=False)

# Preview the results
df_select.head()

# Location of the pipeline metadata store
PIPELINE_DIR = './pipeline'

# Declare the InteractiveContext and use a local sqlite file as the metadata store.
context = InteractiveContext(pipeline_root=PIPELINE_DIR)




