import tensorflow as tf 
import numpy as np 
import pandas as pd 

# load sample data 
data = pd.read_csv('../datasets/titanic/titanic_dataset.csv')
data = data.dropna()
data.columns = data.columns.str.lower()
data = data.select_dtypes(include=np.number)
# select label data
y = data['survived'].values
# select features
X = data.drop('survived', axis='columns')
feature_names = X.columns.values
X = X.values 
print('features:', X.shape, feature_names)

# tf normalize 
input_tensor = tf.constant(X)

def tf_standardize(x: tf.Tensor):
    """
    Igore missing data. 
        count non-missing data only for average
        Impute to zero and make sum. 
    """
    mask = tf.math.is_nan(input_tensor)
    input_tensor_no_nan = tf.where(
        mask, 
        tf.zeros_like(input_tensor),
        input_tensor)
    count_non_nan = tf.reduce_sum(tf.cast(~mask, tf.float64), axis=0)
    sum_features = tf.reduce_sum(input_tensor_no_nan, axis=0)
    mean_features = tf.math.divide(sum_features, count_non_nan)
    std_features = tf.math.reduce_std(input_tensor_no_nan, axis=0)
    return (input_tensor_no_nan - mean_features) / std_features

# standardize input data
input_tensor_scaled = tf_standardize(input_tensor)
# define output size
output_size = len(np.unique(y))

def logits_layers(inputs, output_size):
    logits = tf.keras.layers.Dense(output_size, name='logits')(inputs)
    return logits

logits = logits_layers(input_tensor_scaled, output_size)
probs = tf.sigmoid(logits)
rounded_probs = tf.math.round(probs)
predictions = tf.cast(rounded_probs, tf.int32)
labels = tf.one_hot(y, depth=2, dtype=tf.int32)
is_correct = tf.math.equal(predictions, labels)
is_correct = tf.cast(is_correct, tf.float32)
accuracy = tf.math.reduce_mean(is_correct, axis=0)
print("accuracy: ", accuracy.numpy())

# cross entropy
labels = tf.cast(labels, dtype=tf.float32)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=labels, logits=logits)


