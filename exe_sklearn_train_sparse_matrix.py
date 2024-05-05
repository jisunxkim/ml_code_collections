from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix
import time 

# Load the 20 newsgroups dataset
data = fetch_20newsgroups(subset='all', categories=['sci.space', 'rec.autos'], remove=('headers', 'footers', 'quotes'))

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Convert text data to a bag-of-words representation using CountVectorizer
vectorizer = CountVectorizer(max_features=1000)
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

### Training with regular data
print("training with regular dataframe")
start_time = time.time()
model = LogisticRegression(max_iter=1000)
model.fit(X_train_counts, y_train)
predictions = model.predict(X_test_counts)
accuracy = (predictions == y_test).mean()
print("Accuracy:", accuracy)
print(f"total time: {round(time.time() - start_time, 2)} secs")


### Converting to sparse matrix and training  with it. 
print("training with sparse matrix")
start_time = time.time()
# Convert the bag-of-words representation to a csr_matrix
X_train_sparse = csr_matrix(X_train_counts)
X_test_sparse = csr_matrix(X_test_counts)
print("converted to sparse matrix")
# Train a Logistic Regression model using the csr_matrix
model = LogisticRegression(max_iter=1000)
model.fit(X_train_sparse, y_train)

# Make predictions on test data
predictions = model.predict(X_test_sparse)

# Evaluate the model
accuracy = (predictions == y_test).mean()
print("Accuracy:", accuracy)
print(f"total time: {round(time.time() - start_time, 2)} secs")