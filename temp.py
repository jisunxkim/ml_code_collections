import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Custom function for text cleaning
def text_remove_numbers(text_series):
    """
    Return text after replacing ['1970', '12345,', '20', '150', '$1,000',
    '2.01', '270,000', '$800,000', '1980', '20%'] with a ''
    => remove them from the text 
    """
    num_pattern = r"(?:\$?\d+\.?\,?\d*%?)"
    res_series = text_series.apply(lambda x: re.sub(num_pattern, '', x))
    return res_series

# DataFrame with Titanic passenger data
data = pd.DataFrame({
    'Name': ["Kelly, Mr. James",
             "Wilkes, Mrs. James (Ellen Needs)",
             "Myles, Mr. Thomas Francis",
             "Wirz, Mr. Albert",
             "Hirvonen, Mrs. Alexander (Helga E Lindqvist)"],
    'Sex': ['male', 'female', 'male', 'male', 'female'],
    'Age': [34.5, 47.0, 62.0, 27.0, 22.0],
    'SibSp': [0, 1, 0, 0, 1],
    'Parch': [0, 0, 0, 0, 1],
    'Survived': [1, 0, 1, 0, 1] 
})

# Define the preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('text', Pipeline([
        ('clean_text', FunctionTransformer(text_remove_numbers, validate=False)),  # Apply custom text cleaning function
        ('tfidf', TfidfVectorizer())           # Apply TfidfVectorizer
    ]), 'Name'),
    ('categorical', OneHotEncoder(), ['Sex'])  # Apply one-hot encoding to 'Sex' column
], remainder='passthrough')  # Include other columns in the transformed dataset

# Apply the preprocessing pipeline to the data
X = preprocessor.fit_transform(data.drop('Survived', axis=1))  # Excluding the target column
y = data['Survived']  # Target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
