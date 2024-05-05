import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt

# Create a sample dataset
data = {
    'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    'Gender': ['M', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'F', 'M'],
    'Purchased': [0, 1, 0, 1, 1, 1, 0, 1, 0, 1]  # 0: Not purchased, 1: Purchased
}

df = pd.DataFrame(data)

# Convert categorical variable 'Gender' into numerical using LabelEncoder
# method1
# le = LabelEncoder()
# df['Gender'] = le.fit_transform(df['Gender'])
# metho2
df = pd.get_dummies(df, columns=['Gender'], drop_first=True, dtype=int)
x_cols = [col for col in df.columns if col !='Purchased']
print(x_cols)
# Define features and target variable
X = df[x_cols]  # Features: Age and Gender
y = df['Purchased']  # Target variable: Purchased


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create and train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred_prob

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
plt.scatter(y_test, y_pred,
         marker ='o')
plt.legend()
plt.show()


accuracy = accuracy_score(y_test, y_pred_prob)
print("Accuracy:", accuracy)
plt.scatter(y_test, y_pred_prob,
         marker ='o')
plt.legend()
plt.show()

# hyper parameter tunning
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.1, 1, 10],
}

log_reg = LogisticRegression()

grid_search = GridSearchCV(
    estimator=log_reg, param_grid=param_grid, cv=2,
    scoring='accuracy')
grid_search.fit(X_train, y_train)

best_log_reg = grid_search.best_estimator_
y_pred = best_log_reg.predict(X_test)
print("Classification Report:\n", classification_report
      (y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

y_pred_prob = best_log_reg.predict_proba(X_test)[:, 1]
plt.scatter(y_test, y_pred_prob)
plt.show()

