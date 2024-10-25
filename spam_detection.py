import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from joblib import dump

# Function to load data from the folder structure
def load_data_from_folder(folder_path):
    texts = []
    labels = []
    for label in ['ham', 'spam']:
        label_folder = os.path.join(folder_path, label)
        for filename in os.listdir(label_folder):
            if filename.endswith('.txt'):
                file_path = os.path.join(label_folder, filename)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    texts.append(file.read())
                    labels.append(label)
    return texts, labels

# Load all training data
train_texts, train_labels = load_data_from_folder('train_test_mails/train-mails')

# Combine data into a DataFrame
data = pd.DataFrame({
    'text': train_texts,
    'label': train_labels
})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF and Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'classifier__C': [0.1, 1, 10],  # Regularization parameter
    'classifier__solver': ['liblinear', 'lbfgs'],  # Solvers
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model and parameters
best_model = grid_search.best_estimator_

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the best model pipeline
os.makedirs('models', exist_ok=True)
dump(best_model, 'models/best_model.pkl')

print("Best model saved successfully.")