import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Function to parse the training and test data files
def parse_train_file(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(' ::: ')
            if len(parts) == 4:
                data.append((parts[2], parts[3]))  # genre, description
    return data

def parse_test_file(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(' ::: ')
            if len(parts) == 3:
                data.append(parts[2])  # description only
    return data

def parse_test_solution_file(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(' ::: ')
            if len(parts) == 4:
                data.append(parts[2])  # genre only
    return data

# Load training data
train_data = parse_train_file('train_data.txt')

# Load test data
test_data = parse_test_file('test_data.txt')
test_solutions = parse_test_solution_file('test_data_solution.txt')

# Convert to DataFrame
train_df = pd.DataFrame(train_data, columns=['genre', 'description'])
test_df = pd.DataFrame({'description': test_data, 'genre': test_solutions})

# Display first few rows of training data
print(train_df.head())
print(test_df.head())

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the training data
X_train_tfidf = vectorizer.fit_transform(train_df['description'])

# Transform the test data
X_test_tfidf = vectorizer.transform(test_df['description'])

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, train_df['genre'])

# Predict
nb_predictions = nb_model.predict(X_test_tfidf)

# Evaluate
print("Naive Bayes Accuracy:", accuracy_score(test_df['genre'], nb_predictions))
print(classification_report(test_df['genre'], nb_predictions))

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, train_df['genre'])

# Predict
lr_predictions = lr_model.predict(X_test_tfidf)

# Evaluate
print("Logistic Regression Accuracy:", accuracy_score(test_df['genre'], lr_predictions))
print(classification_report(test_df['genre'], lr_predictions))

# Support Vector Machine
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, train_df['genre'])

# Predict
svm_predictions = svm_model.predict(X_test_tfidf)

# Evaluate
print("SVM Accuracy:", accuracy_score(test_df['genre'], svm_predictions))
print(classification_report(test_df['genre'], svm_predictions))

# Evaluate all models
models = {
    'Naive Bayes': (nb_model, nb_predictions),
    'Logistic Regression': (lr_model, lr_predictions),
    'Support Vector Machine': (svm_model, svm_predictions)
}

for model_name, (model, predictions) in models.items():
    print(f"=== {model_name} ===")
    print(f"Accuracy: {accuracy_score(test_df['genre'], predictions)}")
    print(classification_report(test_df['genre'], predictions))
    print()

