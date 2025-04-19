import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

data = pd.read_csv("dataset/spam.csv", encoding="latin-1")
data = data[['v1', 'v2']]
data.columns = ['label', 'text']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

data['text'] = data['text'].apply(preprocess_text)
X = data['text']
y = data['label']
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)
model_nb = MultinomialNB()
model_logreg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
meta_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
stacking_model = StackingClassifier(
    estimators=[('nb', model_nb), ('logreg', model_logreg)],
    final_estimator=meta_model,
    stack_method='predict_proba', 
    cv=5  
)
stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_test)
print("Classification Report: Ensemble Model\nStacking :MultinomialNB+Logistic Regression")
print(classification_report(y_test, y_pred))
joblib.dump(stacking_model, "phishing_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Stacking model and vectorizer saved!")
