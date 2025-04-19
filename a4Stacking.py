import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

data = pd.read_csv("dataset/emails.csv", encoding="latin-1")
data = data[['Text', 'Spam']]
data.columns = ['text', 'label']

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
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_vectorized.toarray())
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=55)
logistic_model = LogisticRegression(max_iter=1000)
naive_bayes_model = MultinomialNB()
model = StackingClassifier(
    estimators=[
        ('logistic', logistic_model),
        ('naive_bayes', naive_bayes_model)
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Classification Report: Ensemble Model\nStacking :MultinomialNB+Logistic Regression")
print(classification_report(y_test, y_pred))
joblib.dump(model, "phishing_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Stacking Model, vectorizer, and scaler saved!")
