import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report,accuracy_score, roc_auc_score
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import StratifiedKFold
from urllib.parse import urlparse
import re

def extract_features(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path
    query = parsed_url.query
    features = {
        'url_length': len(url),
        'domain_length': len(domain),
        'path_length': len(path),
        'contains_https': 1 if url.startswith("https") else 0,
        'contains_www': 1 if 'www' in domain else 0,
        'contains_subdomain': 1 if domain.count('.') > 1 else 0,
        'contains_at_symbol': 1 if '@' in url else 0,
        'contains_hyphen': 1 if '-' in domain else 0,
        'contains_digits': sum(c.isdigit() for c in url),
        'num_params': len(query.split('&')) if query else 0,
        'num_dots': url.count('.'),
        'num_slashes': url.count('/'),
        'num_special_chars': sum(c in "@?#%=&$!" for c in url),
        'contains_ip_address': 1 if re.match(r'\d+\.\d+\.\d+\.\d+', domain) else 0,
        'num_subdirectories': path.count('/'),
        'contains_encoded_chars': 1 if '%' in url else 0,
        'contains_redirect': 1 if '//' in path[1:] else 0,
    }
    return features

def preprocess_data(data, feature_extraction_func):
    feature_list = [feature_extraction_func(url) for url in data['url']]
    feature_df = pd.DataFrame(feature_list)
    scaler = MinMaxScaler()
    feature_df_scaled = pd.DataFrame(scaler.fit_transform(feature_df), columns=feature_df.columns)
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_features = vectorizer.fit_transform(data['url'].str.lower())
    X_combined = hstack([csr_matrix(feature_df_scaled.values), tfidf_features])
    return X_combined, data['type'], vectorizer, scaler

def train_and_save_model(data_path, model_path='url_model1.pkl'):
    data = pd.read_csv(data_path)
    data.dropna(inplace=True)
    X, y, vectorizer, scaler = preprocess_data(data, extract_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=55, stratify=y)
    base_models = [
        ('multinb', MultinomialNB()),
        ('log_reg_base', LogisticRegression(max_iter=2000, random_state=50, n_jobs=1))
    ]
    model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(max_iter=2000, random_state=50, n_jobs=1),
        cv=StratifiedKFold(n_splits=5)
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("AUC-ROC Score:", roc_auc_score(y_test, y_pred_proba))
    joblib.dump({'model': model, 'vectorizer': vectorizer, 'scaler': scaler}, model_path)

if __name__ == "__main__":
    train_and_save_model('dataset/URL dataset.csv');
