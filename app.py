from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from urllib.parse import urlparse
import re
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)
email_model = joblib.load('phishing_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
model_data = joblib.load('url_model1.pkl')
url_model = model_data['model']
url_vectorizer = model_data['vectorizer']
scaler = model_data['scaler']

def extract_url_features(url):
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

def explain_email_decision(email_text, prediction):
    reasons = []
    if prediction == 1:
        if "http://" in email_text or "https://" in email_text:
            reasons.append("Contains suspicious links.")
        if re.search(r'\b(win|free|prize|urgent|offer|lottery|reward|congratulations|claim)\b', email_text, re.IGNORECASE):
            reasons.append("Contains words commonly used in phishing emails (e.g., 'win', 'free', 'urgent', 'prize').")
        if re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', email_text):
            reasons.append("Contains suspicious or unusual email addresses.")
        if re.search(r'\b(password|credit card|PIN|SSN|account number)\b', email_text, re.IGNORECASE):
            reasons.append("Contains a request for sensitive information like passwords or credit card details.")
        if re.search(r'\b(click here|update now|log in|verify|reset)\b', email_text, re.IGNORECASE):
            reasons.append("Encourages clicking on links or taking immediate action, a common phishing tactic.")
        if email_text.isupper():
            reasons.append("Email text is entirely uppercase, which can indicate urgency or spam.")
        if re.search(r'[0-9]{4,}', email_text):
            reasons.append("Contains long numeric sequences, often used in phishing links.")
        if re.search(r'\b(account will be suspended|legal action|immediate response required|final notice)\b', email_text, re.IGNORECASE):
            reasons.append("Contains threatening language to pressure the recipient into action.")
        if re.search(r'From:.*@(?!trusted-domain\.com)', email_text, re.IGNORECASE):
            reasons.append("Sender's email domain does not match the claimed organization.")
        if re.search(r'\b(gratulations|acheive|recieve|comfirm|verfy|urgnt)\b', email_text, re.IGNORECASE):
            reasons.append("Contains misspellings or poor grammar, which is common in phishing emails.")
        if re.search(r'\b(urgent|immediately|act now|as soon as possible)\b', email_text, re.IGNORECASE):
            reasons.append("Creates a sense of urgency to trick the recipient.")
        if re.search(r'\b(attachment|file|invoice|download)\b', email_text, re.IGNORECASE):
            reasons.append("Mentions attachments, which could contain malicious files.")
        if email_text.count("http://") + email_text.count("https://") > 3:
            reasons.append("Contains an unusually high number of links, which is suspicious.")
    else:
        reasons.append("Does not contain suspicious links or keywords.")
        reasons.append("No signs of phishing behavior detected.")
    
    return reasons

def explain_url_decision(features, prediction):
    reasons = []
    if prediction == 1:
        if features['url_length'] > 675:
            reasons.append("URL is unusually long, which is often suspicious.")
        if features['domain_length'] > 50:
            reasons.append("Domain name is unusually long, which is a red flag.")
        if features['path_length'] > 100:
            reasons.append("URL path is too long, often seen in phishing URLs.")
        if not features['contains_https']:
            reasons.append("URL does not use HTTPS, indicating it may not be secure.")
        if features['contains_at_symbol']:
            reasons.append("URL contains '@' symbol, which can redirect users to a different domain.")
        if features['contains_hyphen']:
            reasons.append("URL contains hyphens, which are common in phishing URLs.")
        if features['contains_subdomain']:
            reasons.append("URL has multiple subdomains, which may indicate phishing.")
        if features['contains_digits'] > 10:
            reasons.append("URL contains an unusually high number of digits, which is suspicious.")
        if features['num_params'] > 5:
            reasons.append("URL contains too many query parameters, which is unusual and suspicious.")
        if features['num_dots'] > 5:
            reasons.append("URL contains too many dots, often seen in phishing domains.")
        if features['num_slashes'] > 5:
            reasons.append("URL contains too many slashes, indicating a deep directory structure.")
        if features['num_special_chars'] > 5:
            reasons.append("URL contains too many special characters, which is often suspicious.")
        if features['contains_ip_address']:
            reasons.append("URL contains an IP address instead of a domain, which is suspicious.")
        if features['num_subdirectories'] > 3:
            reasons.append("URL has too many subdirectories, which can indicate phishing.")
        if features['contains_encoded_chars']:
            reasons.append("URL contains encoded characters, often used to obfuscate phishing links.")
        if features['contains_redirect']:
            reasons.append("URL contains multiple forward slashes ('//'), which may indicate redirection to another domain.")
        if features.get('tld') in ['.xyz', '.top', '.info', '.tk', '.cn', '.ru']:
            reasons.append("URL uses a top-level domain commonly associated with phishing.")
        if any(keyword in features.get('url', '').lower() for keyword in ['login', 'secure', 'verify', 'account', 'update']):
            reasons.append("URL contains keywords like 'login' or 'verify', which are common in phishing attempts.")
    
    else:
        if features['contains_https']:
            reasons.append("URL uses HTTPS, indicating a secure connection.")
        if not features['contains_at_symbol']:
            reasons.append("URL does not contain '@' symbol, which is a good sign.")
        if not features['contains_hyphen']:
            reasons.append("URL does not contain hyphens, which is a good sign.")
        if not features['contains_subdomain']:
            reasons.append("URL does not have suspicious subdomains.")
        if features['contains_digits'] <= 10:
            reasons.append("URL contains a reasonable number of digits.")
        if features['num_params'] <= 5:
            reasons.append("URL has a reasonable number of query parameters.")
        if features['num_dots'] <= 5:
            reasons.append("URL contains a normal number of dots.")
        if features['num_slashes'] <= 5:
            reasons.append("URL contains a normal number of slashes.")
        if features['num_special_chars'] <= 5:
            reasons.append("URL contains a normal number of special characters.")
        if not features['contains_ip_address']:
            reasons.append("URL does not contain an IP address, which is a good sign.")
        if features['num_subdirectories'] <= 3:
            reasons.append("URL has a reasonable number of subdirectories.")
        if not features['contains_encoded_chars']:
            reasons.append("URL does not contain encoded characters, which is a good sign.")
        if not features['contains_redirect']:
            reasons.append("URL does not contain unexpected redirections.")
        if features.get('tld') not in ['.xyz', '.top', '.info', '.tk', '.cn', '.ru']:
            reasons.append("URL uses a top-level domain commonly associated with legitimate websites.")
    
    return reasons

@app.route('/email_predict', methods=['POST'])
def email_predict():
    email_text = request.form.get('email_text', "")
    email_result = None
    email_reasons = []
    color = "black"

    if email_text.strip():
        email_vector = vectorizer.transform([email_text])
        email_prediction = email_model.predict(email_vector)[0]
        email_result = "Phishing Email Detected!" if email_prediction == 1 else "Safe Email"
        email_reasons = explain_email_decision(email_text, email_prediction)
        color = "red" if email_prediction == 1 else "green"

    return render_template('mail.html', 
                           logged_in=session.get('logged_in', False), 
                           username=session.get('username', ''), 
                           email_result=email_result, 
                           prediction_color=color, 
                           email_text=email_text,
                           email_reasons=email_reasons)

@app.route('/url_predict', methods=['POST'])
def url_predict():
    url = request.form.get('url', "")
    url_result = None
    url_reasons = []
    color = "black"

    if url.strip():
        url_features = extract_url_features(url)
        features_df = pd.DataFrame([url_features])
        scaled_features = scaler.transform(features_df)
        scaled_sparse = csr_matrix(scaled_features)
        tfidf_features = url_vectorizer.transform([url])
        combined_features = hstack([scaled_sparse, tfidf_features])
        
        if hasattr(url_model, 'predict'):
            probabilities = url_model.predict_proba(combined_features)[0]
            threshold = 0.7
            url_prediction = 0 if probabilities[0] > threshold else 1
            url_result = "Phishing URL" if url_prediction == 1 else "Legitimate URL"
            url_reasons = explain_url_decision(url_features, url_prediction)
            color = "red" if url_prediction == 1 else "green"
        else:
            url_result = "Model not loaded correctly."

    return render_template('url.html', 
                           logged_in=session.get('logged_in', False), 
                           username=session.get('username', ''),
                           url_result=url_result, 
                           prediction_color=color,
                           url_text=url, 
                           url_reasons=url_reasons)

# Home Route
@app.route('/')
def home():
    return render_template('main.html', logged_in=session.get('logged_in', False), username=session.get('username', ''))

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username == "admin" and password == "admin":
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login1.html', error="Invalid credentials. Try again.")

    return render_template('login1.html')

@app.route('/logout')
def logout():
    session.clear()  
    return redirect(url_for('home'))  

@app.route('/url-phishing')
def url_phishing():
    if not session.get('logged_in'):
        return redirect(url_for('login'))  
    return render_template('url.html', logged_in=True, username=session.get('username', ''))

@app.route('/email-phishing')
def email_phishing():
    if not session.get('logged_in'):
        return redirect(url_for('login'))  
    return render_template('mail.html', logged_in=True, username=session.get('username', ''))


if __name__ == '__main__':
    app.run(debug=True)