# 🛡️ Phishing Detection using Machine Learning and NLP

This project presents a robust and intelligent phishing detection system that leverages **Machine Learning (ML)** and **Natural Language Processing (NLP)** to detect and classify phishing attacks, including **spear-phishing**, **smishing (SMS phishing)**, and **URL-based phishing**. The system employs a **stacking ensemble model** that outperforms traditional single-model approaches in accuracy, precision, and recall.

---

## 📌 Project Overview

Phishing attacks have become a major cybersecurity threat, tricking users into revealing sensitive information through fraudulent emails, messages, or websites. Traditional rule-based systems often fail to detect modern phishing tactics. To overcome these limitations, this project explores:

- Textual and URL-based feature extraction using **TF-IDF** and lexical analysis
- Model comparison between:
  - Multinomial Naïve Bayes
  - Logistic Regression
  - Voting Classifier
  - **Stacking Classifier** (Naïve Bayes + Logistic Regression)
- A **real-time web interface** for phishing detection

---

## 🎯 Objectives

- Improve phishing detection accuracy using ensemble learning
- Reduce false positives using NLP-based feature engineering
- Provide a scalable model capable of identifying new, zero-day phishing attacks
- Enable real-time phishing classification via a web dashboard

---

## 🧠 Methodology

- **Data Collection**: A dataset of ~10,000 emails and URLs from public sources
- **Preprocessing**: Cleaning, tokenization, stopword removal
- **Feature Extraction**:
  - Textual: Urgency indicators, suspicious keywords, sentence structure
  - URL-based: Length, special characters, domain reputation
- **Model Training**: Scikit-learn models trained with 70-30 train-test split & 5-fold cross-validation
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

## ⚙️ Technologies Used

- Python 🐍
- Scikit-learn
- Pandas & NumPy
- Natural Language Toolkit (NLTK)
- Flask (for web interface)
- Matplotlib & Seaborn (for visualizations)

---

## 📊 Results Summary

| Attack Type      | Best Accuracy | Best Precision | Best Recall |
|------------------|---------------|----------------|-------------|
| Spear-Phishing   | 96.4%         | 0.95           | 0.96        |
| Smishing         | 95.8%         | 0.96           | 0.95        |
| URL Phishing     | 97.2%         | 0.96           | 0.97        |

📌 The **Stacking Classifier** consistently outperformed all other models across phishing types.

---

## 🖥️ Web Interface

A user-friendly interface allows users to input:
- Email content
- SMS messages
- URLs

And get real-time phishing detection results.

---

## 🔒 Security Considerations

- Adaptive model updates to combat evolving phishing tactics
- Threshold filtering to reduce false positives
- Feature importance analysis for better model interpretability

---

## 🚀 Future Enhancements

- Integrating deep learning models (e.g., BERT, GPT) for richer text understanding
- Adversarial robustness against phishing obfuscation
- Multi-language support for global phishing detection
- Optimizing for low-resource environments

---

## 👨‍💻 Authors

- Dinesh Kumar Malempati
- Dinesh Krishna Penumarti
- Sindhuri Nandyala
- Lakshmi Sahithya Miryala
- Dr. Vedantham Rama Chandran (Guide)

---

> ⚠️ **Disclaimer**: This project is intended for academic and research purposes. It should not be used as a standalone security solution for critical infrastructure without proper validation and testing.
