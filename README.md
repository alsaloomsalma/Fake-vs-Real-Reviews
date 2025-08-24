# Fake-vs-Real-Reviews
Training a machine learning model to identify computer generated vs real reviews
# Fake Review Detection

## Project Overview
This project focuses on detecting **fake reviews** in e-commerce platforms using natural language processing (NLP) and machine learning techniques. Fake reviews mislead consumers, harm trust, and create compliance risks. By analyzing review text patterns and applying both unsupervised and supervised learning models, we aim to improve detection accuracy.

---

## Business Problem
Fake reviews distort consumer perception and impact purchasing decisions. They can lead to financial loss and legal issues for businesses.

**Stakeholders:**
- Trust & Safety Teams at e-commerce companies
- Product Managers responsible for content integrity

**Value Proposition:**
- Improve platform credibility and user trust
- Reduce fraud and rating manipulation
- Automatically flag suspicious reviews for auditing

---

## Dataset
- **Source:** Kaggle Fake Reviews Dataset (Synthetic Data)
- **Size:** ~40,000 reviews (20K fake, 20K real)
- **Structure:** Text-based reviews with binary labels (`fake` or `real`)

---

## Tech Stack
- **Language:** Python 3.x
- **Libraries:**
  - Data Handling: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`, `wordcloud`
  - NLP: `nltk`, `gensim`
  - Machine Learning: `scikit-learn`
  - Dimensionality Reduction: `PCA`
  - Clustering: `KMeans`, `DBSCAN`

---

## 🔍 Key Steps
1. **Data Cleaning & Preprocessing**
   - Removed duplicates and missing values
   - Tokenization, stopword removal, regex-based text cleaning
2. **Feature Engineering**
   - TF-IDF & CountVectorizer
   - Word2Vec embeddings
3. **Unsupervised Learning**
   - KMeans & DBSCAN clustering
   - PCA for dimensionality reduction
4. **Supervised Learning**
   - Logistic Regression
   - Random Forest
   - SVM
5. **Evaluation**
   - Accuracy
   - Confusion Matrix
   - Classification Report

---

## Visualizations
- Word clouds for frequent terms
- Cluster visualizations using PCA
- Distribution plots for review length

---

## Usage
Run the notebook:
```bash
jupyter notebook "Project 3.ipynb"
