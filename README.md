# Movie Review Sentiment Analysis

## Project Overview

This project develops a machine learning solution to classify movie reviews as positive or negative based on their text.

The project was created for **Film Junky Union**, a community for classic movie enthusiasts. The objective was to build a sentiment classification model capable of identifying negative reviews while achieving an **F1 score of at least 0.85**.

The project includes exploratory data analysis, text preprocessing, TF-IDF feature extraction, model comparison, and testing on custom movie reviews.

---

## Dataset

The dataset contains **47,331 movie reviews** along with movie metadata and sentiment labels.

During the initial data analysis:

- The positive and negative classes were found to be well balanced.
- Only a small number of missing values were identified.
- No duplicated records were found.
- Training and test sets showed similar sentiment distributions.

The target variable is:

- `pos = 1` → Positive review
- `pos = 0` → Negative review

---

## Project Workflow

The project follows these main steps:

1. Data inspection and exploratory data analysis
2. Text cleaning and normalization
3. Train/test dataset preparation
4. TF-IDF feature extraction
5. Baseline model evaluation
6. Logistic Regression modeling
7. spaCy lemmatization
8. LightGBM modeling
9. Model comparison
10. Testing predictions on custom reviews

---

## Models

### Baseline

A `DummyClassifier` was used to establish a baseline for comparison.

### Model 1 — TF-IDF + Logistic Regression

Normalized review text was transformed using TF-IDF and classified using Logistic Regression.

This model achieved the strongest performance while maintaining a relatively simple preprocessing pipeline.

### Model 2 — spaCy + TF-IDF + Logistic Regression

Reviews were lemmatized using spaCy before TF-IDF transformation.

The additional preprocessing produced results very similar to Model 1, indicating that lemmatization did not provide a meaningful performance improvement for this dataset.

### Model 3 — spaCy + TF-IDF + LightGBM

The same lemmatized TF-IDF features were used with a LightGBM classifier to compare Logistic Regression with a gradient boosting approach.

---

## Model Performance

| Model | Test Accuracy | Test F1 | Test APS | Test ROC AUC |
|---|---:|---:|---:|---:|
| Baseline | 0.50 | 0.00 | 0.50 | 0.50 |
| TF-IDF + Logistic Regression | **0.88** | **0.88** | **0.95** | **0.95** |
| spaCy + TF-IDF + Logistic Regression | **0.88** | **0.88** | **0.95** | **0.95** |
| spaCy + TF-IDF + LightGBM | 0.86 | 0.86 | 0.93 | 0.94 |

The two Logistic Regression models achieved the highest F1 score of **0.88**, exceeding the project requirement of **0.85**.

Because Model 1 achieved the same test F1 score as the lemmatized version while using a simpler preprocessing pipeline, **TF-IDF + Logistic Regression was selected as the preferred model**.

---

## Testing on New Reviews

The trained models were also tested on a small set of custom movie reviews containing positive, negative, and mixed sentiment.

The models produced confident predictions for most clearly positive and negative examples. Reviews containing mixed sentiment were more challenging and generally produced less consistent probabilities, illustrating the difficulty of classifying text that contains both praise and criticism.

---

## Key Findings

- TF-IDF provides an effective representation for movie review sentiment classification.
- Logistic Regression performed better than LightGBM for this high-dimensional text classification problem.
- spaCy lemmatization did not meaningfully improve test performance.
- The best model achieved an **F1 score of 0.88**, exceeding the required target.
- A simpler preprocessing pipeline achieved the same test performance as the more computationally intensive lemmatization approach.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- TF-IDF
- Logistic Regression
- spaCy
- LightGBM
- NLTK
- Jupyter Notebook

---

## Repository Structure

```text
Movie-Review-Sentiment-Analysis/
│
├── Movie_Review_Sentiment_Analysis.ipynb
├── data.tsv
├── README.md
├── requirements.txt
└── .gitignore

---

## Conclusion

The project demonstrates an end-to-end NLP classification workflow, from exploratory data analysis and text preprocessing to model evaluation and testing on unseen examples.

Among the evaluated approaches, TF-IDF combined with Logistic Regression provided the best balance between predictive performance and model simplicity, achieving an F1 score of 0.88 on the test set.
