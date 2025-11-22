# Email Classifier using Naive Bayes

This project is a machine learning solution designed to classify emails as either "Spam" or "Ham" (legitimate) using the Multinomial Naive Bayes algorithm. It involves natural language processing (NLP) techniques to process text data and build a predictive model.

## Overview

The workflow of the project includes:

1.  **Data Loading & Exploration**: Loading the email dataset and visualizing class distribution.
2.  **Data Cleaning**: Handling duplicates and missing values.
3.  **Text Preprocessing**:
    - Lowercasing text.
    - Removing URLs, social media handles, hashtags, numbers, and punctuation.
    - Tokenization.
    - Stemming (using SnowballStemmer).
    - Stopword removal.
4.  **Feature Extraction**: Converting text into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency).
5.  **Model Training**: Training a Multinomial Naive Bayes classifier on the processed data.
6.  **Evaluation**: Assessing model performance using accuracy scores and classification reports.
7.  **Saving Artifacts**: Exporting the trained model and vectorizer for future use.

## Dependencies

The project requires the following Python libraries:

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `matplotlib` & `seaborn`: For data visualization.
- `wordcloud`: For generating word clouds of email content.
- `nltk`: For natural language processing tasks (stopwords, stemming).
- `scikit-learn`: For machine learning algorithms and metrics.
- `joblib`: For saving and loading models.

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install pandas numpy matplotlib seaborn wordcloud nltk scikit-learn joblib
```

## Usage

1.  **Prepare Data**: Ensure the dataset `spam_Emails_data.csv` is available. _Note: The notebook currently points to `D:\cuoiky\spam_Emails_data.csv`, you may need to update the path in the code to match your local setup._
2.  **Run Notebook**: Open `multiNB.ipynb` in Jupyter Notebook or a compatible environment.
3.  **Execute Cells**: Run the cells in order to process the data, train the model, and view results.
4.  **Output**:
    - The script generates a cleaned dataset: `cleaned_spam_data.csv`.
    - The trained model is saved as: `model_naivebayes.pkl`.
    - The TF-IDF vectorizer is saved as: `stfidf_vectorizer.pkl`.

## Files in Repository

- `multiNB.ipynb`: The main Jupyter Notebook with the implementation.
- `model_naivebayes.pkl`: Serialized Naive Bayes model.
- `stfidf_vectorizer.pkl`: Serialized TF-IDF vectorizer.
- _(Data files like `spam_Emails_data.csv` are expected to be in the working directory or specified path)_.

## Results

The model evaluates the classification performance on a test set, providing metrics such as precision, recall, and F1-score for both Spam and Ham classes.
