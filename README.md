# SMS Spam Classification Using Linear SVM

A machine learning project that classifies SMS messages as **Spam** or **Ham (Legitimate Message)** using Natural Language Processing (NLP), TF-IDF feature extraction, and a Linear Support Vector Machine (SVM) classifier.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Data Exploration](#data-exploration)
- [Text Preprocessing](#text-preprocessing)
- [Feature Extraction Using TF-IDF](#feature-extraction-using-tf-idf)
- [Why Linear SVM?](#why-linear-svm)
- [Model Training](#model-training)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Predicting New Messages](#predicting-new-messages)
- [Installation and Usage](#installation-and-usage)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Learning Outcomes](#learning-outcomes)
- [Author](#author)

## Project Overview

SMS spam messages often contain unwanted advertisements, fraudulent offers, suspicious links, or misleading prize notifications. Automatically identifying these messages can help users recognize potentially harmful or unwanted content.

In this project, a supervised machine learning model is developed to classify SMS messages into two categories:

- **Ham (0):** A normal, legitimate SMS message.
- **Spam (1):** An unwanted or potentially fraudulent SMS message.

The project follows a complete machine learning workflow, including data exploration, text cleaning, feature extraction, model training, hyperparameter optimization, evaluation, and prediction on new messages.

## Objectives

The main objectives of this project are to:

1. Explore and understand an SMS spam dataset.
2. Perform exploratory data analysis (EDA) to identify patterns in the messages.
3. Clean and preprocess raw text data.
4. Convert text messages into numerical features using TF-IDF.
5. Implement a Linear Support Vector Machine classifier.
6. Optimize the model's hyperparameters using GridSearchCV.
7. Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
8. Predict whether newly provided SMS messages are spam or legitimate.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Google Colab | Cloud-based development environment |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| NLTK | Stopword removal and lemmatization |
| WordCloud | Visualizing frequent words |
| Scikit-learn | Feature extraction, model training, tuning, and evaluation |
| TF-IDF | Converting text into numerical features |
| Linear SVM | SMS classification |

## Dataset

**Dataset:** SMS Spam Collection Dataset

**Source:** [kaggle  — SMS Spam Collection]( https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download)

The dataset contains SMS messages labeled as either spam or ham.

The project uses the `v1` column for the message category and the `v2` column for the message text.

### Initial Dataset Summary

| Property | Value |
|---|---:|
| Total records | 5,572 |
| Original columns | 2 relevant columns |
| Ham messages | 4,825 |
| Spam messages | 747 |
| Missing values | 0 |
| Duplicate records | 403 |

The dataset is imbalanced because legitimate messages substantially outnumber spam messages. Therefore, accuracy alone is not sufficient to judge the model's effectiveness.

*Note: Duplicate records are removed during preprocessing. Dataset statistics above describe the initial data inspection.*

## Project Workflow

The project follows these steps:

1. Load the SMS dataset.
2. Inspect the data structure and check missing values.
3. Identify and remove duplicate records.
4. Perform exploratory data analysis.
5. Clean and normalize the text messages.
6. Encode the labels as numerical values.
7. Split the dataset into training and testing sets.
8. Extract text features using TF-IDF.
9. Train the Linear SVM classifier.
10. Evaluate the initial model.
11. Tune the hyperparameter using GridSearchCV.
12. Evaluate the tuned model.
13. Test the model on new SMS messages.

## Data Exploration

Exploratory Data Analysis (EDA) is performed to understand the dataset before training the model.

The analysis includes:

- Checking the dataset's shape and column names.
- Identifying missing values and duplicate records.
- Counting ham and spam messages.
- Calculating message character counts and word counts.
- Visualizing the distribution of message lengths.
- Generating word clouds to examine frequently occurring words in spam and ham messages.

These visualizations help identify differences in message length and vocabulary between the two categories.

## Text Preprocessing

Raw SMS messages may contain URLs, email addresses, numbers, punctuation, and words that do not contribute much to classification. Text preprocessing helps standardize the messages before feature extraction.

The preprocessing pipeline includes the following operations:

1. **Lowercasing:** Converts all text to lowercase so that words such as `FREE` and `free` are treated consistently.
2. **URL removal:** Removes web links from messages.
3. **Email removal:** Removes email addresses.
4. **Digit removal:** Removes numerical characters.
5. **Punctuation removal:** Removes punctuation and other unwanted characters.
6. **Stopword removal:** Removes common English words using NLTK stopwords.
7. **Lemmatization:** Uses WordNetLemmatizer to reduce words to their base forms.
8. **Label encoding:** Maps `ham` to `0` and `spam` to `1`.

Duplicate records are removed before the final train-test split.

### Example

**Original message:**

> Congratulations! You have won a free prize. Click the link now!

**After preprocessing:**

The message is converted to lowercase, URLs and punctuation are removed, stopwords are filtered, and remaining words are lemmatized.

*The exact cleaned output depends on the preprocessing function and the stopword list.*

### Important Consideration

Removing digits, URLs, punctuation, and stopwords may also remove useful spam indicators. For example, a message containing a phone number or a suspicious link might carry valuable classification information. This preprocessing strategy is a baseline that can be improved through experimentation.

## Feature Extraction Using TF-IDF

Machine learning classifiers require numerical input, but SMS messages are text. The project uses **Term Frequency–Inverse Document Frequency (TF-IDF)** to represent each message as a numerical feature vector.

TF-IDF gives greater importance to words that are informative for a particular message and less importance to words that occur frequently across many messages.

The general formula is:

\[
\operatorname{TFIDF}(t,d)
=
\operatorname{TF}(t,d)
\times
\operatorname{IDF}(t)
\]

Where:

- \(t\) represents a term or word.
- \(d\) represents a document or SMS message.
- TF measures how frequently a term appears in a document.
- IDF measures how informative the term is across the document collection.

### TF-IDF Configuration

The project uses the following settings:

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)
```

- `max_features=5000`: Limits the vocabulary to a maximum of 5,000 features.
- `ngram_range=(1, 2)`: Considers both unigrams (single words) and bigrams (two-word sequences).

For example, the phrase `free prize` can be represented using individual word features and a two-word phrase feature.

TF-IDF produces a high-dimensional, sparse feature matrix that can be used by the classifier.

The vectorizer is fitted on the training data within a Scikit-learn pipeline, preventing the test data from influencing vocabulary fitting.

## Why Linear SVM?

The classifier used in this project is **Linear Support Vector Machine (Linear SVM)**, implemented with Scikit-learn's `LinearSVC`.

Linear SVM learns a decision boundary that separates spam messages from legitimate messages based on their TF-IDF feature vectors.

The decision boundary can be represented conceptually as:

\[
w^T x+b=0
\]

Where:

- \(x\) is the numerical feature vector of an SMS.
- \(w\) represents the learned feature weights.
- \(b\) is the bias term.

### Reasons for Selecting Linear SVM

**1. Suitable for high-dimensional text data**

TF-IDF can produce thousands of features. Linear SVM is well suited to many text classification tasks involving high-dimensional, sparse representations.

**2. Effective classification performance**

The model achieved strong test-set accuracy and spam F1-score in this experiment.

**3. Regularization**

The `C` parameter controls the trade-off between regularization and classification errors on the training data.

**4. Efficient implementation**

`LinearSVC` provides an efficient linear classifier for this type of task.

**5. Compatibility with TF-IDF**

Linear SVM works naturally with the sparse numerical features produced by the TF-IDF vectorizer.

Naive Bayes and Logistic Regression are also reasonable alternatives. A fair comparison using the same data splits and evaluation metrics would be needed to determine which model performs best for this particular dataset.

## Model Training

The dataset is divided into training and testing subsets.

### Train-Test Split

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The configuration means:

- **Training set:** 80% of the cleaned dataset.
- **Testing set:** 20% of the cleaned dataset.
- `random_state=42`: Makes the split reproducible.
- `stratify=y`: Preserves the approximate class proportions in both subsets.

After duplicate removal, the split used in the project contains:

| Subset | Number of messages |
|---|---:|
| Training set | 4,135 |
| Testing set | 1,034 |
| Total | 5,169 |

### Machine Learning Pipeline

The project combines TF-IDF feature extraction and classification into a Scikit-learn pipeline.

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )
    ),
    (
        "svm",
        LinearSVC()
    )
])
```

The pipeline ensures that text vectorization and classification are performed consistently during training and prediction.

## Hyperparameter Tuning

The project uses `GridSearchCV` to find a suitable value for the Linear SVM `C` parameter.

The tested values are:

```python
param_grid = {
    "svm__C": [0.1, 0.5, 1, 2, 5, 10]
}
```

### Cross-Validation Configuration

- **Method:** StratifiedKFold
- **Number of folds:** 5
- **Scoring metric:** F1-score
- **Hyperparameter:** `svm__C`

GridSearchCV evaluates the candidate values across stratified training folds and selects the value with the highest mean cross-validation F1-score.

### Best Parameter

| Property | Value |
|---|---|
| Best `C` value | 2 |
| Best mean cross-validation F1-score | Approximately 0.9052 |

The tuned model is then evaluated on the held-out test set.

## Evaluation Metrics

The project uses several evaluation metrics because the dataset contains substantially more ham messages than spam messages.

### Accuracy

The proportion of all predictions that are correct.

\[
\text{Accuracy}=
\frac{TP+TN}{TP+TN+FP+FN}
\]

### Precision

The proportion of messages predicted as spam that are actually spam.

\[
\text{Precision}=\frac{TP}{TP+FP}
\]

High precision means fewer legitimate messages are incorrectly flagged as spam.

### Recall

The proportion of actual spam messages correctly identified by the model.

\[
\text{Recall}=\frac{TP}{TP+FN}
\]

High recall means fewer spam messages are missed.

### F1-Score

The harmonic mean of precision and recall.

\[
F1=2\times
\frac{\text{Precision}\times\text{Recall}}
{\text{Precision}+\text{Recall}}
\]

The F1-score is especially useful when both false alarms and missed spam messages matter.

Where:

- `TP`: Spam correctly predicted as spam.
- `TN`: Ham correctly predicted as ham.
- `FP`: Ham incorrectly predicted as spam.
- `FN`: Spam incorrectly predicted as ham.

## Results

The tuned Linear SVM model achieved the following results on the held-out test dataset.

### Overall Performance

| Metric | Result |
|---|---:|
| Accuracy | 98.45% |
| Spam precision | 97.52% |
| Spam recall | 90.08% |
| Spam F1-score | 93.65% |

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Ham | 98.59% | 99.67% | 99.13% | 903 |
| Spam | 97.52% | 90.08% | 93.65% | 131 |

*Values are rounded. Support represents the number of actual examples in each test class.*

### Interpretation

- The model correctly classified approximately 98.45% of test messages.
- Spam precision of 97.52% indicates that most messages predicted as spam were genuinely spam.
- Spam recall of 90.08% indicates that the model identified about nine out of every ten spam messages in the test set.
- Spam F1-score of 93.65% shows a strong balance between spam precision and recall.

Despite the high accuracy, the model still misses some spam messages. This is why class-specific recall and F1-score are important for evaluating spam detection.

### Cross-Validation Result

A separate five-fold cross-validation experiment reported a mean accuracy of approximately **97.93%**.

This result should be interpreted carefully: the same dataset was also used during hyperparameter selection. Therefore, this score is not a fully independent estimate of generalization performance after tuning. A nested cross-validation procedure or a separate validation set would provide a more rigorous estimate.

## Predicting New Messages

After training, the model can classify new SMS messages using the same preprocessing function and fitted pipeline.

Conceptually, prediction follows these steps:

1. Accept a new SMS message.
2. Apply the project's text-cleaning function.
3. Convert the cleaned text into TF-IDF features using the fitted vectorizer.
4. Pass the feature vector to the trained Linear SVM.
5. Return the predicted class.

Example:

```python
def predict_sms(message):
    cleaned_message = clean_text(message)
    prediction = best_model.predict([cleaned_message])[0]

    if prediction == 1:
        return "Spam"
    return "Ham"
```

This function assumes that `clean_text` and the tuned pipeline `best_model` have already been defined and fitted.

Example messages for testing:

```python
messages = [
    "Hey, are we meeting tomorrow?",
    "Congratulations! You have won a free cash prize. Claim now!",
    "Please call me when you arrive."
]

for message in messages:
    print(message, "->", predict_sms(message))
```

The output depends on the fitted model and the exact preprocessing implementation.

**Important:** In the project experiment, a suspicious message advertising an iPhone prize was incorrectly classified as ham. This demonstrates that high overall accuracy does not guarantee that every spam message will be detected.

## Installation and Usage

The project was developed in Google Colab. To reproduce the experiment, you will need Python and the listed libraries.

### 1. Clone the Repository

Replace the placeholder with the actual GitHub repository URL.

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud
```

### 3. Prepare the Dataset

Download the SMS Spam Collection dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

Place the CSV file in the project directory or upload it to Google Colab.

The notebook expects a CSV containing the `v1` and `v2` columns, or the corresponding columns must be renamed to match the code.

### 4. Download NLTK Resources

The notebook may require the following NLTK resources:

```python
import nltk

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

### 5. Run the Notebook

Open the `.ipynb` notebook in Google Colab or Jupyter Notebook and execute the cells in order, from dataset loading to model evaluation and prediction.

**Reproducibility note:** The exact results may vary if the dataset, preprocessing logic, train-test split, vectorizer settings, or model parameters are changed.

## Limitations

Although the project achieves strong test-set results, several limitations remain:

1. **Dataset limitations:** The model was trained on one SMS dataset and may not generalize to all languages, regions, or modern spam patterns.
2. **Class imbalance:** Ham messages are more common than spam messages, so accuracy alone can be misleading.
3. **False negatives:** Some spam messages are incorrectly classified as ham.
4. **Preprocessing trade-offs:** Removing numbers, URLs, punctuation, and stopwords can discard useful information.
5. **Limited model comparison:** The experiment focuses on Linear SVM rather than conducting a complete benchmark against alternative classifiers.
6. **No production deployment:** The project demonstrates a classification workflow and does not include a deployed API or application.
7. **Evaluation limitations:** Hyperparameter selection and cross-validation on the same dataset can produce an optimistic performance estimate if the evaluation process is not carefully separated.

## Future Improvements

Potential improvements include:

- Compare Linear SVM with Multinomial Naive Bayes, Logistic Regression, and other text classifiers.
- Experiment with different TF-IDF vocabulary sizes and n-gram ranges.
- Evaluate class weighting or decision-threshold alternatives where appropriate.
- Preserve and test useful indicators such as URLs, numbers, and suspicious phrases.
- Add precision-recall curves and a confusion matrix for more detailed error analysis.
- Use nested cross-validation or a separate untouched test set for a more robust performance estimate.
- Test the model on more recent and diverse SMS datasets.
- Save the trained pipeline for reuse.
- Build a simple web interface using Flask or Streamlit.
- Explore multilingual SMS classification.

## Learning Outcomes

Through this project, I practiced:

- Working with a real-world text classification dataset.
- Performing exploratory data analysis.
- Cleaning and preprocessing textual data.
- Applying TF-IDF for numerical text representation.
- Implementing a Linear SVM classifier.
- Building a Scikit-learn machine learning pipeline.
- Using GridSearchCV and stratified cross-validation.
- Evaluating models using precision, recall, F1-score, and accuracy.
- Understanding the importance of false negatives and model limitations.

## Author

**Laiba Sumaiya Nazim**

BSc in Information and Communication Engineering (ICE)  
Bangladesh University of Professionals (BUP)

GitHub: [puuumeow](https://github.com/puuumeow)

---

*This project was developed for educational purposes to explore NLP and supervised machine learning for SMS spam classification.*
