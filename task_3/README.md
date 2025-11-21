# Spam vs Ham Email Classifier (TF-IDF + Logistic Regression)

This project builds a **machine learning model** that can distinguish between **spam** and **non-spam (ham)** emails.
It uses a popular text representation technique called **TF-IDF (Term Frequency – Inverse Document Frequency)**
combined with a **Logistic Regression** classifier.

The repository contains:

- A Python script to **train** the model and generate visualizations.
- A trained model saved to disk for **reuse**.
- PNG figures that help you **understand** the dataset and model behaviour.
- This `README.md` as a **user guide**.

---

## 1. Project Structure

After you run the training script once, your folder will look like this:

```text
.
├── spam.csv                      # Input dataset (provided)
├── spam_tfidf_classifier.py      # Training and inference script
├── models/
│   └── spam_tfidf_logreg.joblib  # Saved TF-IDF + Logistic Regression model
└── docs/
    ├── label_distribution.png    # Ham vs spam count plot
    ├── top_spam_terms.png        # Top TF-IDF terms for spam
    └── confusion_matrix_tfidf.png# Confusion matrix for the classifier
```

---

## 2. Dataset: `spam.csv`

The dataset file `spam.csv` is assumed to be located in the **same directory** as the script.

It has at least the following columns:

- **`Category`** – Text label: either `ham` (non-spam) or `spam` (undesired message).
- **`Message`** – The full email / SMS text content.

Example rows:

| Category | Message                                                              |
|----------|----------------------------------------------------------------------|
| ham      | Go until jurong point, crazy.. Available only in bugis n great...   |
| spam     | Free entry in 2 a wkly comp to win FA Cup final tkts 21st May...    |

During training, the script:

- Uses `Category` as the **target** (what we want to predict).
- Uses `Message` as the **input text**.

---

## 3. Requirements and Installation

To run everything, you need **Python 3.8+** and some common libraries.

```bash
# (Optional but recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate      # On Linux/macOS
# .venv\Scripts\activate     # On Windows PowerShell / CMD

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install pandas scikit-learn matplotlib joblib
```

Make sure that:

- `spam.csv` is in the **same folder** as `spam_tfidf_classifier.py` (or you pass the full path using `--data`).
- You have permission to create folders `models/` and `docs/` in the project directory.

---

## 4. How to Run the Script

The main script is **`spam_tfidf_classifier.py`**.

From the folder where the file is located, run:

```bash
python spam_tfidf_classifier.py
```

This will:

1. Read `spam.csv`.
2. Split data into **training** and **test** sets.
3. Build a TF-IDF + Logistic Regression pipeline.
4. Train the model on the training data.
5. Evaluate accuracy and print a **classification report** and **confusion matrix** to the console.
6. Save the trained model into `models/spam_tfidf_logreg.joblib`.
7. Generate PNG files under `docs/` for use in your documentation.

You can also customize folders and dataset path, for example:

```bash
python spam_tfidf_classifier.py   --data spam.csv   --models-dir models   --docs-dir docs
```

---

## 5. What the Script Does (Step by Step)

Below is a high-level, human-readable description of what happens inside the script, in order:

1. **Load the dataset**  
   - Reads `spam.csv` into a pandas DataFrame.  
   - Uses the `Message` column as input texts and `Category` (`ham` or `spam`) as labels.

2. **Split into train and test sets**  
   - Uses `train_test_split` with **stratification**, so the spam/ham ratio is similar in both sets.  
   - Typical split is 80% training, 20% test.

3. **Build the TF-IDF + Logistic Regression pipeline**  
   - `TfidfVectorizer` converts raw text into a sparse numeric matrix, giving higher weights to important words and down-weighting very frequent ones.  
   - `LogisticRegression` learns how to separate spam from ham using these TF-IDF features.

4. **Train the model**  
   - Fits the pipeline on the **training set** only.  
   - The model learns which words and phrases are strong indicators of spam or ham.

5. **Evaluate model performance**  
   - Uses the trained pipeline to predict labels for the **test set**.  
   - Computes **accuracy**, **precision**, **recall**, and **F1-score** for each class.  
   - Computes a **confusion matrix** that shows how many spam/ham messages were correctly or incorrectly classified.

6. **Generate visualizations (PNG files)**  
   - Creates a bar chart of how many `ham` vs `spam` messages exist in the dataset.  
   - Extracts the **top TF-IDF terms for the spam class** (highest logistic regression coefficients) and plots them as a horizontal bar chart.  
   - Plots the confusion matrix as a heatmap-like image with counts printed in each cell.

7. **Save the trained model**  
   - Stores the entire trained pipeline (TF-IDF + classifier) in `models/spam_tfidf_logreg.joblib` using `joblib.dump`.  
   - You can later load this file to classify new, unseen emails without retraining.

---

## 6. Generated Visualizations (PNG Files)

The script produces three main images in the `docs/` folder:

1. **`docs/label_distribution.png`**  
   - A simple bar chart showing how many `ham` and `spam` examples are in the dataset.  
   - Helps you see **class imbalance** (usually many more ham than spam).

2. **`docs/top_spam_terms.png`**  
   - A horizontal bar chart listing the **top TF-IDF terms for the spam class**.  
   - Each bar shows the logistic regression coefficient for that term:  
     - Higher positive values mean the term is a strong indicator that a message is spam.  
   - This is useful for explaining *why* the model marks some messages as spam.

3. **`docs/confusion_matrix_tfidf.png`**  
   - A visualization of the **confusion matrix** with rows = true labels and columns = predicted labels.  
   - Each cell contains the count of messages.  
   - Quickly shows where the model makes mistakes (e.g., spam misclassified as ham).

You can insert these PNG files into your report or presentation as visual support.

---

## 7. Using the Saved Model for New Emails

Once the model has been trained, you can load it and predict new emails without retraining.

### 7.1. Example: Python usage

```python
from pathlib import Path
import joblib

model_path = Path("models/spam_tfidf_logreg.joblib")

# Load the trained pipeline (TF-IDF + Logistic Regression)
pipeline = joblib.load(model_path)

# Example new emails
new_messages = [
    "Congratulations! You have won a free ticket. Click here to claim now!!!",
    "Hi, are we still meeting for lunch tomorrow?"
]

predictions = pipeline.predict(new_messages)

for msg, label in zip(new_messages, predictions):
    print(f"TEXT: {msg}")
    print(f"PREDICTED LABEL: {label}")
    print("-" * 40)
```

Output could look like:

```text
TEXT: Congratulations! You have won a free ticket. Click here to claim now!!!
PREDICTED LABEL: spam
----------------------------------------
TEXT: Hi, are we still meeting for lunch tomorrow?
PREDICTED LABEL: ham
----------------------------------------
```

### 7.2. Example: Command-line workflow

1. Train the model (only once, unless you change the dataset):  
   ```bash
   python spam_tfidf_classifier.py
   ```

2. Use a small custom Python script or notebook (like the example above) to load the model and classify messages.

---

## 8. TF-IDF and Accuracy – Short Explanation for Your Report

- **TF-IDF** transforms raw text into numbers by:  
  - Counting how often each word appears in a message (**term frequency**).  
  - Down-weighting words that appear in almost every message (**inverse document frequency**).  
  - The result is a matrix where each email is a vector of TF-IDF scores for each term.

- **Logistic Regression** uses these TF-IDF vectors to learn a boundary between `spam` and `ham`.  
- **Accuracy** is computed as:  
  

  > **Accuracy = (Number of correct predictions) / (Total number of predictions)**

  

- In our case, on the default train/test split, the model reaches an accuracy around **0.9713**.  
  This means roughly **97.1%** of emails in the test set are correctly labeled as spam or ham.

Remember that high accuracy is good, but for spam detection you should also pay attention to:

- **False negatives** (spam detected as ham) – can be dangerous in real-life usage.  
- **False positives** (ham detected as spam) – can annoy users if important messages are lost.

The confusion matrix and classification report (precision, recall, F1-score) help you analyze these aspects.

---

## 9. Summary

- You now have a **complete pipeline** to classify spam vs ham emails using **TF-IDF** and **Logistic Regression**.  
- The project includes:
  - A training script: `spam_tfidf_classifier.py`  
  - A dataset: `spam.csv`  
  - A trained model: `models/spam_tfidf_logreg.joblib`  
  - Visual explanations in the `docs/` folder (PNG files).  
- You can extend this work by:
  - Trying different models (e.g., Linear SVM, Naive Bayes).  
  - Tuning TF-IDF parameters (ngram range, stop words, min/max document frequency).  
  - Adding more data or cleaning the text (lowercasing, removing punctuation, etc.).

Use this project as the **documentation and code base** for your assignment to show both practical implementation and clear explanation.
