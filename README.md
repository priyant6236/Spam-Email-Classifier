# 📧 Spam Email & SMS Classifier

An end-to-end Machine Learning project to detect and classify spam emails and SMS messages. The model implements text preprocessing with TF-IDF vectorization and compares multiple algorithms (Logistic Regression, Support Vector Machines, Random Forest) alongside a Soft Voting Ensemble model for optimal classification performance.

---

## 🚀 Features

- **Automated Text Preprocessing:** Cleans and tokenizes text with TF-IDF n-gram vectorization (`ngram_range=(1, 2)`).
- **Multi-Model Evaluation:** Trains and evaluates:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest Classifier
  - **Soft Voting Ensemble** (combining all models for superior prediction stability)
- **Comprehensive Metrics:** Tracks Accuracy, Precision, Recall, and F1-score with classification reports and automated Confusion Matrix plots.
- **Robust Dataset Handling:** Automatically handles UTF-8 and Latin-1 encodings and supports standard dataset schemas (Kaggle `Category`/`Message`, SMS Spam Collection `v1`/`v2`, or `label`/`text`).
- **Interactive Sample Testing:** Classifies custom user-provided messages with prediction confidence scores.

---

## 📂 Project Structure

```text
Spam Email Classifier/
│
├── Spam_Non-Spam_Email_Detection.py   # Main Python script (training, evaluation, testing)
├── requirements.txt                  # Python dependencies
├── run.bat                           # Windows batch launcher (optional)
├── spam.csv                          # Dataset file (SMS/Email Spam dataset)
├── plots/                            # Directory containing generated confusion matrix plots
└── README.md                         # Project documentation
```

---

## 🛠️ Requirements & Installation

### 1. Clone or Download the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

The project is designed to work with standard SMS/Email Spam datasets (such as the [SMS Spam Collection dataset from Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)).

- Place your dataset named `spam.csv` in the root directory.
- *Note:* If `spam.csv` is not present, the script will automatically create a sample starter dataset so you can run and test the pipeline immediately.

---

## 🏃 Usage

Run the main classification script:

```bash
python Spam_Non-Spam_Email_Detection.py
```

Or on Windows, simply double-click or run:
```cmd
run.bat
```

### What happens when you run it:
1. Loads and checks the dataset distribution.
2. Encodes labels (`ham` -> 0, `spam` -> 1).
3. Splits data into 80% training and 20% stratified test sets.
4. Trains and evaluates each model, printing detailed classification metrics.
5. Saves confusion matrix heatmaps to the `plots/` folder.
6. Trains the Ensemble model and tests custom messages with confidence percentages.

---

## 📈 Sample Results

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | ~96% | High | High | High |
| Support Vector Machine (SVM) | ~98% | High | High | High |
| Random Forest | ~97% | High | High | High |
| **Ensemble (Voting)** | **~98%+** | **High** | **High** | **High** |

---

## 📦 Dependencies

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
