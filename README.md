# Intrusion Detection using Multi-Classification Techniques on Kyoto_2015_May_day14 Dataset

This repository contains the complete source code and preprocessing techniques used for evaluating multiple machine learning classifiers on the **Kyoto_2015_May_day14 Dataset**.

The objective of this project is to improve the performance of Network Intrusion Detection Systems (NIDS) by testing different machine learning algorithms and comparing their accuracy.

---

## 📝 Project Overview

Network Intrusion Detection is essential for safeguarding digital infrastructure against cyberattacks.  
This project applies multiple machine learning models to identify the best-performing classifiers for intrusion detection based on the Kyoto_2015_May_day14 dataset.

---

## 📚 Dataset

- **Dataset**: Kyoto_2015_May_day14
- **Source**: [Kyoto University Dataset](https://www.takakura.com/Kyoto_data/new_data201704/)
- **Features Selected**: 14 key attributes
- **Target**: Intrusion label (binary classification: normal vs attack)

---

## 🚀 Machine Learning Models Used

- Decision Tree Classifier
- Gradient Boosting Classifier
- Random Forest Classifier
- LightGBM Classifier
- XGBoost Classifier
- K-Nearest Neighbors (KNN)
- Logistic Regression
- AdaBoost Classifier
- Naïve Bayes Classifier
- Support Vector Machine (SVM)
- and more...

---

## ⚙️ Project Structure

Intrusion-Detection-Precision/ │ ├── README.md ├── requirements.txt ├── train_and_evaluate.py ├── data/ │ └── kyoto_2015_may_day14.csv ├── results/ │ └── graphs/ └── models/ (optional, for saving trained models)

yaml
Always show details

Copy

---


## 🛠️ Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Intrusion-Detection-Precision.git
   cd Intrusion-Detection-Precision

# 📚 Kyoto Dataset Classifier Training

---

## 📦 Installation and Setup

1. Install required libraries:

    ```bash
    pip install -r requirements.txt
    ```

2. Place the **Kyoto dataset CSV file** inside the `data/` directory.

---

## 🚀 How to Run

Run the training script:

```bash
python train_and_evaluate.py
```

---

## 📈 Results (Accuracy)

| Model                 | Accuracy Achieved |
|-----------------------|-------------------|
| Decision Tree         | 100%               |
| Random Forest         | 100%               |
| Gradient Boosting     | 100%               |
| LightGBM              | 100%               |
| XGBoost               | ~99.99%            |
| KNN                   | ~99.91%            |
| Logistic Regression   | ~99.76%            |
| Naïve Bayes           | ~99.80%            |
| AdaBoost              | ~99.73%            |

✅ **Top classifiers achieved perfect accuracy on different train-test splits (80-20, 70-30, 60-40).**

Visualizations and comparison graphs are available in the `results/graphs/` folder.

---

## 🧠 Techniques Used

- Label Encoding for categorical attributes
- Handling missing values
- Feature Scaling (Standardization)
- Train-Test Splitting
- Model Evaluation:
  - Accuracy
  - Confusion Matrix
  - Classification Report
- Comparative Analysis across multiple classifiers

---

## 🔗 Useful Links

- 📂 [Kyoto Dataset Download](#)

---

## 🤝 Acknowledgements

Thanks to **Kyoto University** for providing the Kyoto Dataset.

---

## 🛠️ Libraries Used

- Scikit-learn
- XGBoost
- LightGBM
- Matplotlib
- And other Python libraries for development support
