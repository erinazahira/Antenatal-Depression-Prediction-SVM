# Antenatal Depression Prediction
This project focuses on predicting the risk of antenatal depression using a machine learning model. The entire workflow, from data preprocessing to model training and evaluation, is documented in the Jupyter Notebook. The final trained model is saved and can be used for real-time predictions.

---

## The Dataset

### **File:** `post natal data.csv`

This dataset contains anonymized features collected from expectant mothers to identify factors associated with antenatal depression.

* **Source:** Kaggle
* **Content:** The dataset consists of 1,503 patient records and includes 15 attributes, of which 9 attributes relevant to pregnant women were selected for this study. Among these 9 attributes, 8 were used as analysis features, while 1 attribute served as the target label.
* **Key Features:**
    * `Age`
    * `Feeling sad or tearful`
    * 'Irritable towards baby & partner'
    * 'Trouble sleeping at night'
    * 'Problems concentrating or making decisions'
    * 'Overeating or loss of appetite'
    * 'Feeling of guilt'
    * 'Suicide attempt'
* **Target Variable:**
    * `Feeling Anxious': A binary variable (1 for high risk, 0 for low risk) indicating the likelihood of antenatal depression.

---

## Jupyter Notebook

### **File:** `Antenatal_Depression_Prediction.ipynb`

This notebook is the core of the project. It provides a step-by-step walkthrough of the entire machine learning pipeline. Anyone can open this notebook to understand the methodology and reproduce the results.

The notebook is structured into the following key stages:

1.  **Data Loading & Initial Exploration:**
    * Loading the dataset using Pandas.
    * Performing Exploratory Data Analysis (EDA) to understand data distributions, correlations, and initial patterns.

2.  **Data Preprocessing & Cleaning:**
    * Handling missing values (`NaN`).
    * Encoding categorical features into a numerical format (based on dataset).
    * Scaling numerical features (`StandardScaler) to prepare them for the model.

3.  **Model Training:**
    * Splitting the data into training and testing sets with 80:20 split.
    * Utilizing SVM algorithm to identify indications of depression based on questionnaire answers with several key parameters.

4.  **Model Evaluation:**
    * Assessing the model's performance on the test set using metrics such as:
        * **Accuracy**
        * **Precision**
        * **Recall**
        * **F1-Score**
        * **5-Fold Cross-Validation**
    * Visualizing the results using a **Confusion Matrix**.

5.  **Model Saving:**
    * The final, trained model is serialized and saved into a `.pkl` file using the `pickle` library for future use.

---

## The Saved Model

### **File:** `svm_model_depression.pkl`

This file is the final, ready-to-use prediction engine. It is a pre-trained model that has learned the patterns from the dataset.

Instead of re-training the model every time, you can simply load this file to make instant predictions on new data.
