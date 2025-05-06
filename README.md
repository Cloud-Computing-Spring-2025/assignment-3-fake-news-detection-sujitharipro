# Assignment-5-FakeNews-Detection

# Fake News Detection with Spark

## Overview

This project implements a **Fake News Detection pipeline** using **Apache Spark** and **MLlib**. The pipeline performs data preprocessing, feature extraction using **TF-IDF**, model training with **Logistic Regression**, and model evaluation. The model is used to classify news articles as either real or fake.

## Dataset Used

The dataset used for this project is a **Fake News Dataset** containing articles, their titles, and labels indicating whether the article is fake or real.

**Columns:**

* `id`: Unique identifier for each article.
* `title`: Title of the article.
* `text`: Full text of the article.
* `label`: The label, indicating whether the article is fake (0) or real (1).

You can modify the dataset or add more data for enhanced training.

## Tasks Description

### Task 1: Data Preprocessing

* Load the dataset and preprocess it by:

  * Cleaning and handling missing values.
  * Combining the `title` and `text` columns into one.
  * Tokenizing the text and removing stop words.
  * Splitting the dataset into training and test sets.

### Task 2: Feature Extraction

* Apply **TF-IDF** (Term Frequency-Inverse Document Frequency) to convert the text data into a numerical representation suitable for machine learning models.

### Task 3: Model Training and Evaluation

* Train a **Logistic Regression** model using the TF-IDF features and label encoding.
* Evaluate the model's performance using **accuracy** and **F1 score**.

### Task 4: Model Saving and Loading

* Save the trained model and load it to make predictions on new data.

### Task 5: Running the Full Pipeline

* Train the model using the full pipeline and evaluate it on the test dataset.

## How to Run the Code

### Prerequisites

* Python 3.6 or higher
* Apache Spark 3.x
* PySpark

### Installation

1. Clone this repository:

   ```bash
   git clone <repository_url>
   cd fake-news-detection
   ```

2. Install the necessary dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download or place the dataset (`fake_news_sample.csv`) in the project directory.

### Running the Code

1. **Preprocessing & Splitting the Data**:
   To preprocess and split the data into training and testing sets:

   ```bash
   python task4.py
   ```

2. **Train and Evaluate the Model**:
   To train the model and evaluate it:

   ```bash
   python task5.py
   ```

3. **Viewing the Results**:
   After running `task5.py`, you will see the model's accuracy and F1 score printed in the terminal.

### Expected Output

* `train_data.csv`: Preprocessed training data saved as a CSV file.
* `test_data.csv`: Preprocessed test data saved as a CSV file.
* Model accuracy and F1 score in the terminal.


