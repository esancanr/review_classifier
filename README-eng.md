# Movie Review Sentiment Classification

A **Natural Language Processing (NLP)** and **Machine Learning** project developed in Python to classify movie reviews from a dataset as **positive** or **negative**.

The project implements a complete Machine Learning workflow that includes data loading and exploration, creation of an imbalanced dataset, balancing using *Random Under-Sampling*, data splitting, text transformation using **TF-IDF**, training of different classification algorithms, and evaluation of their results.

---

## Project Objective

The main objective is to build and compare different Machine Learning models capable of determining the sentiment of a review.

The project compares the following algorithms:

* **Support Vector Machine (SVM)**
* **Decision Tree**
* **Gaussian Naive Bayes**
* **Logistic Regression**

This makes it possible to determine which model achieves the best performance for sentiment classification using TF-IDF text representations.

---

## Technologies Used

* **Python**
* **Pandas**
* **Scikit-learn**
* **imbalanced-learn**
* **TF-IDF**
* **Machine Learning**
* **Natural Language Processing (NLP)**

---

## Project Structure

```text
project/

│

├── data/

│   └── IMDB Dataset.csv

│

├── discovery_data.py

│

└── README-eng.md
```

### Dataset

The project uses the following file:

```text
data/IMDB Dataset.csv
```

This dataset contains movie reviews along with their corresponding sentiment:

* `positive`
* `negative`

---

# Project Development

## 1. Load the Dataset

First, the necessary libraries are imported and the dataset is loaded using **Pandas**.

```python
data = pd.read_csv('data/IMDB Dataset.csv')
```

This allows the reviews to be stored in a `DataFrame` for subsequent filtering, transformation, and analysis operations.

---

## 1.1. Check the Number of Data Points per Sentiment

The following can be used:

```python
data_count_balance = data.value_counts('sentiment')

print(data_count_balance)
```

to determine how many reviews exist for each category.

This allows us to check the original sentiment distribution:

```text
positive

negative
```

This step is important before training the models because it allows us to determine whether the classes are balanced.

---

# 2. Create an Imbalanced Dataset

To experiment with data balancing techniques, an intentionally imbalanced dataset is created.

The following are selected:

```python
df_postive = data[data['sentiment'] == 'positive'][:9000]

df_negative = data[data['sentiment'] == 'negative'][:1000]
```

Therefore, the following distribution is obtained:

| Sentiment |  Quantity |
| --------- | --------: |
| Positive  |      9000 |
| Negative  |      1000 |
| **Total** | **10000** |

This creates a distribution of **90% positive reviews and 10% negative reviews**.

The purpose is to simulate a classification problem where one class has significantly more representation than the other.

---

## 2.1. Combine the Reviews

Both categories are used to build a single `DataFrame`:

```python
df_review_des = pd.concat([df_postive, df_negative])
```

This produces the imbalanced dataset that will be used later.

---

## 2.2. Check the Imbalanced Dataset

To verify the distribution:

```python
data_count_des = df_review_des.value_counts('sentiment')

print(data_count_des)
```

This allows us to verify that there are:

```text
positive    9000

negative    1000
```

---

# 3. Balance the Dataset

Due to the imbalance created previously, the following technique is used:

```python
RandomUnderSampler
```

This technique belongs to the **imbalanced-learn** library.

```python
rus = RandomUnderSampler()

df_review_bal, df_review_bal['sentiment'] = rus.fit_resample(
    df_review_des[['review']], 
    df_review_des['sentiment']
)
```

### What Does RandomUnderSampler Do?

The **Random Under-Sampling** technique randomly reduces the number of examples from the majority class to match the number of examples in the minority class.

In this case:

```text
Before:

Positive → 9000

Negative → 1000

After:

Positive → 1000

Negative → 1000
```

The result is a balanced dataset containing:

```text
2000 reviews
```

with an equal distribution between both categories.

---

# 4. Split the Data

After balancing the dataset, it is divided into two groups:

* **Training Set:** data used to train the models.
* **Test Set:** data used to evaluate the models.

The following is used:

```python
train, test = train_test_split(
    df_review_bal,
    test_size=0.33,
    random_state=42
)
```

### Parameters

`test_size=0.33` means that approximately **33% of the data** is used for testing.

The remaining `67%` is used for training.

`random_state=42` allows the split to be reproducible.

The features and labels are then separated:

```python
train_x, train_y = train['review'], train['sentiment']

test_x, test_y = test['review'], test['sentiment']
```

Where:

* `train_x` → training reviews.
* `train_y` → training sentiments.
* `test_x` → test reviews.
* `test_y` → actual test sentiments.

---

# 5. Text Transformation with TF-IDF

Machine Learning algorithms cannot work directly with text.

For this reason, the following is used:

```python
TfidfVectorizer
```

```python
tfidf = TfidfVectorizer(stop_words='english')
```

TF-IDF converts words from the reviews into numerical representations.

### What Does TF-IDF Mean?

**TF-IDF** stands for:

> Term Frequency - Inverse Document Frequency

Its purpose is to assign a weight to words based on how important they are within the documents.

Frequent but less informative words receive lower importance, while more representative terms can receive higher weights.

---

## Vectorizer Training

```python
train_x_vector = tfidf.fit_transform(train_x)
```

Here, the vectorizer learns the vocabulary using only the training data and transforms the reviews into numerical vectors.

For the test data:

```python
test_x_vector = tfidf.transform(test_x)
```

The same vocabulary learned during training is used.

This prevents information from the test set from influencing the training process.

---

# 6. Support Vector Machine (SVM)

The first model used is **Support Vector Machine**.

```python
svc = SVC(kernel="linear")

svc.fit(train_x_vector, train_y)
```

A linear kernel is used because it is suitable for working with high-dimensional text representations.

The model learns to separate positive and negative reviews using a decision boundary.

A prediction can also be made using a custom review:

```python
svc.predict(
    tfidf.transform(["i don't like this movie"])
)
```

This allows the model to be tested with text that does not belong to the original dataset.

---

# 7. Decision Tree

The second model used is **Decision Tree**.

```python
des_tree = DecisionTreeClassifier()

des_tree.fit(train_x_vector, train_y)
```

A decision tree builds a decision structure based on the available features to determine the corresponding class.

In this project, the TF-IDF representation is used as input for the model.

---

# 8. Gaussian Naive Bayes

The third algorithm used is **Gaussian Naive Bayes**.

```python
gnb = GaussianNB()

gnb.fit(train_x_vector.toarray(), train_y)
```

Unlike some of the other models used, `GaussianNB` requires working with a dense matrix.

For this reason, the following is used:

```python
train_x_vector.toarray()
```

to convert the sparse matrix generated by TF-IDF into a conventional matrix.

---

# 9. Logistic Regression

The fourth model used is **Logistic Regression**.

```python
lr = LogisticRegression()

lr.fit(train_x_vector, train_y)
```

Logistic Regression is an algorithm widely used for binary classification problems.

In this project, it is used to determine whether a review belongs to:

```text
positive
```

or:

```text
negative
```

---

# 10. Model Evaluation

Finally, the performance of each model is evaluated using:

```python
model.score(test_x_vector, test_y)
```

The `score()` method returns the model's **accuracy** on the test data.

Results obtained:

| Model                |   Accuracy |
| -------------------- | ---------: |
| SVM                  | **83.79%** |
| Decision Tree        | **71.67%** |
| Gaussian Naive Bayes | **60.76%** |
| Logistic Regression  | **84.55%** |

---

# Results Comparison

According to the results obtained, **Logistic Regression** achieves the best performance:

```text
Logistic Regression → 84.55%
```

followed by:

```text
SVM → 83.79%
```

Then:

```text
Decision Tree → 71.67%
```

and finally:

```text
Gaussian Naive Bayes → 60.76%
```

Therefore, within this experiment, **Logistic Regression was the model with the highest accuracy** for classifying movie reviews.

However, the difference between Logistic Regression and SVM is small, so both models show good performance for this problem.
