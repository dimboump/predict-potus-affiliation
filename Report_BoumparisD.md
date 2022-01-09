# Predicting POTUS Affiliation with Scikit-Learn and TensorFlow

## 1 Introduction

### 1.1 Overview

The report at hand aims at briefly outlining the main differences found in the results of 6 different algorithm implementations on textual data. The objective of these experiments was to analyze how 3 Machine Learning and 3 Deep Learning algorithms perform on the text classification task by using the Scikit-Learn and TensorFlow Keras libraries respectively. The task at hand is to train these algorithms to predict whether a speech (in written form) belongs to a Democratic or Republican President of the United States (POTUS).

The algorithms used are:

* Machine Learning (`Scikit-Learn`):
  * Logistic Regression
  * Support Vector Machines
  * k Nearest Neighbors
* Deep Learning (`TensorFlow Keras`):
  * Dense Neural Network
  * Simple Recurrent Neural Network (RNN)
  * Long-Short Term Memory (LSTM) Neural Network

### 1.2 Dataset

#### 1.2.1 Origin and Description

The data used for both training and testing are freely available via the [Miller Center](https://millercenter.org/the-presidency/presidential-speeches) website. Unfortunately, however, they do not provide a dataset in `CSV` form. Joseph Lilleberg scraped the data from the website and made them available in [`csv` on Kaggle](https://www.kaggle.com/littleotter/united-states-presidential-speeches). This dataset includes approximately 1000 famous speeches delivered by every POTUS since the days of George Washington until September 2020.

The dataset is structured as such:

| Field name      | Description                                        |
|---------------- |--------------------------------------------------- |
| `Date`          | The date the speech was delivered on (YYYY-MM-DD)  |
| `President`     | The POTUS who delivered the speech                 |
| `Party`         | The POTUS's political affiliation                  |
| `Speech Title`  | A unique title for the speech                      |
| `Summary`       | A brief description of the speech's topic          |
| `Transcript`    | The full text of the speech                        |
| `URL`           | The webpage the speech can be found on             |

In the classification task, I only used the `Party` and `Transcript` columns.

When I downloaded the `csv` file, no speeches delivered by President Biden were included. I had, therefore, to append them to the dataset. The next sub-section sheds a light on how I dealt with this.

#### 1.2.2 Web Scraping Biden's Speeches

Luckily for me, the Miller Center's website provides all the basic info that I could include in the dataset. The key is to automate the process, instead of manually adding the info of the missing speeches.

Given the limited space available in this Report, please take a look at how I went about doing this in the [`update_dataset_biden` notebook](https://github.com/user/repo/blob/dev/update_dataset_biden.ipynb).

#### 1.2.3 Exploratory Data Analysis (EDA)

Before we go ahead and start preprocessing our data (let alone building our models), we should always get a grasp of how the data is structured and briefly describe its main characteristics.

The dataset includes every political affiliation that a President was elected in. That means, `unaffiliated`, `Federalist`, and `Whig` are also included. Initially, I though this could make a great multilabel classification task possible, but `speeches_all['Party'].value_counts()` showed that there is not enough data other than the two main parties:

| Party                  | Speeches  |
|----------------------- |---------: |
| Democratic             |      489  |
| Republican             |      389  |
| Democratic-Republican  |       65  |
| Unaffiliated           |       39  |
| Whig                   |       12  |
| Federalist             |        9  |
| *Total*                |  *1,003*  |

With that being said, I went ahead and dropped the speeches that were delivered by Presidents other that Democratic or Republican. Then, I concatenated the two sub-`DataFrames` into one. The task is now, therefore, a binary classficiation task.

#### 1.2.4 Preprocessing

Preprocessing is maybe the most tedious part of (Text) Data Analysis yet the most essential one. Without proper preprocessing, the models will not be able to perform to their best.

In this project, preprocessing included:

* __Removing punctuation__: Use of punctuation does not provide the algorithm with meaningful data, so we may exclude it from the data.
* __Lowercasing and tokenizing text__: This step is important so casing does not result in idential words being treated as different ones.
* __Removing stopwords__: This is done to avoid feeding the algorithms with high-frequency words that do not carry significant meaning (e.g. 'the', 'of', 'at', etc.). The [Natural Language ToolKit (NLTK)](https://www.nltk.org/) provides us with a dictionary of stopwords in English, among other languages.
* __Splitting the data__: We need to split our dataset into two subsets: training and testing. This way, we use the former to train the algorithms and the latter to test it on and evaluate its performance. The `train_test_split` function found in Scikit-Learn makes sure our data splitting is done randomly.

Vectorization, which usually comes after tokenization, is not done in the preprocessing phase, as I used different methods for each of the two types of algorithms.

## 2 Algorithms: Implementation and Evaluation

After having preprocessed the dataset, it is now time to start implementing the algorithms. In this section, we will briefly discuss the best hyperparapemeters used for each one along with its best scores.

### 2.1 Classical Algorithms

In all cases, the methodology includes:

1. Make a pipeline with the algorithm of choice.
2. Run a `GridSearchCV` with different parameters so the best model is selected.
3. Train the data on the model with the best performing parameters.
4. Test the model on test data.
5. Run a 10-fold Cross Validation and plot the Confusion Matrix.

#### 2.1.1 Logistic Regression

The Logistic Regression algorithms performed relatively high, achieving an accuracy of 77.8%. Following a `GridSearchCV`, the best hyperparameters were:

* `C`: 100
* `Solver`: liblinear

The model did not achieve excellent precision or recall in either classes, although it did perform better on Republican speeches (positive class):

![Logistic Regression Confusion Matrix](img/classical__logreg_confusion_matrix.png)

#### 2.1.2 Support Vector Machines (SVM)

SVM is one of the most used algorithms to be used for, but limited to, text data. In my case, in particular, the model scored an accuracy of 79.5% -- slightly better than the previous one. Following a `GridSearchCV`, the best hyperparameters were:

* `C`: 10
* `Solver`: linear

It is fairly obvious than the following confusion matrix is almost identical to the previous one, given the marginal difference in performance:

![Support Vector Machines Confusion Matrix](img/classical__svm_confusion_matrix.png)

However, we should point out that the `C` hyperparameter dropped to 10 which means that the Regularization in this case is stronger. This means that we should be aware of possible overfitting on our data -- at least compared to Logistic Regression.

#### 2.1.3 k Nearest Neighbors

K Nearest Neighbors turned out to perform equally as well with the SVM model (79.5% accuracy). This was definitely a head-scratcher, although the Classification Report shows different values:

SVM:
|               | precision  | recall  | f1-score  | support  |
|-------------: |----------: |-------: |---------: |--------: |
|            0  |      0.81  |   0.81  |     0.81  |      94  |
|            1  |      0.78  |   0.78  |     0.78  |      82  |
|     accuracy  |            |         |     0.80  |     176  |
|    macro avg  |      0.79  |   0.79  |     0.79  |     176  |
| weighted avg  |      0.80  |   0.80  |     0.80  |     176  |

KNN:
|               | precision  | recall  | f1-score  | support  |
|-------------: |----------: |-------: |---------: |--------: |
|            0  |      0.78  |   0.86  |     0.82  |      94  |
|            1  |      0.82  |   0.72  |     0.77  |      82  |
|     accuracy  |            |         |     0.80  |     176  |
|    macro avg  |      0.80  |   0.79  |     0.79  |     176  |
| weighted avg  |      0.80  |   0.80  |     0.79  |     176  |

As far as the kNN model is concerned, the best estimator is `n_neighbors = 3`.

