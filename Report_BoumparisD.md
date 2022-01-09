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
* __Removing stopwords__: This is done to avoid feeding the algorithms with high-frequency words that do not carry significant meaning (e.g. 'the', 'of', 'at', etc.).

Vectorization, which usually comes after tokenization, is not done in the preprocessing phase, as I used different methods in the two types of algorithms.
