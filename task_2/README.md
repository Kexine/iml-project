# Task 2 – Medical Events Prediction
The task description is in [task_description.pdf](task_description.pdf).

## Results
Our model achieved a score of 0.7795 on the public part of the test set and 0.7745 on the private part. The predictions are in [submission.zip](submission.zip).

The hard baseline was at a score of 0.772.

## Reproducability
The following notebook was used: [main.ipynb](main.ipynb). In order to reproduce the results you need to open the notebook in Google Colab and select 'GPU' as runtime type. Then, under 'Runtime', click on 'Run all'.

## Report
The following report describes the approach that led to our solution in [main.ipynb](main.ipynb).

1)	Training and test data are loaded with pandas. Column names for the predictions are defined as specified in the submission format in [train_labels.csv](train_labels.csv) and partitioned into the corresponding subtasks 1, 2 and 3. The dataframe that will contain the predictions on the test set is initialized.
2)	First, we focus on subtask 1 and 2 where we perform binary classification. 
3)	We specify all columns in X_train to be relevant except for 'pid' and 'Time'. In order to reduce the number of features and get a better score we do feature engineering. Previously, we simply concatenated all 12 observations for each 'pid' which resulted in 421 features and a worse score. We found feature engineering to be crucial for this task. For each 'pid' and relevant column, we compute the following features: mean, min, max, difference between min and max, first available (i.e. not nan) observation, last available observation, difference between first and last, the number of missing values over all 12 observations. Whenever there is no observation per 'pid' and relevant column, we impute the value with the mean of that column over the entire dataset. Features are engineered in this way for the entire training set and for the entire test set. In this way, we reduce the number of features to 273.
4)	We create an artifical neural network (ANN) classifier using the Keras functional API. We use multiple outputs so that we can perform binary classification for every label in subtask 1+2 simultaneously and benefit from the correlations in the labels that our ANN captures. In order to find the correct hyperparameters we create a custom sklearn classifier so that the Keras functional API can be used in the Pipeline and the GridSearchCV functions from scikit-learn. Please note that we previously performed a more extensive hyperparameter search than specified in [main.ipynb](main.ipynb) using ETH Zurich's Euler cluster with 48 CPU cores.
5)  We perform cost-sensitive classification because our dataset is highly imbalanced.
6)	We use StandardScaler from sklearn to scale the features and then 5-fold CV is performed using GridSearchCV with 'roc_auc' as our scoring function and using the best found parameters, we make predictions.
7)	Next, we focus on subtask 3 where we perform regression.
8)	Again, we perform feature engineering using the same procedure as above. The only difference is that we specify less columns to be relevant.
9)	Here we try the models Ridge and RandomForestRegressor from sklearn and for each label separately we perform GridSearchCV using 'r2' as our scoring function and with the best found parameters we make predictions.
10)	Finally, we save our predictions as a zip file using pandas.
