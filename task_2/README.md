# Task 2 – Medical Events Prediction
The task description is in [task_description.pdf](task_description.pdf).

## Results
TBD


## Reproducability
TBD


## Report
The following report describes the approach that led to our solution in [main.ipynb](main.ipynb).

1)	Training and test data are loaded with pandas. Column names for the predictions are defined as specified in the submission format in [train_labels.csv](train_labels.csv) and partitioned into the corresponding subtasks 1, 2 and 3. The dataframe for the predictions is initialized.
2)	First, we focus on subtask 1 and 2 where we perform binary classification. 
3)	We specify all columns in X_train to be relevant except ‘pid’ and ‘Time’. Next, we do feature engineering. For each ‘pid’ and relevant column, we compute mean, min, max, difference between min and max, first available (i.e. not nan) observation, last available observation, difference between first and last and also the number of missing values over all 12 observations. Whenever there is no observation per ‘pid’ and relevant column, the mean of the entire dataset is used. Features are engineered in this way for the entire training set and for the entire test set.
4)	We create an ANN classifier using the Keras functional API. We use multiple outputs so that we can perform binary classification for every label in subtask 1+2 simultaneously and benefit from the correlations in the labels that our ANN captures. In order to find the correct hyperparameters we had to create a custom sklearn classifier so that Keras functional API can be used in the Pipeline and the GridSearchCV functions.
5)	We use StandardScaler from sklearn to scale the features and then 5-fold CV is performed using GridSearchCV with ‘roc_auc’ as our scoring function and using the best found parameters, we make predictions.
6)	Next, we focus on subtask 3 where we focus on regression.
7)	Again, we perform feature engineering using the same procedure as above. The only difference is that we specify less columns to be relevant.
8)	Here we try the models Ridge and RandomForestRegressor from sklearn and for each label separately we perform GridSearchCV using ‘r2’ as our scoring function. And with the best found parameters we make predictions.
9)	Finally, we save our predictions as a zip file using pandas.
