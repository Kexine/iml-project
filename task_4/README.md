# Task 4

The task description is in [task_description.pdf](task_description.pdf).

## Results

Our model achieved an accuracy of 69.86% on the public part of the test set and 69.71% on the private part. The predictions are in [submission.txt](submission.txt). 

The hard baseline was at 68.8% accuracy.


## Reproducability
In order to reproduce the results you need to use Google Colab and select 'TPU' as runtime type. Then, under 'Runtime', click on 'Run all'. 

Some modifications to the original code have been made to ensure easy and fast reproducability. The modifications are clearly marked with comments and mainly deal with the following:
* The needed data is loaded from Google Cloud Storage buckets that are publicly accessible and not from Google Drive.
* The part where images are written to TFRecord files and uploaded to a GCS bucket is not executed as it requires granting access to the Google Cloud Platform. In addition, it takes a long time. (See point 3 in the report below.)
* The submission files containing the predictions for the test set that were saved onto my Google Drive are being saved onto the local disk.


## Report

The following report describes the approach that led to our solution in [main.ipynb](main.ipynb).

1)  We use Google Colab so that we have access to a TPU. Our Google Drive is mounted onto Colab and the necessary files are copied to the local disk.
2)	Images, test and train triplets are loaded. Train triplets are labeled with 1 and the switched train triplets, i.e. columns B and C switched, are labeled with 0 and then both are combined into a balanced training set X_train.
3)	Because we want to use a TPU, we write our test and training datasets into TFRecords. These TFRecords are copied into a Google Cloud Storage bucket so that they can later be accessed by the TPU during training and prediction.
4)	Connection to TPU is established.
5)	We write a _parse_function that can read and pre-process the content of the TFRecord files. The images in the training set are randomly augmented during training. Unfortunately, we couldn’t use the full training set due to a bug that we couldn’t resolve. Hence we only use part of the TFRecord files that belong to the training set.
6)	We create our model using TensorFlow and Keras. We use the Keras Functional API because we have three inputs. For all three inputs we use the VGG16 pre-trained model (other models didn’t perform as well) as a fixed feature extractor, i.e. we freeze the convolutional base with the pre-trained weights. We chose to do so because ImageNet contains lots of food images. We added global average pooling at the end of the convolutional base because it improved our score. After that we concatenate the output of the three VGG16 models and add two dense layers and in the end a dense layer with sigmoid activation because we are doing binary classification. Dropout is put inbetween dense layers to reduce overfitting. The loss is binary cross entropy and we use Adam as an optimizer.
7)	Because we didn’t want to waste a large part of the training set by creating a disjoint training and validation set, we just saved the weights of the model after every epoch during training. We only used 15 epochs because we saw that our accuracy increased quite fast, i.e. surpassed the top accuracy scores on the public leaderboard, which is a sign of overfitting. (By disjoint, we mean that images that occur in triplets of the training set cannot occur in triplets of the validation set. As a consequence of this disjoint partition, a large part of the training set cannot be used.)
8)	After training, we made predictions on the test set with the model weights that we saved after every epoch. We saved all the predictions onto Google Drive. Because we didn’t use a validation set, we had to guess when our model began to overfit. After uploading some of our predictions to the project server, we found that our best score was with the predictions made with the weights after epoch 8.
9)	Because we didn’t want to risk our submission limit by guessing when the model was overfitting, we tried out different model architectures beforehand where we used a disjoint validation set. And only with the best found architecture we trained on the (almost) entire training dataset.
10)	In the end the Google Cloud Storage bucket and its content are deleted and the drive is flushed and unmounted.

