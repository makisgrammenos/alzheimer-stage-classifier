# Alzheimer Stage Classifier

This is my first attempt creating  Convolutional Neural Networks.I created
a CNN to predict  if a patient has Alzheimer's Disease   and to classify the current Alzheimer stage based on patient's brain MRI scan
The CNN has approximately 95% accuracy 

## Stages for classification
The  neural network classifies a patient's brain MRI scan into the following categories
* Non   Demented
* Very Mild Demented
* Mild Demented
* Moderate Demented 

## Dataset
The dataset used can me found [here](https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images). I have merged
the train and test directories found in the dataset , and split them  using **sklearn.modelselection.train_test_split** to  achieve better results in the training process.
## Before you start
Before you start playing with the model run in the repo directory the following command to install the required packages 
for the model to run
```shell script
$ pip install -r requirments.txt
```
## Model Architecture 
Convolutional Neural Network Architecture: 
```shell script
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 118, 118, 64)      640       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 59, 59, 64)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 57, 57, 64)        36928     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 28, 28, 64)        0         
_________________________________________________________________
flatten (Flatten)            (None, 50176)             0         
_________________________________________________________________
dense (Dense)                (None, 128)               6422656   
_________________________________________________________________
dropout (Dropout)            (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 4)                 516       
=================================================================
Total params: 6,460,740
Trainable params: 6,460,740
Non-trainable params: 0
_________________________________________________________________

```
## Training the model
To train the model  all you have to do is to run :
```
$ python train.py
```
Make sure the data folder which contains the training data  has not been altered in anyway
  
The model will be saved in the **model**  directory with name "**model.h5**" overwriting the current pre-trained model.
### Training Statistics
##### Model Accuracy 
![accuracy](/images/accuracy.png)

##### Model Loss

![loss](/images/loss.png)


## Using the model for making predictions
To use the model for making predictions first put  brain **MRI** scans in the **test directory**
  
After, run :
```shell script
$ python predict.py 
```
The script will load all the photos located in the test folder and will try to predict the Alzheimer stage based on the
MRI scan

## Updates and Feedback

I am looking forward to get your feedback on any issues that may occur.
A new update is coming soon to improve the model's accuracy

# License 
All rights reserved.