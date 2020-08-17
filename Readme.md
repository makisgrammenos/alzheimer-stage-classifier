# Alzheimer Stage Classifier

This is my first attempt creating  Convolutional Neural Networks.I created
a CNN to predict  if a patient has Alzheimer's Disease   and to classify the current Alzheimer stage based on patient's brain MRI scan
The CNN has 93% accuracy 

## Stages for classification
The  neural network classifies a patient's brain MRI scan into the following categories
* Non   Demented
* Very Mild Demented
* Mild Demented
* Moderate Demented 

## Dataset
The dataset used can me found [here](https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images). I have merged
the train and test directories found in the dataset , and split them  using **sklearn.modelselection.train_test_split** to  achieve better results in the training process.
##Before you start
Before you start playing with the model run in the repo directory the following command to install the required packages 
for the model to run
```shell script
$ pip install -r requirments.txt
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
The script will take care of the  prediction process
