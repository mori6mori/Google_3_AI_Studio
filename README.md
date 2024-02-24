# Google_3_AI_Studio
Link to Presentation: https://docs.google.com/presentation/d/1XEkQF_JJZUHaHZxSEN8kaCMFxoQvHbh8mOEFUugdR8Y/edit#slide=id.g18a07cd33b2_0_82

Project Goal: 
Using the week long data provided by criteo to predict ad click-through rate (CTR) with Neural Network Algorithm. 
In this project, given the user and the page that they’re visiting, we are going to predict the probability of them clicking on display advertisement.

# Data Processing:
The dataset is from a Kaggle challenge for Criteo display advertising. 
Contains about one week of click-through data.
Begins with the Label Column
Proceeded by 13 Numeric Feature Columns (I1- I13) →  use directly
Ending in 26 Categorical Feature Columns (C1-C26) → apply embedding & represent with dense vector
Label is binary, either 0 or 1. 

# Dataset Split:

Sample command to split files:
split -a 2 -d -l 330034 train.csv train-
train.csv: 33003327
valid.csv: 8250125
test.csv : 4587168

Splits the overly large training data set into 100 smaller, more manageable files

Training data is the data used to train the machine learning model

Validation data is used to fine tune the model during the model training phase

Testing data is used to test the final version of the model

# Hyperparameter

number of HIDDEN LAYERs / number of  hidden units

Located between the input and output of the algorithm, in which the function applies weights to the inputs and directs them through an activation function as the output
- Adding more hidden layers increases accuracy
-But could also lead to overfitting, model becomes more expensive

Learning Rate

The steps that the model must take at each iteration, the speed a model must learn at.
-Increased learning rate leads to faster model convergence 
-But also leads to overshooting.  As a result of large learning rates the model diverges
-Undershooting as a result of small learning rate




