# Coursera-PracticalMachineLearning - Week 4 Project Assignment

**Note: The full-report is in the compiled HTML file which is available in this repository. This HTML report has been created using Knitr.**

## Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. 
One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.
In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.
More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## DataSet
The training data for this project are available here:  
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:  
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.  
If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

## What you should submit
The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. 
You may use any of the other variables to predict with. 
You should create a report (i.e. this file) describing:
* how you built your model
* how you used cross validation
* what you think the expected out of sample error is
* and why you made the choices you did
You will also use your prediction model to predict 20 different test cases.

## Peer Review Portion
Your submission for the Peer Review portion should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders.

## Reproducibility
Due to security concerns with the exchange of R code, your code will not be run during the evaluation by your classmates. 
Please be sure that if they download the repo, they will be able to view the compiled HTML version of your analysis.

## How I built the model, how I used cross-validation, accuracy and errors, and choices I made

### STEP 1 - Loading the R Packages
library(caret)  
library(rpart)  
library(rpart.plot)  
library(RColorBrewer)  
library(RGtk2)  
library(rattle)  
library(randomForest)  

### STEP 2 - Loading the Dataset
Download the data files from the Internet and load them into two data frames. 
We ended up with a training dataset and a 20 observations testing dataset that will be submitted to Coursera.

UrlTrain <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"  
UrlTest  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"  

setwd("C:/Users/Philippe/Documents/DataCourse")

dt_training <- read.csv(url(UrlTrain))  
**19622 obs. of 160 variables**  
dt_testing  <- read.csv(url(UrlTest))  
**20 obs. of 160 variables**  

### STEP 3 - Cleaning the Data
Remove all columns that contains NA and remove features that are not in the testing dataset. 
The features containing NA are the variance, mean and standard devition (SD) within each window for each feature. 
Since the testing dataset has no time-dependence, these values are useless and can be disregarded. 
Also remove the first 7 features as they are related to the time-series or are not numeric.

features <- names(dt_testing[,colSums(is.na(dt_testing)) == 0])[8:59]  
Only use features used in testing cases:  
dt_training <- dt_training[,c(features,"classe")]  
dt_testing <- dt_testing[,c(features,"problem_id")]  

dim(dt_training)  
**19622 obs. of 53 variables**  
dim(dt_testing)  
**20 obs. of 53 variables**  

### STEP 4 - Partitioning the Dataset
Setting the seed for reproducibility  
set.seed(5656)  

The objective is to get training, cross-validation, and testing sets.
I split the training data into a training data set (60% of the total cases) and a testing data set - or cross-validation (40% of the total cases). 
It will help estimate the out of sample error.

inTrain <- createDataPartition(dt_training$classe, p=0.6, list=FALSE)  
training <- dt_training[inTrain,]  
testing <- dt_training[-inTrain,]  

dim(training)  
**11776 obs. of 53 variables**  
dim(testing)  
**7846 obs. of 53 variables**  

### STEP 5 - Building and comparing the Models

#### 1st Model - Decision Tree
First, I will try using Decision Tree. The accuracy may not be high.

Build the model:  
modFitDT <- rpart(classe ~ ., data = training, method="class")  
fancyRpartPlot(modFitDT)  
** Please look at "Week4 Project - Decision Tree Model - Rplot.png" file for the result**  

Prediction:  
set.seed(5656)  

prediction <- predict(modFitDT, testing, type = "class")  
confusionMatrix(prediction, testing$classe)  

**Accuracy of the Decision Tree Model is only 72%.**  
**I will try the Random Forest Model with which the accuracy should be much better.**  

#### 2nd Model - Random Forest
Using random forest, the out of sample error should be small. The error will be estimated using the 40% testing sample.

Build the model:  
set.seed(5656)  
modFitRF <- randomForest(classe ~ ., data = training, ntree = 1000)  

Prediction:  
prediction <- predict(modFitRF, testing, type = "class")  

confusionMatrix(prediction, testing$classe)

**As can be seen from the confusion matrix the Random Forest model is very accurate, about 99.3%.**  

### STEP 5 - Predicting on the Testing Data (pml-testing.csv)

#### Decision Tree Prediction
predictionDT <- predict(modFitDT, dt_testing, type = "class")  
predictionDT  

Results:  
1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20  
C  A  E  D  A  C  D  D  A  A  C  E  A  A  E  E  A  B  B  B  
Levels: A B C D E  

#### Random Forest Prediction
predictionRF <- predict(modFitRF, dt_testing, type = "class")  
predictionRF  
1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20   
**B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B**  
**Levels: A B C D E **  

# END
