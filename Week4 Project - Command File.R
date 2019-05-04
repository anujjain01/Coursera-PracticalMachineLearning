# Coursera Course - Practical Machine Learning
# Week 4 Project
# By PhilAIUK
# May 4, 2019

# Loading the R Packages
# ----------------------
library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(RGtk2)
library(rattle)
library(randomForest)

# Loading the Dataset
# -------------------
# Download the data files from the Internet and load them into two data frames

UrlTrain <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
UrlTest  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

setwd("C:/Users/Philippe/Documents/DataCourse")

dt_training <- read.csv(url(UrlTrain))  # 19622 obs. of 160 variables
dt_testing  <- read.csv(url(UrlTest))   # 20 obs. of 160 variables

# Cleaning the Data
# -----------------
# Remove all columns that contains NA and remove features that are not in the testing dataset. 
# The features containing NA are the variance, mean and standard devition (SD) within each window for each feature. 
# Since the testing dataset has no time-dependence, these values are useless and can be disregarded. 
# Also remove the first 7 features as they are related to the time-series or are not numeric.

features <- names(dt_testing[,colSums(is.na(dt_testing)) == 0])[8:59]

# Only use features used in testing cases
dt_training <- dt_training[,c(features,"classe")]
dt_testing <- dt_testing[,c(features,"problem_id")]

dim(dt_training)  # 19622 obs. of 53 variables
dim(dt_testing);  # 20 obs. of 53 variables

# Partitioning the Dataset
# ------------------------

# Setting the seed for reproducibility
set.seed(5656)  

# The objective is to get training, cross-validation, and testing sets.
# I split the training data into a training data set (60% of the total cases) and a testing data set - or cross-validation (40% of the total cases). 
# It will help estimate the out of sample error.

inTrain <- createDataPartition(dt_training$classe, p=0.6, list=FALSE)
training <- dt_training[inTrain,]
testing <- dt_training[-inTrain,]

dim(training)  # 11776 obs. of 53 variables
dim(testing)   # 7846 obs. of 53 variables

# Decision Tree Model
# -------------------
First, I will try using Decision Tree. The accuracy may not be high.

# Build the model
modFitDT <- rpart(classe ~ ., data = training, method="class")
fancyRpartPlot(modFitDT)
# Please look at plot1.png file for the result

# Prediction
set.seed(5656)

prediction <- predict(modFitDT, testing, type = "class")
confusionMatrix(prediction, testing$classe)

# Accuracy of the Decision Tree Model is only 74%.
# I will try the Random Forest Model with which the accuracy should be much better.

# Random Forest Model
# -------------------
# Using random forest, the out of sample error should be small. The error will be estimated using the 40% testing sample.

# Build the model:
set.seed(5656)
modFitRF <- randomForest(classe ~ ., data = training, ntree = 1000)

# Prediction
prediction <- predict(modFitRF, testing, type = "class")
confusionMatrix(prediction, testing$classe)

# As can be seen from the confusion matrix the Random Forest model is very accurate, about 99.3%.

# Predicting on the Testing Data (pml-testing.csv)
# ------------------------------
  
# Decision Tree Prediction
predictionDT <- predict(modFitDT, dt_testing, type = "class")
predictionDT
# 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
# C  A  E  D  A  C  D  D  A  A  C  E  A  A  E  E  A  B  B  B 
# Levels: A B C D E

# Random Forest Prediction
predictionRF <- predict(modFitRF, dt_testing, type = "class")
predictionRF
# 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
# B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
# Levels: A B C D E

# End
