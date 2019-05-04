# Coursera-PracticalMachineLearning - Week 4 Project Assignment

**Note:** The full-report is available in this file, after the introduction of the project

The compiled HTML file is available here: https://philaiuk.github.io/Coursera-PracticalMachineLearning/

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

## Full Report describing how I built the model, how I used cross-validation, accuracy and errors, and choices I made

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

List of features remaining:  
 [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
 [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
 [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
[10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
[13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
[16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
[19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
[22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
[25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
[28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
[31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
[34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
[37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
[40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
[43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
[46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
[49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
[52] "magnet_forearm_z"    

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

**Results from Confusion Matrix and Statistics:**

Reference  
Prediction    A    B    C    D    E
         A 2026  252   45   81   46
         B   60  868   70   96  114
         C   43  187 1079  181  146
         D   70  111   83  820   86
         E   33  100   91  108 1050

Overall Statistics                           
               Accuracy : 0.7447          
                 95% CI : (0.7349, 0.7543)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16                              
                  Kappa : 0.6761                                
 Mcnemar's Test P-Value : < 2.2e-16       

Statistics by Class:  
                     Class: A Class: B Class: C Class: D Class: E  
Sensitivity            0.9077   0.5718   0.7887   0.6376   0.7282
Specificity            0.9245   0.9463   0.9140   0.9466   0.9482
Pos Pred Value         0.8269   0.7185   0.6595   0.7009   0.7598
Neg Pred Value         0.9618   0.9021   0.9535   0.9302   0.9394
Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
Detection Rate         0.2582   0.1106   0.1375   0.1045   0.1338
Detection Prevalence   0.3123   0.1540   0.2085   0.1491   0.1761
Balanced Accuracy      0.9161   0.7590   0.8514   0.7921   0.8382

**Accuracy of the Decision Tree Model is only 74%.**  
**I will try the Random Forest Model with which the accuracy should be much better.**  

#### 2nd Model - Random Forest
Using random forest, the out of sample error should be small. The error will be estimated using the 40% testing sample.

Build the model:  
set.seed(5656)  
modFitRF <- randomForest(classe ~ ., data = training, ntree = 1000)  

Prediction:  
prediction <- predict(modFitRF, testing, type = "class")  

confusionMatrix(prediction, testing$classe)

**Results from Confusion Matrix and Statistics:**  
Reference  
Prediction    A    B    C    D    E
         A 2228    4    0    0    0
         B    4 1507   13    0    0
         C    0    7 1355   23    1
         D    0    0    0 1263    3
         E    0    0    0    0 1438

Overall Statistics                                         
               Accuracy : 0.993           
                 95% CI : (0.9909, 0.9947)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16                                      
                  Kappa : 0.9911                                       
 Mcnemar's Test P-Value : NA              

Statistics by Class:  
                     Class: A Class: B Class: C Class: D Class: E  
Sensitivity            0.9982   0.9928   0.9905   0.9821   0.9972
Specificity            0.9993   0.9973   0.9952   0.9995   1.0000
Pos Pred Value         0.9982   0.9888   0.9776   0.9976   1.0000
Neg Pred Value         0.9993   0.9983   0.9980   0.9965   0.9994
Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
Detection Rate         0.2840   0.1921   0.1727   0.1610   0.1833
Detection Prevalence   0.2845   0.1942   0.1767   0.1614   0.1833
Balanced Accuracy      0.9987   0.9950   0.9929   0.9908   0.9986

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
