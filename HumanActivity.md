Prediction of How Well Human Activity is Done
---------------------------------------------

Objective
---------

Objectibe of this report is to predict from Weight Lifting Exercises dataset the "manner" in which one did the exercise (Weight Lifting). an activity was performed by the wearer. From the [provider of the data](http://groupware.les.inf.puc-rio.br/har) it is clear that there are five **manners** in which one can perform the wet lifting Exercise. They are - exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

**Objective of this project is to build a model based on the Exercises Dataset which can fairly accurately predict the *class* of an observation**.

Required Libraries
------------------

Required libraries for performing the Data analysis is loaded at the outset.

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    ## 
    ## Attaching package: 'kernlab'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     alpha

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

Data
----

Data set is available at the website and needs to be loaded into the system for doing further processing.

``` r
trainingnurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"

testurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"


download.file(trainingnurl, destfile = "trainingTmp.csv",method = "libcurl")
download.file(testurl, destfile = "testTmp.csv",,method = "libcurl")

## While loading data normalize  "#DIV/0!" , "" or "NA"  to NA.
trainingSet <- read.csv('trainingTmp.csv',header=TRUE, na.strings=c("NA","#DIV/0!", ""))
testSet <- read.csv('testTmp.csv',header=TRUE, na.strings=c("NA","#DIV/0!", ""))
dim(trainingSet)
```

    ## [1] 19622   160

``` r
dim(testSet)
```

    ## [1]  20 160

``` r
head(trainingSet [1:6])
```

    ##   X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
    ## 1 1  carlitos           1323084231               788290 05/12/2011 11:23
    ## 2 2  carlitos           1323084231               808298 05/12/2011 11:23
    ## 3 3  carlitos           1323084231               820366 05/12/2011 11:23
    ## 4 4  carlitos           1323084232               120339 05/12/2011 11:23
    ## 5 5  carlitos           1323084232               196328 05/12/2011 11:23
    ## 6 6  carlitos           1323084232               304277 05/12/2011 11:23
    ##   new_window
    ## 1         no
    ## 2         no
    ## 3         no
    ## 4         no
    ## 5         no
    ## 6         no

``` r
table(trainingSet$classe)
```

    ## 
    ##    A    B    C    D    E 
    ## 5580 3797 3422 3216 3607

Training seat has 19622 observations with features. It is also clear from the priliminary investigation that training data set five classes of data.

``` r
#colnames(trainingSet)
##Following columns seem not useful for our project
  ##[1] "X"                        "user_name"               
  ##[3] "raw_timestamp_part_1"     "raw_timestamp_part_2"    
  ##[5] "cvtd_timestamp"           "new_window"              
  ##[7] "num_window"

trainingSet<-trainingSet[,-c(1:7)]
 
## Remove columns with more than 40% missing values *Arbitary asuumption from experience


  noRows<-dim(trainingSet)[1]
  NAPrct<-colSums(is.na(trainingSet))/noRows
  goodCols<-NAPrct<.4
  
  trainingSet<-trainingSet[,goodCols]

# remove  near zero variance features
nearZeroVrnce <- nearZeroVar(trainingSet, saveMetrics=TRUE)
trainingSet <- trainingSet[, !nearZeroVrnce$nzv]

## Preprocessing is dome for same columns of test data
testSet<-testSet[,-c(1:7)] 
testSet<-testSet[,goodCols]
testSet<-testSet[, !nearZeroVrnce$nzv]


dim(trainingSet)
```

    ## [1] 19622    53

``` r
dim(testSet)
```

    ## [1] 20 53

``` r
##Set seed for reproducability
set.seed(123)
trnrows <- createDataPartition(trainingSet$classe, p=0.60, list=FALSE)
trainingSet <- trainingSet[trnrows,]
validationSet <- trainingSet[-trnrows,]
```

Data Cleaning & Preprocessing
-----------------------------

colnames(trainingSet) shows many columns which from the names can be concluded as less important for our project. They are removed from the datasets. Next Columns with all missing values are removed. Next preprocessing step is using nearZeroVar function of caret package to find those columns from the data for which variance is near to zero(or zero).

So we can reduce the dimension of the data by removing those columns for which varaince is zero, becoz zero variance columns have unique values. So those column doesn't impact the output at all.

We have a large sample size in the Training data set. This allow us to divide our Training sample to allow cross-validation.In order to perform cross-validation, the training data set is partionned into 2 sets: Training (60%) and ValidationSet (40%). For reproducability seed is set to 123.

Data Modelling
--------------

Random Forest is chosen to model the training dat because it does not expect linear features or even features that interact linearly and is very easy to tune. Other reasons for chossing Random Forest is it handles very well high dimensional spaces as well as large number of training examples.Its advantage over selecting Decision Tree is that it is less prone to overfitting.

ability to handle large number of features, especially when the interactions between variables are unknown, flexibility to use unscaled variables and categorical variables, which reduces the need for cleaning and transforming variables, immunity from overfitting and noise, and insensitivity to correlation among the features, Random Forest is chosen to model the training data.

``` r
##For large data sets, especially those with large number of variables, calling randomForest via the formula interface is not advised: There may be too much overhead in handling the formula.
modelExcs <- randomForest(classe~.,data=trainingSet)
```

Validation
----------

Model performance is measured against Training and Cross Validation set.

``` r
##Training set accuracy
predtrain <- predict(modelExcs, trainingSet)
print(confusionMatrix(predtrain, trainingSet$classe))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 3348    0    0    0    0
    ##          B    0 2279    0    0    0
    ##          C    0    0 2054    0    0
    ##          D    0    0    0 1930    0
    ##          E    0    0    0    0 2165
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9997, 1)
    ##     No Information Rate : 0.2843     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
    ## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

``` r
##Validation set accuracy

prdvalidation <- predict(modelExcs, validationSet)
print(confusionMatrix(prdvalidation, validationSet$classe))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1344    0    0    0    0
    ##          B    0  908    0    0    0
    ##          C    0    0  836    0    0
    ##          D    0    0    0  752    0
    ##          E    0    0    0    0  870
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9992, 1)
    ##     No Information Rate : 0.2854     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Prevalence             0.2854   0.1928   0.1775   0.1597   0.1847
    ## Detection Rate         0.2854   0.1928   0.1775   0.1597   0.1847
    ## Detection Prevalence   0.2854   0.1928   0.1775   0.1597   0.1847
    ## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

Overall accuracy of the Model is excellent and calculated at just over 99% with a p-value of 2 x 10^-16, or 0.00000000000000022. Our classifier seems to be doing a pretty reasonable job of classifying items. Next, as a a double check the out of sample error was calculated using the cross validation set. The result shows excellent performance on the validation set as well. With an accuracy rate above 99% on our cross-validation data, we can expect that there is a very high chance none of the test samples will be missclassified with the Model.

Prediction
----------

Prediction is done on Test Data set using the Model developed in the previous step.

``` r
prdctn <- predict(modelExcs, testSet)
prdctn
```

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E

Conclussion
-----------

This report depicts how the model for this classification problem is developed step by step. In the begining after all required libraries are loadded Data was retrieved from the source site. Data is then cleaned and features not required are removed. A PCA would have been appropriate but dure to time contraint it could not be done. In order to cross validate the model, training data is divided (60:40) into Training Set and Validation Set. Random forest is selected as model because of the accuracy, immunity from overfitting and noise as compared to Decision Tree, insensitivity to correlation among the features and flexibility to use unscaled and categorical variables, which reduces the need for cleaning and transforming variables. The confusion matrix created gives an accuracy of 99.9%. This model achieved and excellent accuracy on the validation set as well. The expected out-of-sample error is estimated at 0.001, or 0.1%. The expected out-of-sample error is calculated as 1 - accuracy for predictions made against the cross-validation set. At the end, a predicton is made on the 20 test cases.
