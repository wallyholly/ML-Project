# Weight Lifting Exercise Dataset

## Download, read and preprocessing the data

```r
library(caret, quietly=TRUE)
```

```
## Warning: package 'caret' was built under R version 3.1.3
```

```
## Warning: package 'ggplot2' was built under R version 3.1.3
```

```r
# set URLs:
url_train <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
url_test <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'

# download training and test data 
#download.file(url = url_train, destfile = 'data_train.csv')
#download.file(url = url_test, destfile = 'data_test.csv')

# read in training and test data
pml_train <- read.csv(file = 'data_train.csv',
                      na.strings = c('NA','#DIV/0!',''))
pml_test <- read.csv(file = 'data_test.csv',
                     na.strings = c('NA','#DIV/0!',''))
```

## Split of training data
I split the training data in a training and a validation set in order to calculate the out of sample error: 

```r
# set seed in order to make the results reproducible
set.seed(4711)
# the new training set will contain 60% of the data out of the old training set 
inTrain <- createDataPartition(y=pml_train$classe,p=0.6,list=F)
pml_training<-pml_train[inTrain, ]
# the validation set will contain 40% of the data out of the old training set 
pml_train_val<-pml_train[-inTrain, ]
```

## Data preparation for machine learning algorithm
1. Not all of the variables in the datasets are useful for machine learning. A visual inspection of the data shows that the first seven variables are not useful. Therefore I delete them from both training sets.  

```r
pml_training <- pml_training[, -(1:7)]
pml_train_val <- pml_train_val[, -(1:7)]
```
2. I will delete all variables with nearly zero variance because they will not contribute significantly to the machine learning success.    

```r
zero_var <- nearZeroVar(pml_training)
pml_training <- pml_training[, -zero_var]
pml_train_val <- pml_train_val[, -zero_var]
```
3. I will delete all variables with an excessive high share of missing-values because they will also not contribute significantly to the machine learning success.  

```r
mostlyNA <- sapply(pml_training, function(x) mean(is.na(x))) > 0.95
pml_training <- pml_training[, mostlyNA==F]
pml_train_val <- pml_train_val[, mostlyNA==F]
```
Notice that all steps are accomplished with the training set and the validation set. I use the training set in order to judge which variables should removed but the same variables are removed from the validation set as well.    

# Model fit and cross validation  
I decided to fit a random forest model for the data, because random forests and boosting algorithms are the most frequently used algorithms in machine learning. The model is fit based on the training set

```r
# Model fitting:
# method="rf"  -  random forest
# trControl method="cv", number=4  -  cross validation with 3 calculations
modFit <- train(pml_training$classe ~ ., method="rf", trControl=trainControl(method = "cv", number = 3), data=pml_training)
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.1.3
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```r
# show stats for final model
modFit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.74%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3347    1    0    0    0 0.0002986858
## B   18 2254    6    1    0 0.0109697236
## C    0   17 2035    2    0 0.0092502434
## D    0    0   38 1891    1 0.0202072539
## E    0    0    0    3 2162 0.0013856813
```
## Out of sample error  
In order to calculate the out of sample error, I will use the modFit model in order to make predictions for the validation set.
After that I build a confusion matrix in order to compare the predicted values to the in the validation set. 

```r
# predict pml_train_val
preds_val <- predict(modFit, newdata=pml_train_val)
confusionMatrix(preds_val, pml_train_val$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2229   13    0    0    0
##          B    2 1505   23    0    0
##          C    0    0 1341   31    0
##          D    0    0    4 1255    7
##          E    1    0    0    0 1435
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9897          
##                  95% CI : (0.9872, 0.9918)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9869          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9987   0.9914   0.9803   0.9759   0.9951
## Specificity            0.9977   0.9960   0.9952   0.9983   0.9998
## Pos Pred Value         0.9942   0.9837   0.9774   0.9913   0.9993
## Neg Pred Value         0.9995   0.9979   0.9958   0.9953   0.9989
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2841   0.1918   0.1709   0.1600   0.1829
## Detection Prevalence   0.2858   0.1950   0.1749   0.1614   0.1830
## Balanced Accuracy      0.9982   0.9937   0.9877   0.9871   0.9975
```
# New model fit and prediction on the test set
## New model
In this step I fit a new random forest model with the whole training set (pml_train). Therefore I will do all steps above again without splitting the training set into a training and validation set.

```r
pml_train <- read.csv(file = 'data_train.csv',
                      na.strings = c('NA','#DIV/0!',''))
pml_test <- read.csv(file = 'data_test.csv',
                     na.strings = c('NA','#DIV/0!',''))
pml_train <- pml_train[, -(1:7)]
pml_test  <- pml_test[, -(1:7)]

zero_var <- nearZeroVar(pml_train)
pml_train <- pml_train[, -zero_var]
pml_test  <- pml_test[, -zero_var]

mostlyNA <- sapply(pml_train, function(x) mean(is.na(x))) > 0.95
pml_train <- pml_train[, mostlyNA==F]
pml_test <- pml_test[, mostlyNA==F]


# Model fitting:
# method="rf"  -  random forest
# trControl method="cv", number=3  -  cross validation with 3 calculations
modFit <- train(pml_train$classe ~ ., method="rf", trControl=trainControl(method = "cv", number = 3), data=pml_train)
# show stats for final model
modFit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.43%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 5578    1    0    0    1 0.0003584229
## B   11 3783    3    0    0 0.0036871214
## C    0   19 3402    1    0 0.0058445354
## D    0    0   40 3174    2 0.0130597015
## E    0    0    0    6 3601 0.0016634322
```
## Prediction on test set

```r
preds_test <- predict(modFit, newdata=pml_test)
preds_test
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

