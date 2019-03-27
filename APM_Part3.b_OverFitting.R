require(tidyverse)
require(caret)
require(AppliedPredictiveModeling)

data("GermanCredit")
glimpse(GermanCredit)

GermanCredit[!complete.cases(GermanCredit),]

# Remove near-zero var predictors
zv <- nearZeroVar(GermanCredit)
GermanCredit <- GermanCredit[, -zv]

# Get rid of variables that duplicate values.
# Just to avoid linear dependencies



dependentVars <- c("CheckingAccountStatus.lt.0", "EmploymentDuration.lt.1",
                   "EmploymentDuration.Unemployed","Personal.Male.Married.Widowed",
                   "Property.Unknown","Housing.ForFree")
GermanCredit <- GermanCredit %>%
  select(-dependentVars)


# Split the data into training and test set
set.seed(100)

indexes <- createDataPartition(GermanCredit$Class, p = .8, list = FALSE)
GermanCreditTrain <- GermanCredit[indexes, ]
GermanCreditTest <- GermanCredit[-indexes, ]
# Tuning parameters
library(kernlab)
set.seed(231)
?sigest
sigDist <- sigest(Class ~., data = GermanCreditTrain,
                  frac = 1, na.action = na.fail)

sigDist
svmTuneGrid <- data.frame(sigma = as.vector(sigDist)[1],
                          C = 2^(-2:7)) #C stands for cost


# Prepare for parallel computation 
library(doMC)
registerDoMC(cores = 3)



# 10-fold CV with 5 repeat model

ctrl <- trainControl(method = "repeatedcv",
                     classProbs = TRUE,
                     n = 10,
                     repeats = 5)
set.seed(1056)

svmFit <- train(Class ~., data = GermanCreditTrain,
                method = "svmRadial",
                preProc = c("center", "scale"),
                tuneGrid = svmTuneGrid,
                trControl = ctrl,
                verbose = FALSE)
#print results
svmFit
names(svmFit)
svmFit$resample
range(svmFit$resample$Accuracy) #0.6875-0.8250
svmFit$results
svmFit$bestTune
svmFit$finalModel 
## A line plot of the average performance. The 'scales' argument is actually an 
## argument to xyplot that converts the x-axis to log-2 units.
plot(svmFit, scales = list(x = list(log = 2)),
     main = "Repeated CV Accuracy and Cost Values")


# Test set predictions
predictedClasses <- predict(svmFit, newdata = GermanCreditTest, type = "raw")
str(predictedClasses)

confusionMatrix(predictedClasses, GermanCreditTest$Class) #.77

## Use the "type" option to get class probabilities

predictedProbs <- predict(svmFit, newdata = GermanCreditTest, type = "prob")
head(predictedProbs)
cbind(predictedProbs, GermanCreditTest$Class)

# 10-fold CV with NO repeats
ctrl10 <- trainControl(method = "cv", n = 10, 
                       classProbs = TRUE)
set.seed(1056)
svmFit10CV <- train(Class ~., data = GermanCreditTrain,
                  method = "svmRadial",
                  preProc = c("center", "scale"),
                  tuneGrid = svmTuneGrid,
                  trControl = ctrl10)

svmFit10CV #to print the results
svmFit10CV$results


plot(svmFit10CV, scales = list(x = list(log = 2)),
     main = "10 fold CV and Cost Values") #plot cv accuracy and cost value

predicted10CV <- predict(svmFit10CV, newdata = GermanCreditTest)
confusionMatrix(predicted10CV, GermanCreditTest$Class)

# Bootstrapping with n = 50
ctrlBoot <- trainControl(method = "boot", n = 50)

set.seed(1056)
svmFitBoot <- train(Class ~., data = GermanCreditTrain,
                    method = "svmRadial",
                    trControl = ctrlBoot,
                    tuneGrid = svmTuneGrid)     

plot(svmFitBoot, scales = list(x = list(log = 2)),
     main = "Bootstrap and Cost Values")

svmFitBoot
names(svmFitBoot)
range(svmFitBoot$resample$Accuracy) 
svmFitBoot$bestTune
svmFitBoot$finalModel

predictedBoot <- predict(svmFitBoot, newdata = GermanCreditTest, type = "raw")
confusionMatrix(predictedBoot, GermanCreditTest$Class) #not good enough, bootstrap yields to bias

# Choosing Betwenn Models, see if those high-level complicated models
# are doing better than a simpler model like Logistic Regression

# Logistic Regression Model
ctrlGlm <- trainControl(method = "repeatedcv", n = 10,
                        repeats = 5)
set.seed(1056)
glmModel <- train(Class ~., data = GermanCreditTrain,
                method = "glm",
                trControl = ctrlGlm)

glmModel
names(glmModel)
range(glmModel$resample$Accuracy) #0.64-0.87

predictedGlm <- predict(glmModel, newdata = GermanCreditTest)
confusionMatrix(predictedGlm, GermanCreditTest$Class) #simple, effective and as accurate as svm

# Resample comparison
?resamples
resamp <- resamples(list(SVM = svmFit, Logistic = glmModel))
names(resamp)


modelDiff <- diff(resamp)
summary(modelDiff)

##############################################################
sessionInfo()

