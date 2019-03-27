
# APM - intro to classification
##############################
# Class Performance

library(tidyverse)
library(AppliedPredictiveModeling)

?quadBoundaryFunc
set.seed(975)
trainSet <- quadBoundaryFunc(500)
testSet <- quadBoundaryFunc(1000)

testSet <- testSet %>%
  mutate(class2 = case_when(
    class == "Class1" ~ 1,
    class == "Class2" ~ 0),
          ID = 1:nrow(testSet))


ggplot(trainSet, aes(x = X1, y = X2, color = class)) +
  theme_bw() +
  geom_point(alpha = .4) +
  labs(title = "X1 - X2 distribution")
  

xyplot(X2 ~ X1|"class distributions", data = trainSet,
       groups = class,
       grid = TRUE)

### Fit some models
library(MASS)
qdaFit <- qda(class ~ X1 + X2, data = trainSet)
library(randomForest)
rfFit <- randomForest(class ~ X1 + X2,
                      data = trainSet,
                      ntree = 2000)

# Predict the test set
testSet$qda <- predict(qdaFit, newdata = testSet)$posterior[,1]
testSet$rf <- predict(rfFit, newdata = testSet, type = "prob")[,1]

# Generate Calibration Analysis
library(caret)
calData1 <- calibration(class ~ qda + rf,
                        data = testSet,
                        cuts = 10)

# plot the curve
xyplot(calData1, 
       auto.key = list(columns = 2),
       panel = function(x, y, ...){
         panel.xyplot(x, y, ...)
         panel.abline(0, 1, lty = 2, col = "black")
       })

# To calibrate the data, we treat the probabilities
# as inputs into the model
trainProbs <- trainSet
trainProbs$qda <- predict(qdaFit)$posterior[,1]

# These models take the probabilities
# as inputs, and based on the true class,
# re-calibrate them.

library(klaR)
nbCal <- NaiveBayes(class ~ qda, data = trainProbs,
                    usekernel = TRUE)

# We use relevel() here b/c glm() models
# the probability of the 2nd factor level

lrCal <- glm(relevel(class, "Class2") ~ qda,
             data = trainProbs,
             family = binomial)

# now, we re-predict the test set usnig the
# modified class probability estimates

testSet$qda2 <- predict(nbCal, newdata = testSet[, "qda", drop = F])$posterior[,1]
testSet$qda3 <- predict(lrCal, newdata = testSet[, "qda", drop = F], type = "response")

# manipulate the data a bit for visualization

simulatedProbs <- testSet[, c("class", "rf", "qda3")]
names(simulatedProbs) <- c("TrueClass", "RandomForestProbs", "QDACalibrated")
simulatedProbs$RandomForestClass <- predict(rfFit, newdata = testSet)

calData2 <- calibration(class ~ qda + qda2 + qda3,
                        data = testSet)

calData2$data$calibModelVar <- as.character(calData2$data$calibModelVar)
calData2$data$calibModelVar <- ifelse(calData2$data$calibModelVar == "qda", 
                                      "QDA",
                                      calData2$data$calibModelVar)
calData2$data$calibModelVar <- ifelse(calData2$data$calibModelVar == "qda2", 
                                      "Bayesian Calibration",
                                      calData2$data$calibModelVar)

calData2$data$calibModelVar <- ifelse(calData2$data$calibModelVar == "qda3", 
                                      "Sigmoidal Calibration",
                                      calData2$data$calibModelVar)

calData2$data$calibModelVar <- factor(calData2$data$calibModelVar,
                                      levels = c("QDA", 
                                                 "Bayesian Calibration", 
                                                 "Sigmoidal Calibration"))

xyplot(calData2, 
       auto.key = list(columns = 1))

##################################################
# Use German Credit Data to re-create the model
detach("package:klaR", unload = TRUE)
library(caret)
data(GermanCredit)

# remove near zero var and duplicate values to
# avoid co-linearity
zv <- nearZeroVar(GermanCredit)
names(GermanCredit)[zv] 
GermanCredit <- GermanCredit[, -zv]

GermanCredit$CheckingAccountStatus.lt.0 <- NULL
GermanCredit$SavingsAccountBonds.lt.100 <- NULL
GermanCredit$EmploymentDuration.lt.1 <- NULL
GermanCredit$EmploymentDuration.Unemployed <- NULL
GermanCredit$Personal.Male.Married.Widowed <- NULL
GermanCredit$Property.Unknown <- NULL
GermanCredit$Housing.ForFree <- NULL

# split the data into training and test set
set.seed(100)
ind <- createDataPartition(GermanCredit$Class,
                           p = .8,
                           list = FALSE)

GermanCreditTrain <- GermanCredit[ind,]
GermanCreditTest <- GermanCredit[-ind,]

ctrl <- trainControl(method = "repeatedcv",
                     repeats = 5)
set.seed(1056)
logisticReg <- train(Class ~., data = GermanCreditTrain,
                     method = "glm",
                     trControl = ctrl)

logisticReg

# predict the test set
creditResults <- data.frame(obs = GermanCreditTest$Class)
creditResults$prob <- predict(logisticReg, newdata = GermanCreditTest, type = "prob")[,"Bad"]
creditResults$pred <- predict(logisticReg, newdata = GermanCreditTest, type = "raw")

creditResults <- creditResults %>%
  mutate(label = case_when(
    obs == "Bad" ~ "True Outcome: Bad Credit",
    TRUE ~ "True Outcome: Good Credit"
  ))


# plot the probability of bad credit
histogram(~prob|label,
          data = creditResults,
          layout = c(2,1),
          nint = 20,
          xlab = "probability of Bad Credit",
          type = "count",
          col = "gray")

ggplot(creditResults, aes(x = prob)) +
  theme_bw() +
  geom_histogram(bins = 20) +
  facet_wrap(~ label) +
  labs(xlab = "probability of Bad Credit") 

# calculate and plot the calibration curve
creditCalib <- calibration(obs ~ prob,
                           data = creditResults)

xyplot(creditCalib,
       col = "black")  

# Create the confusion matrix from the test set
confusionMatrix(data = creditResults$pred,
                reference = creditResults$obs)

## ROC curves  
# like glm(), roc() treats the last level
# of the factor as the event of interest
# so we use relevel() to change the observed class data

str(creditResults)
library(pROC)
 
creditROC <- roc(relevel(creditResults$obs, "Good"),
                 creditResults$prob) 

coords(creditROC, "all")[,1:3]
auc(creditROC)  
ci.auc(creditROC) 
plot(creditROC)
  
# Lift charts
creditLift <- lift(obs ~ prob, data = creditResults)
xyplot(creditLift,
       col = "navy")


