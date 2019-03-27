###################

# Applied Predictive Modeling
# Classification Trees
setwd("~/R/Applied_Predictive_Modeling")

library(party)
library(partykit)
library(tidyverse)
library(caret)

load("grantApplication.RData")

# Basic Classification Trees

ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     index = list(TrainSet = pre2008),
                     savePredictions = TRUE)

set.seed(476)
rpartFit <- train(x = training[,fullSet],
                  y = training$Class,
                  method = "rpart",
                  tuneLength = 30,
                  metric = "ROC",
                  trControl = ctrl)

rpartFit

plot(as.party(rpartFit$finalModel))

rpartCM <- confusionMatrix(rpartFit, norm = 'none')

rpartRoc <- roc(response = rpartFit$pred$obs,
                predictor = rpartFit$pred$successful,
                levels = rev(levels(rpartFit$pred$obs)))

auc(rpartRoc)  #.8893

set.seed(476)
rpartFactorFit <- train(x = training[,factorPredictors], 
                        y = training$Class,
                        method = "rpart",
                        tuneLength = 30,
                        metric = "ROC",
                        trControl = ctrl)
rpartFactorFit 
plot(as.party(rpartFactorFit$finalModel))

rpartFactorCM <- confusionMatrix(rpartFactorFit, norm = "none")
rpartFactorCM

rpartFactorRoc <- roc(response = rpartFactorFit$pred$obs,
                      predictor = rpartFactorFit$pred$successful,
                      levels = rev(levels(rpartFactorFit$pred$obs)))


levels = rev(levels(j48Fit$pred$obs)))
auc(rpartFactorRoc) #0.7538

#

set.seed(476)
j48FactorFit <- train(x = training[,factorPredictors], 
                      y = training$Class,
                      method = "J48",
                      metric = "ROC",
                      trControl = ctrl)
j48FactorFit

j48FactorCM <- confusionMatrix(j48FactorFit, norm = "none")
j48FactorCM

j48FactorRoc <- roc(response = j48FactorFit$pred$obs,
                    predictor = j48FactorFit$pred$successful,
                    levels = rev(levels(j48FactorFit$pred$obs)))

set.seed(476)
j48Fit <- train(x = training[,fullSet], 
                y = training$Class,
                method = "J48",
                metric = "ROC",
                trControl = ctrl)

j48CM <- confusionMatrix(j48Fit, norm = "none")
j48CM

j48Roc <- roc(response = j48Fit$pred$obs,
              predictor = j48Fit$pred$successful,

plot(rpartRoc, legacy.axes = TRUE, col = "red")
plot(rpartFactorRoc, add = TRUE, 
     legacy.axes = TRUE, col = "blue")

#############################################################################################
# BAGGED TREES (failed due to limitations on XPS13)
library(doMC)
registerDoMC(cores = 3)

set.seed(476)
treebagFit <- train(x = training[,fullSet], 
                    y = training$Class,
                    method = "treebag",
                    nbagg = 10,
                    metric = "ROC",
                    trControl = ctrl)
treebagFit


treebagCM <- confusionMatrix(treebagFit, norm = "none")
treebagCM

library(pROC)
treebagRoc <- roc(response = treebagFit$pred$obs,
                  predictor = treebagFit$pred$successful,
                  levels = rev(levels(treebagFit$pred$obs)))

auc(treebagRoc) #.905

set.seed(476)
treebagFactorFit <- train(x = training[,factorPredictors], 
                          y = training$Class,
                          method = "treebag",
                          nbagg = 10,
                          metric = "ROC",
                          trControl = ctrl)
treebagFactorFit


treebagFactorCM <- confusionMatrix(treebagFactorFit, norm = "none")
treebagFactorCM
treebagFactorRoc <- roc(response = treebagFactorFit$pred$obs,
                        predictor = treebagFactorFit$pred$successful,
                        levels = rev(levels(treebagFactorFit$pred$obs)))

auc(treebagFactorRoc) #.76

##################################################
# ------------ Random Forests ---------------------(requires too much time)
library(doMC)
registerDoMC(3)

mtryValues <- 100 #other values dropped due to memory restrictions.

set.seed(476)
rfFit <- train(x = training[,fullSet], 
               y = training$Class,
               method = "rf",
               ntree = 501, #1000 is recommended
               tuneGrid = data.frame(mtry = mtryValues),
               importance = TRUE,
               metric = "ROC",
               trControl = ctrl)

rfFit 
rfFit$bestTune #mtry = 100

rfCM <- confusionMatrix(rfFit, norm = "none")
rfCM

library(pROC)
rfRoc <- roc(response = rfFit$pred$obs,
             predictor = rfFit$pred$successful,
             levels = rev(levels(rfFit$pred$obs)))

auc(rfRoc) #.94

rfimp <- varImp(rfFit) 
plot(rfimp , top = 25, col = "blue")


# Test Set performance of RF
rfPred <- predict(rfFit, newdata = testing)
confusionMatrix(rfPred, testing$Class) #.8745 accuracy, good...

results <- data.frame(observations = testing$Class, randomfor = rfPred)
results %>%
  mutate(acc = case_when(
    observations == randomfor ~ 1,
    TRUE ~ 0 -> results)
  ))
