
# REGRESSION TREES EXERCISES
##############################################################


# 8.4. Use a single predictor in the solubility data, such as the molecular weight
# or the number of carbon atoms and fit several models:
#   (a) A simple regression tree
# (b) A random forest model
# (c) A stochastic gradient boosting model
# Plot the predictor data versus the solubility results for the test set. Overlay
# the model predictions for the test set. How do the model differ? Does changing
# the tuning parameter(s) significantly affect the model fit?
#   

require(caret)
require(AppliedPredictiveModeling)
require(tidyverse)

data(solubility)

onevarData <- solTrainXtrans %>%
  select(contains("MolWeight"))

trainData <- onevarData
trainData$y <- solTrainY #rpart&ctree only accepts formula method.

index <- createFolds(solTrainY, k = 10, returnTrain = TRUE)
ctrl <- trainControl(method = "cv", n = 10)

# a) a simple tree:

require(party)

set.seed(100)
simpleTree <- ctree(y ~., data = trainData,
                    controls = ctree_control(maxdepth = 3))

simpleTree

plot(simpleTree)

# random-forest with CV resampling

set.seed(100)
rfTune <- train(y ~., data = trainData,
                method = "rf",
                ntree = 401,
                trControl = ctrl)


rfTune
rfTune$results

# Gradient-Boosting
boostGrid <- expand.grid(shrinkage = c(0.1, 0.01),
                         interaction.depth = c(1,3,5,7),
                         n.trees = 401,
                         n.minobsinnode = 20)


set.seed(100)
gbmTune <- train(y ~., data = trainData,
                 method = "gbm",
                 tuneGrid = boostGrid,
                 trControl = ctrl,
                 verbose = FALSE)

gbmTune

# Compare the prediction accuracy & results

testResults <- data.frame(original = solTestY,
                          singleTree = predict(simpleTree, newdata = solTestXtrans),
                          rfPred = predict(rfTune, newdata = solTestXtrans),
                          gbmPred = predict(gbmTune, newdata = solTestXtrans))

postResults <- data.frame(simple = postResample(obs = solTestY, pred = testResults$y),
                          rfmetric = postResample(obs = solTestY, pred = testResults$rfPred),
                          gbmmetric = postResample(obs = solTestY, pred = testResults$gbmPred))

postResults

################################################################################################3



  