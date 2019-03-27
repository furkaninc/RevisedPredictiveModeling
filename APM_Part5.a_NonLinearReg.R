

# NON-LINEAR REGRESSION
# PART 5

library(tidyverse)
library(caret)
library(AppliedPredictiveModeling)

data(solubility)
# create control function and create fold assigments
# explicitly instead of relying
# on random seed number.

set.seed(100)
indx <- createFolds(solTrainY, returnTrain = TRUE)
ctrl <- trainControl(method = "cv", index = indx)

# 7.1: "Neural Networks"

# nnetGrid <- expand.grid(decay = c(0.01, 0.1),
#                         size = c(3, 7, 13),
#                         bag = FALSE)
# 
# library(doMC)
# registerDoMC(cores = 3)

# nnetTune <- train(x = solTrainXtrans,
#                   y = solTrainY,
#                   method = "avNNet",
#                   tuneGrid = nnetGrid,
#                   preProcess = c("center", "scale"),
#                   linout = TRUE,
#                   trace = FALSE,
#                   MaxNWts = 13 * (ncol(solTrainXtrans) + 1) + 13 + 1,
#                   maxit = 1000,
#                   allowParallel = FALSE)



# 7.2: MARS
library(doMC)
registerDoMC(3)
set.seed(100)
marsTune <- train(x = solTrainXtrans,
                  y = solTrainY,
                  method = "earth",
                  tuneGrid = expand.grid(degree = 1, nprune = 2:38),
                  trControl = ctrl)


marsTune
plot(marsTune)

marsImp <- varImp(marsTune, scale = FALSE)
plot(marsImp, top = 25, col = "red")



xyplot(predMars ~ solTestY|"obs vs. mars pred",
       grid = TRUE,
       col = "black",
       xlab = "observations",
       ylab = "MARS predictions",
       panel = function(x, y, ...){
         panel.xyplot(x, y, ...)
         panel.abline(0, 1 , col = "red")
       })

resMars <- solTestY - predMars

xyplot(predMars ~ resMars|"residuals and mars pred",
       grid = TRUE,
       col = "navy",
       xlab = "preds",
       ylab = "residuals",
       panel = function(x, y, ...){
         panel.xyplot(x, y, ...)
         panel.abline(h = 0 , col = "red")
       })

plotmo(marsTune, col = "red")

# 7.3: SVM
set.seed(100)
svmTune <- train(x = solTrainXtrans, y = solTrainY,
                 method = "svmRadial",
                 preProc = c("center", "scale"),
                 trContol = ctrl,
                 tuneLength = 14)


svmTune
plot(svmTune, type = "b", scales = list(x = list(log =2)))


svmGrid <- expand.grid(degree = 1:2, 
                       scale = c(0.01, 0.005, 0.001), 
                       C = 2^(-2:5))
set.seed(100)
svmPTune <- train(x = solTrainXtrans, y = solTrainY,
                  method = "svmPoly",
                  preProc = c("center", "scale"),
                  tuneGrid = svmGrid,
                  trControl = ctrl)


svmPTune
plot(svmPTune, scales = list(x = list(log = 2)),
     between = list(x = .5, y = 1))



# 7.4: KNN
# first we need to remove near-zero-var predictors

knnDesrc <- solTrainXtrans[, -nearZeroVar(solTrainXtrans)]

set.seed(100)
knnTune <- train(x = knnDesrc, y = solTrainY,
                 method = "knn",
                 preProc = c("center", "scale"),
                 tuneGrid = expand.grid(k = 1:20),
                 trControl = ctrl)

knnTune
plot(knnTune, main = "K-parameter decision")

# see the predictions for all models with relevant metrics (i.e RMSE)
testResults <- data.frame(original = solTestY, predMars = predict(marsTune, newdata = solTestXtrans))
testResults$radialSVM <- predict(svmTune, newdata = solTestXtrans)
testResults$polySVM <- predict(svmPTune, newdata = solTestXtrans)
testResults$knn <- predict(knnTune, newdata = solTestXtrans)

postResample(obs = solTestY, pred = testResults$polySVM)
postResample(obs = solTestY, pred = testResults$radialSVM)
postResample(obs = solTestY, pred = testResults$predMars)
postResample(obs = solTestY, pred = testResults$knn)


