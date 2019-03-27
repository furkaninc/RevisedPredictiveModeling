
# Non Linear Regression Exercise

# load the required packages and dataset
require(caret)
require(tidyverse)
require(AppliedPredictiveModeling)

data(tecator)

absorp <- absorp %>%
  as.data.frame()

endpoints <- endpoints %>%
  as.data.frame()

names(endpoints) <- c("moisture", "fat", "protein")

tecator <- cbind(absorp, endpoints[2]) #"fat" is the response

# Split the data into Trainig and Test set:

set.seed(123)
indSelect <- createDataPartition(tecator$fat, p = .7,
                                 list = FALSE)

tecatorTrain <- tecator[indSelect,]
tecatorTest <- tecator[-indSelect,]

# Tune some models

ctrl <- trainControl(method = "cv", n = 10,
                     savePredictions = "final")

# svm radial 
sigCal <- sigest(fat ~., data = tecatorTrain)

svmRadialGrid <- data.frame(sigma = rep(sigCal[1], 14),
                            C = 2^(1:14))

library(doMC)
registerDoMC(3)
set.seed(123)

svmRadialTune <- train(fat ~., data = tecatorTrain,
                       method = "svmRadial",
                       preProc = c("center", "scale"),
                       trControl = ctrl,
                       tuneGrid = svmRadialGrid)

svmRadialTune
plot(svmRadialTune, scales = list(x = list(log = 2)),
     main = "RMSE vs Cost Values")


# svm polynomial
svmPolyGrid <- expand.grid(scale = c(0.01, 0.005, 0.001),
                           degree = 1:2,
                           C = 2^(1:14))


registerDoMC(3)

set.seed(123)
svmPolyTune <- train(fat ~., data = tecatorTrain,
                       method = "svmPoly",
                       preProc = c("center", "scale"),
                       trControl = ctrl,
                       tuneGrid = svmPolyGrid)

svmPolyTune
plot(svmPolyTune, scales = list(x = list(log = 2)))

# knn

knnGrid <- expand.grid(k = 1:25)

registerDoMC(3)
set.seed(123)
knnTune <- train(fat~., data = tecatorTrain,
                 method = "knn",
                 preProc = c("center", "scale"),
                 trControl = ctrl,
                 tuneGrid = knnGrid)

knnTune
plot(knnTune)

#mars

registerDoMC(3)
set.seed(123)
marsTune <- train(fat~., data = tecatorTrain,
                  method = "earth",
                  preProc = c("BoxCox", "center", "scale"),
                  trControl = ctrl,
                  tuneGrid = expand.grid(nprune = c(2:25),
                                         degree = 1))

marsTune
plot(marsTune)

marsImp <- varImp(marsTune, scale = FALSE)
plot(marsImp, top = 25, col = "red")

# trying mars model with only the most important predictor
registerDoMC(3)
set.seed(123)
marsV40Tune <- train(fat ~ V40, data = tecatorTrain,
                          method = "earth",
                          preProc = c("BoxCox", "center", "scale"),
                          trControl = ctrl)



xyplot(fat ~ V40, data = tecatorTrain,
       grid = TRUE,
       xlab = "V40",
       ylab = "fat",
       main = "V40 and fat interaction",
       panel = function(x, y, ...){
         panel.xyplot(x, y, ...)
         panel.lines(x, fitted(marsV40Tune), ..., col ="red")
         }) #seems reduntant?

# prediction results

testResults <- data.frame(original = tecatorTest$fat, 
                          svmRad = predict(svmRadialTune, newdata = tecatorTest),
                          svmPoly = predict(svmPolyTune, newdata = tecatorTest),
                          knnResults = predict(knnTune, newdata = tecatorTest),
                          multiVariate = predict(marsTune, newdata = tecatorTest))

postResample(pred = testResults$svmRad, obs = tecatorTest$fat)
postResample(pred = testResults$svmPoly, obs = tecatorTest$fat)
postResample(pred = testResults$knnResults, obs = tecatorTest$fat)
postResample(pred = testResults$y, obs = tecatorTest$fat) #multiVariate colum name is changed to y



