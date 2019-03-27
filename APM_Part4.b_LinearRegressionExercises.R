

# Linear Regression Exercises
# from the book Applied Predictive Modeling
################################################

library(tidyverse)
library(caret)
library(AppliedPredictiveModeling)

# Exercise 6.2:

data("permeability")

# filter out near zero var predictors
fingerprints <- as.data.frame(fingerprints)
permeability <- as.data.frame(permeability)

zv <- nearZeroVar(fingerprints)
fingerprints <- fingerprints[, -zv]
fingerSet <- cbind(fingerprints, permeability)

# Split data
set.seed(111)
ind <- createDataPartition(fingerSet$permeability, p = .7,
                           list = FALSE)

fingerTrain <- fingerSet[ind,]
fingerTest <- fingerSet[-ind,]

# Tune a PLS model
ctrl <- trainControl(method = "cv", n = 10,
                     savePredictions = "final")

set.seed(111)

plsTune <- train(permeability ~., data = fingerTrain,
                 method = "pls",
                 tuneGrid = expand.grid(ncomp = 1:35),
                 trControl = ctrl)

plsTune #ncomp = 7 is optimal, decided by the model. correspondent R^2 = 0.53
plot(plsTune, type = "b", col = "red",
     main = "PLS tuning graph with cross-validated RMSE")

plsPred <- predict(plsTune, newdata = fingerTest)
caret::postResample(obs = fingerTest$permeability, pred = plsPred) # R^2 = 0.436

testResults <- data.frame(observations = fingerTest$permeability,
                          plsPredictions = plsPred)

testResults$residuals <- testResults$observations - testResults$plsPredictions
testResults

par(mfrow = c(1,2))
plot(x = testResults$observations, y = testResults$plsPredictions,
     col = "red",
     type = "p",
     main = "original  values vs. predictions")
abline(0, 1, col = "black")
plot(x = testResults$residuals, y = testResults$plsPredictions,
     col = "blue",
     type = "p",
     main = "residuals vs. predictions")
abline(h = 0, col = "black") #not quite ok
#some other consiretion might be to wtich with an ordinary regression...

plsImp <- varImp(plsTune, scale = FALSE)
plot(plsImp, top = 25, col = "red",
     main = "Variable Importance")


# Exercise 6.3

data("ChemicalManufacturingProcess")
glimpse(ChemicalManufacturingProcess)

# check missing values and impute
CMP <- ChemicalManufacturingProcess #shorten the name
rm(ChemicalManufacturingProcess) #remove the duplicate

CMP[!complete.cases(CMP),]

cmpPP <- preProcess(CMP, method = "bagImpute")
chemical <- predict(cmpPP, newdata = CMP)

histogram(~BiologicalMaterial01, data = chemical,
          type = "count",
          main = "histogram og bio.mat.01",
          col = "grey")


histogram(~BiologicalMaterial02, data = chemical,
          type = "count",
          main = "histogram og bio.mat.02",
          col = "grey")



# split data
set.seed(123)
ind <- createDataPartition(chemical$Yield, p = 0.7, list = FALSE)
chemicalTrain <- chemical[ind,]
chemicalTest <- chemical[-ind,]

# preprocess data
chemicalTrainPP <- preProcess(chemicalTrain, method = c("BoxCox", "center", "scale"))
chemicalTestPP <- preProcess(chemicalTest, method = c("BoxCox", "center", "scale"))

chemicalTrainTrans <- predict(chemicalTrainPP, newdata = chemicalTrain)
chemicalTestTrans <- predict(chemicalTestPP, newdata = chemicalTest)

# try pls and pcr tune for modeling
ctrl <- trainControl(method = "cv", n = 10)
tunegrid <- expand.grid(ncomp = 1:25)

set.seed(123)
plsTune <- train(Yield~., data = chemicalTrainTrans,
                 trControl = ctrl,
                 tuneGrid = tunegrid,
                 method = "pls")

plsTune #ncomp optimal is 3 according to model with r^2 = 0.642
plot(plsTune, type = "b")

testResults <- data.frame(obs = chemicalTestTrans$Yield,
                          pls = predict(plsTune, newdata = chemicalTestTrans))


set.seed(123)
pcrTune <- train(Yield~., data = chemicalTrainTrans,
                 trControl = ctrl,
                 tuneGrid = tunegrid,
                 method = "pcr")

pcrTune #ncomp = 16 according to model with r^2 = 0.632
plot(pcrTune, type = "b")

testResults$pcr <- predict(pcrTune, newdata = chemicalTestTrans)
testResults$resPls <- testResults$obs - testResults$pls
testResults$resPcr <- testResults$obs - testResults$pcr

par(mfrow = c(2,2))

plot(testResults$pls, testResults$obs,
     type = "p",
     main = "original values and pls predictions")
abline(0, 1, col = "red")

plot(testResults$pls, testResults$resPls,
     type = "p",
     main = "pls residuals and predictions")
abline(h = 0, col = "red")

plot(testResults$pcr, testResults$obs,
     type = "p",
     main = "original values and pcr predictions")
abline(0, 1, col = "red")

plot(testResults$pcr, testResults$resPcr,
     type = "p",
     main = "pcr residuals and predictions")
abline(h = 0, col = "red")

plsImp <- varImp(plsTune, scale = F)
pcrImp <- varImp(pcrTune, scale = F)

par(mfrow = c(1,2))
plot(plsImp, top = 20, col = "red")
plot(pcrImp, top = 20, col = "blue")

RMSE(obs = chemicalTestTrans$Yield, pred = testResults$pls) #0.74, pls is prefferable
RMSE(obs = chemicalTestTrans$Yield, pred = testResults$pcr) #0.77 pcr is inferior
