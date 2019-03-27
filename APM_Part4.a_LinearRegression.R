


# APPLIED PREDICTIVE MODELING
# Linear Regression
################################

require(caret)
require(tidyverse)
require(AppliedPredictiveModeling)

data(solubility)

# Some initial plots of data

xyplot(solTrainY ~ solTrainX$MolWeight,
       type = c("g", "p"),
       ylab = "Solubility (log)",
       xlab = "Molecular Weight",
       main = "(a)")

xyplot(solTrainY ~ solTrainX$NumRotBonds,
       type = c("p", "g"),
       ylab = "Solubility (log)",
       xlab = "Number of Rotatable Bonds",
       main = "# of rotatable bonds and solubility")

bwplot(solTrainY ~ ifelse(solTrainX[,100] == 1, 
                          "structure present", 
                          "structure absent"),
       ylab = "Solubility (log)",
       main = "(b)",
       horizontal = FALSE)

# Find the columns that are not fingerprints (i.e continious
# predictors) 

fingerprints <- grep("FP", names(solTrainXtrans))

?featurePlot
featurePlot(solTrainXtrans[, -fingerprints],
            solTrainY,
            between = list(x = 1, y = 1),
            type = c("g", "p", "smooth"),
            col = "gray",
            labels = rep("", 2))

# check out the high-correlated features

library(corrplot)
notFingerprintsCor <- cor(solTrainXtrans[, -fingerprints])
corrplot::corrplot(notFingerprintsCor, order = "hclust",
                  tl.cex = .8)

highCor <- findCorrelation(notFingerprintsCor, cutoff = .9)

# Linear Regression
?createFolds
set.seed(100)
indx <- createFolds(solTrainY, returnTrain = TRUE)
ctrl <- trainControl(method = "cv", n = 10)

### Linear regression model with all of the predictors. This will
### produce some warnings that a 'rank-deficient fit may be
### misleading'. This is related to the predictors being so highly
### correlated that some of the math has broken down.
set.seed(100)
lmtune0 <- train(x = solTrainXtrans,
                 y = solTrainY,
                 method = "lm",
                 trControl = ctrl)

### And another using a set of predictors reduced by unsupervised
### filtering. We apply a filter to reduce extreme between-predictor
### correlations. Note the lack of warnings.

trainXfiltered <- solTrainXtrans[, -highCor]
testXfiltered <- solTestXtrans[, -highCor]

lmtune <- train(x = trainXfiltered,
                y = solTrainY,
                method = "lm",
                trControl = ctrl)

lmtune

testResults <- data.frame(observations = solTestY,
                          LinearRegression = predict(lmtune, newdata = testXfiltered))

head(testResults)
residuals <- testResults$LinearRegression - testResults$observations


# Some plots of Linear Regression Predictions, 
# Original Observations and Residuals

xyplot(testResults$LinearRegression ~ testResults$observations|"observations vs. predicted",
       col = "navy",
       type = 'p',
       grid = TRUE,
       panel = function(x, y, ...){
         panel.xyplot(x, y)
         panel.abline(0, 1, col = "red")},
       xlab = "observations",
       ylab = "predicted")
       


axisRange <- extendrange(c(testResults$LinearRegression, residuals))
plot(x = testResults$LinearRegression,
     y = residuals,
     xlab = "predictions",
     ylab = "residuals",
     xlim = axisRange,
     ylim = axisRange,
     type = "p",
     col = "black")
abline(h = 0, col = "red")

######################################################
# Run PLS and PCR on solubility data and compare results
set.seed(100)
plsTune <- train(x = solTrainXtrans,
                 y = solTrainY,
                 method = "pls",
                 trControl = ctrl,
                 tuneGrid = expand.grid(ncomp = 1:35))

plsTune

testResults$PLS <- predict(plsTune, newdata = solTestXtrans)

set.seed(100)
pcrTune <- train(x = solTrainXtrans,
                 y = solTrainY,
                 method = "pcr",
                 trControl = ctrl,
                 tuneGrid = expand.grid(ncomp = 1:35))

pcrTune

testResults$PCR <- predict(pcrTune, newdata = solTestXtrans)

names(plsTune)

plsResamples <- plsTune$results
plsResamples$Model <- "PLS"
pcrResamples <- pcrTune$results
pcrResamples$Model <- "PCR"
plsPlotData <- rbind(plsResamples, pcrResamples)

xyplot(RMSE ~ ncomp|"PLS versus PCR techniques",
      data = plsPlotData,
      xlab = "# Components",
      ylab = "RMSE (Cross-Validation)",
      auto.key = list(columns = 2),
      groups = Model,
      type = c("o", "g"))

plsImp <- varImp(plsTune, scale = FALSE)
plot(plsImp, top = 25, scales = list(y = list(cex = .95)))

     