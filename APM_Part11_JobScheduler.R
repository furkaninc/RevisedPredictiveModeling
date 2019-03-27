

# Applied Predictive modeling
# Case Study: Job Scheduler

library(AppliedPredictiveModeling)
data("schedulingData")

library(lattice)
library(latticeExtra)
library(tidyverse)
library(caret)

glimpse(schedulingData)

predictors <- names(schedulingData)[1:7]

splom(schedulingData,
      col = "gray70",
      auto.key = TRUE,
      between = list(x = .5))


xyplot(Compounds ~ InputFields | Protocol,
       data = schedulingData,
       scales = list(x = list(log = 10), y = list(log = 10)),
       groups = Class,
       xlab = "Input Fields",
       auto.key = list(columns = 4),
       aspect = 1,
       as.table = TRUE)

# Split the Data
set.seed(1104)
ind <- createDataPartition(schedulingData$Class,
                           p = .8,
                           list = FALSE)

# There are lots of Zeros and the distubition is right-skewed
# We add one to NumPending column so that we can take log transform

schedulingData$NumPending <- schedulingData$NumPending + 1

trainData <- schedulingData[ind,]
testData <- schedulingData[-ind,]

# Create a "main effects only model formula" to use repeatedly
modForm <- as.formula(Class ~ Protocol + log10(Compounds) +
                        log10(InputFields) + log10(Iterations) +
                        log10(NumPending) + Hour + Day)

# Create an expanded set of predictors with interaction and nonlinear effects
modForm2 <- as.formula(Class ~ (Protocol + log10(Compounds) +
                        log10(InputFields) + log10(Iterations) +
                          log10(NumPending) + Hour + Day)^2)


# We use model.matrix() to create the whole set of predictor columns,
# and then remove those that are zero variance
expandedTrain <- model.matrix(modForm2, data = trainData)
expandedTest <- model.matrix(modForm2, data = testData)

expandedTrain <- as.data.frame(expandedTrain)
expandedTest <- as.data.frame(expandedTest)

# Some models have issues when there is a zero-variance predictor
# we find the offending columns and remove them

?checkConditionalX
zv <- checkConditionalX(expandedTrain, trainData$Class)

expandedTrain <- expandedTrain[, -zv]
expandedTest <- expandedTest[, -zv]

# Create A Cost Matrix
costMatrix <- ifelse(diag(4) == 1, 0, 1)
costMatrix[4, 1] <- 10
costMatrix[3, 1] <- 5
costMatrix[4, 2] <- 5
costMatrix[3, 2] <- 5


costMatrix <- t(costMatrix) #now observed class are cols.
costMatrix
rownames(costMatrix) <- colnames(costMatrix) <- levels(trainData$Class)

# Create A Cost Function
cost <- function(pred, obs){
  isNA <- is.na(pred)
  if(!all(isNA)){
    pred <- pred[!isNA]
    obs <- obs[!isNA]
    
    cost <- ifelse(pred == obs, 0, 1)
    if(any(pred == "VF" & obs == "L"))
      cost[pred == "VF" & obs == "L"] <- 10
    if(any(pred == "F" & obs == "L"))
      cost[pred == "F" & obs == "L"] <- 5
    if(any(pred == "F" & obs == "M"))
      cost[pred == "F" & obs == "M"] <- 5
    if(any(pred == "VF" & obs == "M"))
      cost[pred == "VF" & obs == "M"] <- 5
    return(out <- mean(cost))
  }
  else
    out <- NA
  return(out)
}

# Make a summary function that can be used with
# caret's train() function
costSummary <- function(data, lev = NULL, model = NULL){
  if(is.character(data$obs))
    data$obs <- factor(data$obs, levels = lev)
  c(postResample(data[, "pred"], data[, "obs"]),
    Cost = cost(data[, "pred"], data[, "obs"]))
}

# Create a control Object for the Models
library(doMC)
registerDoMC(cores = 3)

ctrl <- trainControl(method = "cv",
                     n = 10,
                     summaryFunction = costSummary)


# Creating Various Models for Machine Learning Process
#----------------------------------------------------

# CART Model without Costs
set.seed(857)
rpFit <- train(x = trainData[, predictors],
               y = trainData$Class,
               method = "rpart",
               metric = "Cost",
               maximize = FALSE,
               tuneLength = 20,
               trControl = ctrl)

rpFit
plot(rpFit)

# CART Model with Costs
set.seed(857)
rpFitCost <- train(x = trainData[, predictors],
                   y = trainData$Class,
                   method = "rpart",
                   metric = "Cost",
                   maximize = FALSE,
                   tuneLength = 20,
                   parms = list(loss = costMatrix),
                   trControl = ctrl)

rpFitCost
plot(rpFitCost)

# LDA
set.seed(857)
ldaFit <- train(x = expandedTrain,
                y = trainData$Class,
                method = "lda",
                metric = "Cost",
                maximize = FALSE,
                trControl = ctrl)

ldaFit

# Sparse LDA (takes very longtime)
# sldaGrid <- expand.grid(NumVars = seq(2, 112, by = 5),
#                         lambda = c(0, .01, .1, 1, 10))
# 
# set.seed(857)
# sldaFit <- train(x = expandedTrain,
#                  y = trainData$Class,
#                  method = "sparseLDA",
#                  metric = "Cost",
#                  maximize = FALSE,
#                  tuneGrid = sldaGrid,
#                  preProc = c("center", "scale"),
#                  trControl = ctrl)


# Random Forests w/out Cost


set.seed(857)
rfFit <- train(x = trainData[, predictors],
               y = trainData$Class,
               method = "rf",
               importance = TRUE,
               metric = "Cost",
               maximize = FALSE,
               trControl = ctrl,
               ntree = 401,
               tuneLength = 5)

rfFit

confusionMatrix(rfFit, norm = "none") #0.842

# Random Forests with Costs

set.seed(857)
rfFitC <- train(x = trainData[, predictors],
               y = trainData$Class,
               method = "rf",
               metric = "Cost",
               maximize = FALSE,
               tuneLength = 5,
               ntree = 401, #1000 is recommended
               classwt = c(VF = 1, F = 1, M = 5, L = 10),
               importance = TRUE,
               trControl = ctrl)

rfFitC

confusionMatrix(rfFitC, norm = "none") #.840

rfRes <- predict(rfFit, newdata = testData)
confusionMatrix(rfRes, testData$Class) #.85
postResample(pred = rfRes, obs = testData$Class) #.85 acc
