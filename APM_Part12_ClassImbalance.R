

# This part is from the book
# Applied Predictive Modeling
# Concerning Class-Imbalance on datasets

#####################################

###################################################
# Case Study: Predicting Caravan Policy Ownership
##################################################


# SECTION 1
####################################################
# load the data
library(kernlab)
data("ticdata")

# load the required libraries
library(pROC)
library(tidyverse)
library(caret)
library(doMC)


# explore the data
glimpse(ticdata)
levels(ticdata$CARAVAN)
typeof(ticdata$MGODPR)
class(ticdata)

#un-order the factors to convert them into nominal values

isOrdered <- unlist(map(ticdata, ~ any(class(.) == "ordered")))

convertCols <- c("STYPE", "MGEMLEEF",
                 "MOSHOOFD", names(isOrdered)[isOrdered])

for(i in convertCols){
  ticdata[, i] <-
  factor(gsub(" ", "0", format(as.numeric(ticdata[, i]))))
}

ticdata$CARAVAN <- factor(as.character(ticdata$CARAVAN),
                          levels = rev(levels(ticdata$CARAVAN)))




glimpse(ticdata)

map_dbl(ticdata, ~ length(unique(.))) # yes

# Split the data into three sets:
# training, evaluation and test set.

set.seed(156)
split1 <- createDataPartition(ticdata$CARAVAN,
                              p = .7,
                              list = FALSE)

training <- ticdata[split1, ]
other <- ticdata[-split1, ]

set.seed(934)
split2 <-createDataPartition(other$CARAVAN,
                             p = 1/3,
                             list = FALSE)

evaluation <- other[split2, ]
test <- other[-split2, ]

predictors <- names(training)[names(training) != "CARAVAN"]


prop.table(table(training$CARAVAN)) #severe class imbalance (approx %6-%94)

testResults <- data.frame(CARAVAN = test$CARAVAN)
evalResults <- data.frame(CARAVAN = evaluation$CARAVAN)

trainingInd <- data.frame(model.matrix(CARAVAN ~.,
                                       data = training[,-1]))

evaluationInd <- data.frame(model.matrix(CARAVAN ~.,
                                       data = evaluation[,-1]))

testInd <- data.frame(model.matrix(CARAVAN ~.,
                                       data = test[,-1]))


trainingInd$CARAVAN <- training$CARAVAN
evaluationInd$CARAVAN <- evaluation$CARAVAN
testInd$CARAVAN <- test$CARAVAN

isNZV <- nearZeroVar(trainingInd)
noNZVSet <- names(trainingInd)[-isNZV]


##############################################

# SECTION 2 - The Effect of Class Imbalance
# Functions are used to measure performance.

fiveStats <- function(...) c(twoClassSummary(...),
                             defaultSummary(...))

fourStats <- function(data, lev = levels(data$obs),
                      model = NULL){
  accKapp <- postResample(data[, "pred"], data[, "obs"])
  out <- c(accKapp,
           sensitivity(data[, "pred"], data[, "obs"], lev[1]),
           specificity(data[, "pred"], data[, "obs"], lev[2]))
  
  names(out)[3:4] <- c("Sens", "Spec")
  return(out)
}

ctrl <- trainControl(method = "cv",
                     n = 10,
                     classProbs = TRUE,
                     summaryFunction = fiveStats)

ctrlNoProb <- ctrl
ctrlNoProb$summaryFunction <- fourStats
ctrlNoProb$classProbs <- FALSE

set.seed(1410)
registerDoMC(2)
rfFit <- train(CARAVAN ~.,
               data = trainingInd,
               method = "rf",
               ntree = 1000,
               tuneLength = 4,
               metric = "ROC",
               trControl = ctrl)





