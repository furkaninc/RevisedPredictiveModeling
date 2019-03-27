
# From the book Applied Predictive Modeling
##
# Case Study!
# Predicting Cognitive Impairment
# (Alzheimer Disease)

###
library(tidyverse)
library(caret)

library(AppliedPredictiveModeling)
data(AlzheimerDisease)

head(predictors)

# the baseline set
base1 <- c("Genotype", "age", "tau",
           "p_tau", "AB_42", "male")

# the set of new assays
newAssays <- colnames(predictors)
newAssays <- newAssays[!(newAssays %in% c("Class", base1))]

# Decompose the genotype factor into binary dummy variables

predictors$E2 <- predictors$E3 <- predictors$E4 <- 0
predictors$E2[grepl("2", predictors$Genotype)] <- 1
predictors$E3[grepl("3", predictors$Genotype)] <- 1
predictors$E4[grepl("4", predictors$Genotype)] <- 1

genotype <- predictors$Genotype

# Partition the Data
set.seed(730)
split <- createDataPartition(diagnosis, p = .8, list = FALSE)

adData <- predictors
adData$Class <- diagnosis

trainSet <- adData[split, ]
testSet <- adData[-split, ]

predVars <- names(adData)[!names(adData) %in% c("Class", "Genotype")]

# This Summar Function is used to evaluate the models.
fiveStats <- function(...){
  c(twoClassSummary(...), defaultSummary(...))
}

# Create CV files to use with different functions
set.seed(104)
index <- createMultiFolds(trainSet$Class, times = 5)

# the candidate set of the number of predictors to evaluate
varSeqRF <- c(6, 10, 14)

library(doMC)
registerDoMC(2)

# using recursive feature elimination
ctrl <- rfeControl(method = "repeatedcv",
                   n = 10,
                   repeats = 5,
                   saveDetails = FALSE,
                   index = index,
                   returnResamp = "final")

fullCtrl <- trainControl(method = "repeatedcv",
                         repeats = 5,
                         summaryFunction = fiveStats,
                         classProbs = TRUE,
                         savePredictions = TRUE,
                         index = index)

# The correlation matrix of the new data
predCor <- cor(trainSet[, newAssays])

library(RColorBrewer)
coloring <- c(rev(brewer.pal(7, "Blues")),
              brewer.pal(7, "Reds"))

library(corrplot)
corrplot(predCor,
         order = "hclust",
         tl.pos = "n", addgrid.col = rgb(1, 1, 1, .01),
         col = colorRampPalette(coloring)(51))


# Fit a series of models using full set of predictors
set.seed(721)
rfFull <- train(x = trainSet[, predVars],
                y = trainSet$Class,
                method = "rf",
                metric = "ROC",
                tuneGrid = expand.grid(mtry = floor(sqrt(length(predVars)))),
                ntree = 1000,
                trControl = fullCtrl)

rfFull
head(rfFull$pred)

confusionMatrix(rfFull, norm = "none")

library(pROC)

rfPred <- predict(rfFull, newdata = testSet, type = "prob")
rfRocFull <- roc(testSet$Class,
                 rfPred[, 1])

auc(rfRocFull)
ggroc(rfRocFull, legacy.axes = TRUE,
      linetype = 2)


# LDA

set.seed(721)
ldaFull <- train(x = trainSet[, predVars],
                 y = trainSet$Class,
                 method = "lda",
                 metric = "ROC",
                 tol = 1.0e-12,
                 trControl = fullCtrl)

ldaFull


ldaPred <- predict(ldaFull, newdata = testSet, type = "prob")
ldaRoc <- roc(response = testSet$Class,
              predictor = ldaPred[, 1])

auc(ldaRoc)
ggroc(ldaRoc, legacy.axes = T,
      linetype = 2, color = "blue") + theme_minimal()

set.seed(721)
svmFull <- train(trainSet[, predVars],
                 trainSet$Class,
                 method = "svmRadial",
                 metric = "ROC",
                 tuneLength = 12,
                 preProc = c("center", "scale"),
                 trControl = fullCtrl)
svmFull

set.seed(721)
nbFull <- train(trainSet[, predVars],
                trainSet$Class,
                method = "nb",
                metric = "ROC",
                trControl = fullCtrl)
nbFull

lrFull <- train(trainSet[, predVars],
                trainSet$Class,
                method = "glm",
                metric = "ROC",
                trControl = fullCtrl)
lrFull

set.seed(721)
knnFull <- train(trainSet[, predVars],
                 trainSet$Class,
                 method = "knn",
                 metric = "ROC",
                 tuneLength = 20,
                 preProc = c("center", "scale"),
                 trControl = fullCtrl)
knnFull

## Now fit the RFE versions. To do this, 
## the 'functions' argument of the rfe()
## object is modified to the approproate functions.

ctrl$functions <- rfFuncs
ctrl$functions$summary <- fiveStats

set.seed(721)
rfRFE <- rfe(trainSet[, predVars],
             trainSet$Class,
             sizes = varSeqRF,
             metric = "ROC",
             ntree = 1000,
             rfeControl = ctrl)

names(rfRFE)
rfRFE$optsize
rfRFE$bestSubset
rfRFE$optVariables

rfRFE

rfRFEpred <- predict(rfRFE, newdata = testSet, type = "prob")
confusionMatrix(rfRFE, testSet$Class)

