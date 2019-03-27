
# Applied Predictive Modeling
# Non-linear Classification Models

###################################
library(tidyverse)
library(caret)
library(lattice)
library(pROC)

setwd("~/R/Applied_Predictive_Modeling")
load("grantApplication.RData")

##################################


library(doMC)
registerDoMC(cores = 2)

# Creating a control object
ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     index = list(TrainSet = pre2008),
                     savePredictions = TRUE)


#######################################
# # Mixture Discriminant Analysis (MDA)
set.seed(476)
mdaFit <- train(x = training[,reducedSet],
                y = training$Class,
                method = "mda",
                metric = "ROC",
                tries = 40,
                tuneGrid = expand.grid(subclasses = 1:8),
                trControl = ctrl)


mdaFit
mdaFit$results <- mdaFit$results %>%
  filter(!is.na(ROC))

mdaFit$pred <- merge(mdaFit$pred,
                     mdaFit$bestTune)

mdaCM <- confusionMatrix(mdaFit, norm = "none")
mdaCM

mdaRoc <- roc(response = mdaFit$pred$obs,
              predictor = mdaFit$pred$successful,
              levels = rev(levels(mdaFit$pred$obs)))

auc(mdaRoc)

plot(mdaFit,
     ylab = "ROC AUC [2008 Hold-Out Data]")

plot(mdaRoc, legacy.axes = TRUE)



# Flexible Discriminant Analysis
set.seed(476)

fdaFit <- train(x = training[,reducedSet],
                y = training$Class,
                method = "fda",
                metric= "ROC",
                tuneGrid = expand.grid(degree = 1,
                                       nprune = 2:25),
                trControl = ctrl)

fdaFit
xyplot(ROC ~ nprune|"FDA",
       data = fdaFit$results,
       grid = TRUE,
       type = c("p","l"),
       col.line = "blue",
       cex = 1.5,
       pch  = "o",
       col = "black",
       main = "Flexible Discriminant Analysis Model",
       ylab = "2008 Hold-out Set AUC ROC",
       xlab = "# of Terms")

fdaCM <- confusionMatrix(fdaFit, norm = "none")
fdaCM

fdaROC <- roc(response = fdaFit$pred$obs,
              predictor = fdaFit$pred$successful,
              levels = rev(levels(fdaFit$pred$obs)))

plot(fdaROC, col = "red", legacy.axes = TRUE)
auc(fdaROC)


# Support Vector Machines (SVM)
library(kernlab)

set.seed(201)
sigmaRangeFull <- sigest(as.matrix(training[,fullSet]))
svmRadGridFull <- expand.grid(sigma = as.vector(sigmaRangeFull)[1],
                              C = 2^(-3:4))

set.seed(476)
svmRadFitFull <- train(x = training[, fullSet],
                       y = training[, "Class"],
                       method = "svmRadial",
                       trControl = ctrl,
                       preProc = c('center','scale'),
                       tuneGrid = svmRadGridFull,
                       metric = "ROC")

svmRadFitFull

svmRadFitFull$results %>%
  select(-contains("SD")) -> svmRadFitFull$results

xyplot(ROC ~ C, data = svmRadFitFull$results,
       scales = list(x = list(log = 2)),
       type = c("g","o"),
       col.line = "blue",
       col = "black",
       lwd = 2,
       main = "ROC vs. Cost Value",
       xlab = "",
       ylab = "AUC ROC for 2008 hold-out data")

svmRadROC <- roc(response = svmRadFitFull$pred$obs,
              predictor = svmRadFitFull$pred$successful,
              levels = rev(levels(svmRadFitFull$pred$obs)))

svmRadCM <- confusionMatrix(svmRadFitFull, norm = "none")

plot(svmRadROC, legacy.axes = T)
auc(svmRadROC)
svmRadCM


## -- for reduced set
library(kernlab)
set.seed(202)
sigmaRangeRedu <- sigest(as.matrix(training[,reducedSet]))
svmRadGridRedu <- expand.grid(sigma = as.vector(sigmaRangeRedu)[1],
                              C = 2^ seq(-4, 4))

set.seed(476)
svmRadFitRedu <- train(x = training[, reducedSet],
                       y = training[, "Class"],
                       method = "svmRadial",
                       trControl = ctrl,
                       preProc = c('center','scale'),
                       tuneGrid = svmRadGridRedu,
                       metric = "ROC")

svmRadFitRedu

svmRadFitRedu$results %>%
  select(-contains("SD")) -> svmRadFitRedu$results

xyplot(ROC ~ C, data = svmRadFitRedu$results,
       scales = list(x = list(log = 2)),
       type = c("g","b"),
       col.line = "navy",
       lwd = 2,
       auto.key = TRUE)

svmRedROC <- roc(response = svmRadFitRedu$pred$obs,
                 predictor = svmRadFitRedu$pred$successful,
                 levels = rev(levels(svmRadFitRedu$pred$obs)))

auc(svmRedROC) 

svmReduCM <- confusionMatrix(svmRadFitRedu, norm = "none")
svmReduCM


# -------- polynomial svm
svmPgrid <- expand.grid(scale = c(0.01, 0.005),
                        degree = 1:2,
                        C = 2^(seq(-6, 3, length = 10)))

set.seed(476)
svmPfit <- train(x = training[,reducedSet],
                 y = training$Class,
                 method = "svmPoly",
                 metric = "ROC",
                 preProc = c('center', 'scale'),
                 tuneGrid = svmPgrid,
                 trControl = ctrl,
                 fit = FALSE)

svmPfit

svmPfit$results %>%
  select(-contains("SD")) -> svmPfit$results

ggplot(svmPfit$results, aes(x = C, y = ROC)) +
  theme_light() +
  geom_point() +
  geom_line() +
  facet_grid(scale ~ degree)

svmPfitROC <- roc(response = svmPfit$pred$obs,
                 predictor = svmPfit$pred$successful,
                 levels = rev(levels(svmPfit$pred$obs)))

auc(svmPfit) 

svmPfitCM <- confusionMatrix(svmPfit, norm = "none")
svmPfitCM
