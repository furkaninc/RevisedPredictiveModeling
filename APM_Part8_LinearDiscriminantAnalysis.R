####
# CASE STUDY
# (with grant application data)
# for classification

#############################
# load the grant application data

setwd(dir = "~/R/Applied_Predictive_Modeling")
load("grantApplication.RData")

library(caret)
library(tidyverse)
library(AppliedPredictiveModeling)
library(doMC)
library(pROC)
registerDoMC(2)

#######################
# looking at two different ways of split and resample the data
# SVM is used to illustrate the differences.
# full set of predictors is used

pre2008Data <- training[pre2008,]
year2008Data <- rbind(training[-pre2008,], testing)

set.seed(552)
test2008 <- createDataPartition(year2008Data$Class,
                                p = .25)[[1]]

allData <- rbind(pre2008Data, year2008Data[-test2008,])
holdout2008 <- year2008[test2008,]

# Day of the year should be passed again to training & test set
# b/c Day column only ranges from 1 to 31 when it is supposed to 1 to 360+
# To address this issue F.İ re-organized the Day column and override the existing one
daySorted <- training[,c("Jan","Feb","Mar","Apr",
                         "May","Jun","Jul","Aug",
                         "Sep","Oct","Nov","Dec",
                         "Day")]

daySorted <- daySorted %>%
  mutate(DayofYear = case_when(
    Jan == 1  ~ daySorted$Day,
    Feb == 1  ~ daySorted$Day + 31,
    Mar == 1  ~ daySorted$Day + 59,
    Apr == 1  ~ daySorted$Day + 90,
    May == 1  ~ daySorted$Day + 120,
    Jun == 1  ~ daySorted$Day + 151,
    Jul == 1  ~ daySorted$Day + 181,
    Aug == 1  ~ daySorted$Day + 212,
    Sep == 1  ~ daySorted$Day + 243,
    Oct == 1  ~ daySorted$Day + 273,
    Nov == 1  ~ daySorted$Day + 304,
    Dec == 1  ~ daySorted$Day + 334))

training$Day <- daySorted$DayofYear

## use a common tuning grid for both approaches
#svmRadGrid <- expand.grid(sigma = c(.00007, .00009, .0001, .0002),
                          # C = 2^(-3:8))

# Evaluate the model using 10-fold CV
# ctrl0 <- trainControl(method = "cv",
#                       n = 10,
#                       classProbs = TRUE)
# 
# set.seed(477)
# svmFit0 <- train(x = pre2008Data[,fullSet],
#                  y = pre2008Data$Class,
#                  method = "svmRadial",
#                  preProc = c("center", "scale"),
#                  metric = "ROC",
#                  tuneGrid = svmRadGrid,
#                  trControl = ctrl0)
# 

###############################
# Logistic Regression
modelFit <- glm(Class ~ Day,
                data = training[pre2008,],
                family = "binomial")

dataGrid <- data.frame(Day = seq(0, 365, length = 500))
dataGrid$Linear <- 1 - predict(modelFit, dataGrid, type = "response")

linear2008 <- auc(roc(response = training[-pre2008, "Class"],
                      predictor = 1 - predict(modelFit, training[-pre2008,],
                                             type = "response"),
                      levels = rev(levels(training[-pre2008, "Class"]))))

linear2008

modelFit2 <- glm(Class ~ Day + I(Day^2), 
                 data = training[pre2008,], 
                 family = "binomial")
dataGrid$Quadratic <- 1 - predict(modelFit2, dataGrid, type = "response")
quad2008 <- auc(roc(response = training[-pre2008, "Class"],
                    predictor = 1 - predict(modelFit2, 
                                            training[-pre2008,], 
                                            type = "response"),
                    levels = rev(levels(training[-pre2008, "Class"]))))

quad2008

dataGrid <- tidyr::gather(dataGrid, "variable", "value", Linear, Quadratic)

byDay <- training[pre2008, c("Day", "Class")]
byDay$Binned <- cut(byDay$Day, seq(0, 360, by = 5))

#observedProps <- ddply(byDay, .(Binned),
#                       function(x) c(n = nrow(x), mean = mean(x$Class == "successful")))
#(with plyr, old fashion)  

observedProps <- byDay %>%
  group_by(Binned) %>%
  dplyr::summarise(mean = mean(Class == "successful"),
                   n = n()) #with dplyr

observedProps$midpoint <- seq(2.5, 357.5, by = 5)
xyplot(value ~ Day|variable, data = dataGrid,
       ylab = "Probability of A Successful Grant",
       ylim = extendrange(0:1),
       between = list(x = 1),
       panel = function(...)
       {
         panel.xyplot(x = observedProps$midpoint, observedProps$mean,
                      pch = 16., col = rgb(.2, .2, .2, .5))
         panel.xyplot(..., type = "l", col = "black", lwd = 2)
       })

# for the reduced set of factors,
# fit the LogReg model and evaluate...
training$Day2 <- training$Day^2
testing$Day2 <- testing$Day^2
fullSet <- c(fullSet, "Day2")
reducedSet <- c(reducedSet, "Day2")

# create a control object to use
# multiple models so that
# the data splitting is consistent

ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE,
                     index = list(TrainSet = pre2008))
registerDoMC(2)
set.seed(476)
lrFit <- train(x = training[,reducedSet],
               y = training$Class,
               method = "glm",
               metric = "ROC",
               trControl = ctrl)

set.seed(476)
lrFit2 <- train(x = training[,fullSet],
                y = training[,"Class"],
                method = "glm",
                metric = "ROC",
                trControl = ctrl)

lrFit2

# get confusion matrices
confusionMatrix(lrFit, norm = "none")
confusionMatrix(lrFit2, norm = "none")

# get the area under the ROC curve for hold-out set
registerDoMC(3)
lrRoc <- roc(response = lrFit$pred$obs,
             predictor = lrFit$pred$successful,
             levels = rev(levels(lrFit$pred$obs)))

lrRoc2 <- roc(response = lrFit2$pred$obs,
             predictor = lrFit2$pred$successful,
             levels = rev(levels(lrFit2$pred$obs)))

lrImp <- varImp(lrFit, scale = FALSE)
plot(lrImp, top = 15)
plot(lrRoc)

 
#######################################################################################
# Linear Discriminant Analysis (LDA)

# fit the model to the reduced set.
set.seed(476)
registerDoMC(2)
ldaFit <- train(x = training[,reducedSet],
                 y = training$Class,
                 method = "lda",
                 preProc = c("center", "scale"),
                 metric = "ROC",
                 trControl = ctrl)

ldaFit
names(ldaFit)
head(ldaFit$pred)

# Get the ROC curve for LDA model
# remember for ROC we need to relevel the event-nonevent

ldaRoc <- roc(response = ldaFit$pred$obs,
              predictor = ldaFit$pred$successful,
              levels = rev(levels(ldaFit$pred$obs)))

plot(ldaRoc, legacy.axes = TRUE)
auc(ldaRoc)

# Get the confusion matrix for LDA model
ldaCM <- confusionMatrix(ldaFit , norm = "none")
ldaCM

plot(lrRoc, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(ldaRoc, add = TRUE, type = "s", legacy.axes = TRUE)

#PLSDA (partial least square discriminant analysis)

set.seed(476)
plsFit <- train(x = training[,fullSet],
                y = training$Class,
                method = "pls",
                tuneGrid = expand.grid(ncomp = 1:10),
                preProc = c('center','scale'),
                metric = "ROC",
                probMethod = "Bayes",
                trControl = ctrl)

plsFit
plsFit$bestTune
plsFit$pred <- merge(plsFit$pred, plsFit$bestTune) #only keep the final parameter

plsRoc<- roc(response = plsFit$pred$obs,
             predictor = plsFit$pred$successful,
             levels = rev(levels(plsFit$pred$obs)))
auc(plsRoc)

plsCM <- confusionMatrix(plsFit, norm = "none")
plsCM

# use the reduced set for pls
set.seed(476)
plsFit2 <- train(x = training[,reducedSet],
                 y = training$Class,
                 method = "pls",
                 tuneGrid = expand.grid(ncomp = 1:10),
                 preProc = c('center','scale'),
                 metric = "ROC",
                 probMethod = "Bayes",
                 trControl = ctrl)

plsFit2
names(plsFit2)

plsFit2$pred <- plsFit2$pred %>%
  filter(ncomp == 4) # retain only the best tune

plsRoc2 <- roc(response = plsFit2$pred$obs,
               predictor = plsFit2$pred$successful,
               levels = rev(levels(plsFit2$pred$obs)))

auc(plsRoc2)

plsCM2 <- confusionMatrix(plsFit2, norm = "none")
plsCM2

pls.ROC <- plsFit$results
pls.ROC$Descriptor <- "Full Set"
pls2.ROC <- plsFit2$results
pls2.ROC$Descriptor <- "Reduced Set"

plsCompareROC <- data.frame(rbind(pls.ROC, pls2.ROC))
plsCompareROC <- plsCompareROC %>%
  select(-ROCSD, -SensSD, -SpecSD) # remove NA columns.

xyplot(ROC ~ ncomp,
       data = plsCompareROC,
       xlab = "# of Components",
       ylab = "ROC (2008 hold-out data)",
       auto.key = list(columns = 2),
       groups = Descriptor,
       type = "o",
       pch = "o",
       grid = TRUE,
       lwd = 2)

#########################################################################
### Penalized Models
### glmnet

glmGrid <- expand.grid(alpha = c(0, .1, .2, .4,
                                 .6, .8, 1),
                       lambda = seq(.01, .2, length = 40))

set.seed(476)
glmFit <- train(x = training[,fullSet],
                y = training$Class,
                method = "glmnet",
                tuneGrid = glmGrid,
                preProc = c('center', 'scale'),
                metric = "ROC",
                trControl = ctrl)

glmFit

glmnetCM <- confusionMatrix(glmFit, norm = "none")
glmnetCM

glmnet2008 <- merge(glmFit$pred, glmFit$bestTune)

glmnetRoc <- roc(response = glmnet2008$obs,
                 predictor = glmnet2008$successful,
                 levels = rev(levels(glmnet2008$obs)))

glmFit0 <- glmFit
glmFit0$results$lambda <- format(round(glmFit0$results$lambda))

glmPlot <- plot(glmFit,
                plotType = "level",
                cuts = 15,
                scales = list(x = list(rot = 90, cex = .65)))

update(glmPlot,
       ylab = "Mixing Percentage\nRidge <------> Lasso",
       sub = "",
       main = "Area Under the ROC Curve",
       xlab = "Amount of Regularization")

auc(glmnetRoc)
plot(glmnetRoc, legacy.axes = TRUE)

################################################
# Sparse Logistic Regression

set.seed(476)
sparseLDAfit <- train(x = training[,fullSet],
                      y = training$Class,
                      method = "sparseLDA",
                      tuneGrid = expand.grid(lambda = c(.1),
                                             NumVars = c(20, 50, 75, 100, 250, 500, 1000)),
                      preProc = c("center", "scale"),
                      metric = "ROC",
                      trControl = ctrl)

sparseLDAfit

plot(sparseLDAfit,
     scales = list(x = list(log = 10)),
     main = "Sparse LDA Predictors vs ROC",
     ylab = "ROC AUC (2008 hold-out data")

sparseLDAfit$bestTune
head(sparseLDAfit$pred)

spLDA2008 <- merge(sparseLDAfit$pred, sparseLDAfit$bestTune)

spLDACM <- confusionMatrix(sparseLDAfit, norm = "none")
spLDACM

spLDARoc <- roc(response = spLDA2008$obs,
                predictor = spLDA2008$successful,
                levels = rev(levels(spLDA2008$obs)))

auc(spLDARoc)

#########################################################3
# Nearest Shrunken Centroids

set.seed(476)
nscFit <- train(x = training[,fullSet],
                y = training$Class,
                method = "pam",
                preProc = c("center", "scale"),
                tuneGrid = data.frame(threshold = seq(0, 20, length = 25)),
                metric  ="ROC",
                trControl = ctrl)

nscFit
plot(nscFit)

nsc2008 <- merge(nscFit$pred, nscFit$bestTune)
nscCM <- confusionMatrix(nscFit, norm = "none")
nscCM

nscRoc <- roc(response = nsc2008$obs,
              predictor = nsc2008$successful,
              levels = rev(levels(nsc2008$obs)))
auc(nscRoc)
plot(nscRoc, legacy.axes = TRUE)

# all ROC plots comparison:
plot(plsRoc2, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(glmnetRoc, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(ldaRoc, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(lrRoc, type = "s", col = rgb(.2, .2, .2, .2), add = TRUE, legacy.axes = TRUE)
plot(spLDARoc, type = "s", col = rgb(.2, .2, .2, .2), add = TRUE, legacy.axes = TRUE)
plot(nscRoc, type = "s", add = TRUE, legacy.axes = TRUE)

# over for the chapter.
