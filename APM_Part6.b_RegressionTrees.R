
# Applied Predictive Modeling
# Regression Trees
######################################################################
# Simple Regression Trees

library(caret)
library(tidyverse)
library(AppliedPredictiveModeling)

data(solubility)

# create a control function,
# create fold assignments explicitly

set.seed(100)
indx <- createFolds(solTrainY, returnTrain = TRUE)
ctrl <- trainControl(method = "cv", n = 10,
                     index = indx)

# Basic Regression Trees

library(rpart)

# party package accepts formula method
# unify the response with dataset

trainData <- solTrainXtrans
trainData$y <- solTrainY


xyplot(y ~ NumCarbon|"an analysis on # of C", data = trainData,
       grid = TRUE,
       xlab = "# of carbon atoms",
       ylab = "solubility",
       main = "Response and No of C atoms",
       type = c("p", "g", "smooth"))




# rpStump <- rpart(y ~., data = trainData,
#                  control = rpart.control(maxdepth = 1))
# 
# rpSmall <- rpart(y ~., data = trainData,
#                  control = rpart.control(maxdepth = 2))
# 

partiStump <- ctree(y ~., data = trainData,
                    controls = ctree_control(maxdepth = 1))

partiSmall <- ctree(y ~., data = trainData,
                    controls = ctree_control(maxdepth = 2))

plot(partiStump)
plot(partiSmall)
# Tune the model

set.seed(100)
cartTune <- train(x = solTrainXtrans, y = solTrainY,
                  method = "rpart",
                  tuneLength = 25,
                  trControl = ctrl)

cartTune
plot(cartTune, scales = list(x = list(log = 10))) #best complexity @ c =~ 0.003



# for nice plots, attach partykit package
library(partykit)
cartTree <- as.party(cartTune$finalModel)
plot(cartTree)

# get the variable importance
# 'competes' argument controls whether variables not used in splits
# should be included...

cartImp <- varImp(cartTune, scale = FALSE, competes = FALSE)
plot(cartImp, top = 25, col = "red")

# save the predictions

testResults <- data.frame(original = solTestY,
                          CART = predict(cartTune, newdata = solTestXtrans))

postCART <- postResample(obs = solTestY, pred = testResults$CART)

# for c = 0.01 (using one standart error rule), we can plot a smaller tree

smallGrid <- expand.grid(cp = 0.01)

set.seed(100)
smallTune <- train(x = solTrainXtrans,
                   y = solTrainY,
                   method = "rpart",
                   tuneGrid = smallGrid,
                   trControl = ctrl)

smallCart <- as.party(smallTune$finalModel)
plot(smallCart) # more interpretable

testResults$simpleCART <- predict(smallTune, newdata = solTestXtrans)
postSimple <- postResample(obs = solTestY, pred = testResults$simpleCART)

# treat the significance threshold as a tuning parameter

cGrid <- data.frame(mincriterion = sort(c(.95, seq(.75, .99, length = 15))))

library(doMC)
registerDoMC(3)
set.seed(100)

ctreeTune <- train(x = solTrainXtrans,
                   y = solTrainY,
                   method = "ctree",
                   tuneGrid = cGrid,
                   trControl = ctrl)


ctreeTune 
dev.off() # to reset the graphic messed up earlier
plot(ctreeTune)
ctreeTune$finalModel

testResults$ctreeCART <- predict(ctreeTune, newdata = solTestXtrans)
postCtree <- postResample(obs = solTestY, pred = testResults$ctreeCART)

# see the prediction RMSE for each tree.
rmseCompare <- rbind(postCART, postSimple, postCtree)
rmseCompare

####################################################################################
# Regression Model Trees

# Tune the model tree. method = "M5" tunes over the
# tree and rule-based versions of the model. M = 10 is
# also pass in as a control to make sure that
# there are larger terminal nodes for the regression models

set.seed(100)
m5Tune <- train(x = solTrainXtrans, y = solTrainY,
                method = "M5",
                trControl = ctrl,
                control = Weka_control(M = 10))


m5Tune
plot(m5Tune)
m5Tune$finalModel 
plot(m5Tune$finalModel)

#Show the rule-based model too
ruleFit <- M5Rules(y~., data = trainData, control = Weka_control(M = 10))
ruleFit



###########################################################################3

# Bagged Trees
library(doMC)
registerDoMC(cores = 3)
set.seed(100)

treebagTune10<- train(x = solTrainXtrans, y = solTrainY,
                       method = "treebag",
                       nbagg = 10,
                       trControl = ctrl)


set.seed(100)
treebagTune20 <- train(x = solTrainXtrans, y = solTrainY,
                       method = "treebag",
                       nbagg = 20,
                       trControl = ctrl)



set.seed(100)
treebagTune50 <- train(x = solTrainXtrans, y = solTrainY,
                     method = "treebag",
                     nbagg = 50,
                     trControl = ctrl)


treebagTunes <- as.data.frame(rbind(treebagTune10, treebagTune20,
                                    treebagTune50))
plot(treebagTunes)
#############################################################3

# Random Forests

mtryGrid <- data.frame(mtry = seq(50, ncol(solTestXtrans)/1.5, length = 3))

library(doMC)
registerDoMC(2)

set.seed(100)
rfTune <- train(x = solTrainXtrans, y = solTrainY,
                method = "rf",
                ntrees = 401,
                trControl = ctrl,
                tuneGrid = mtryGrid,
                importance = TRUE)

rfTune
plot(rfTune)

rfImp <- varImp(rfTune, scale = FALSE)
plot(rfImp, top = 15)

# tune RF with oob estimate
ctrlOOB <- trainControl(method = "oob")

library(doMC)
registerDoMC(2)

set.seed(100)
rfTuneOOB <- train(x = solTrainXtrans, y = solTrainY,
                   method = "rf",
                   ntrees = 401,
                   trControl = ctrlOOB,
                   tuneGrid = mtryGrid,
                   importance = TRUE)

rfTuneOOB
plot(rfTuneOOB)

# Tune the conditional inference model
library(doMC)
registerDoMC(2)

set.seed(100)
condrfTune <- train(x = solTrainXtrans, y = solTrainY,
                   method = "cforest",
                   controls = cforest_classical(ntree = 401),
                   trControl = ctrl,
                   tuneGrid = mtryGrid)
                   

# out-of-bag estimate on conditional inference
library(doMC)
registerDoMC(2)

set.seed(100)
condrfTune <- train(x = solTrainXtrans, y = solTrainY,
                    method = "cforest",
                    controls = cforest_unbiased(ntree = 401),
                    trControl = ctrlOOB,
                    tuneGrid = mtryGrid)

#####################################################################
# BOOSTING:

gbmGrid <- expand.grid(interaction.depth = seq(1, 7, by= 2),
                       shrinkage = c(0.1, 0.01),
                       n.trees = seq(100, 800, by = 50),
                       n.minobsinnode = 20) #Integer specifying the min
                                       # number of observations in the
                                       # terminal nodes

clockNow <- Sys.time()
set.seed(100)
gbmTune <- train(x = solTrainXtrans, y = solTrainY,
                 method = "gbm",
                 trControl = ctrl,
                 tuneGrid = gbmGrid,
                 verbose = FALSE)

TimeTrain <- Sys.time() - clockNow

gbmTune
plot(gbmTune)

(summary(gbmTune, plotit = TRUE, normalize = FALSE)) # to see var importance

#############################################################################
# Compare Random Forest and Stochastic Boosting Performance

ensembleResults <- data.frame(original = solTestY,
                              rfPred = predict(rfTune, newdata = solTestXtrans),
                              gbmPred = predict(gbmTune, newdata = solTestXtrans))

ensembleResults
