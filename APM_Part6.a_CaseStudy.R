
# APM - regression CASE STUDY
################################################

library(caret)
library(AppliedPredictiveModeling)
library(tidyverse)

data(concrete)

theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2

trellis.par.set(theme1)
featurePlot(x = concrete[, -9],
            y = concrete$CompressiveStrength,
            type = c("g", "p", "smooth"),
            between = list(x = 1, y = 1))


# MODEL BUILDING STRATEGY
# there are replicated mixtures,
# so take the average per mixture

# averaged <- ddply(mixtures,
#                   .(Cement, BlastFurnaceSlag, FlyAsh, Water, 
#                     Superplasticizer, CoarseAggregate, 
#                     FineAggregate, Age),
#                   function(x) c(CompressiveStrength = 
#                                   mean(x$CompressiveStrength)))  ###plyr approach


averaged <- mixtures %>%
  group_by(Cement, BlastFurnaceSlag, FlyAsh, Water, Superplasticizer,
           CoarseAggregate, FineAggregate, Age) %>%
  summarise(CompressiveStrength = mean(CompressiveStrength))    ###dplyr approach, recent!



# Split the data and create a control function
set.seed(975)
ind <- createDataPartition(averaged$CompressiveStrength, p = .75, list = FALSE)

trainData <- averaged[ind,]
testData <- averaged[-ind,]

ctrl <- trainControl(method = "repeatedcv", n = 10, repeats = 5,
                     savePredictions = "final")

modForm <- paste0("CompressiveStrength ~ (.)^2 + (Cement^2) + (BlastFurnaceSlag^2) +",
                 "(FlyAsh^2)  + (Water^2) + (Superplasticizer^2)  +",
                 "(CoarseAggregate^2) +  (FineAggregate^2) + (Age^2)")
modForm <- as.formula(modForm)

# Tune different models for the data
library(doMC)
registerDoMC(cores = 3)

# linear regression
set.seed(669)
lmTune <- train(modForm, data = trainData,
                method = "lm",
                trControl = ctrl)

lmTune

# partial least squares
set.seed(669)
plsTune <- train(modForm, data = trainData,
                 method = "pls",
                 preProc = c("center", "scale"),
                 tuneLength = 25,
                 trControl = ctrl)

plsTune
plot(plsTune)

# multivariate adaptive spline
set.seed(669)
earthTune <- train(CompressiveStrength ~., data = trainData,
                   method = "earth",
                   tuneGrid = expand.grid(degree = 1,
                                          nprune = 2:25),
                   trControl = ctrl)

earthTune
plot(earthTune)

# SVM with radial kernel
set.seed(669)
svmTuneRad <- train(CompressiveStrength ~., data = trainData,
                    method = "svmRadial",
                    preProc = c("center", "scale"),
                    tuneLength = 15,
                    trControl = ctrl)

svmTuneRad
plot(svmTuneRad, scales = list(x = list(log = 2)))

# Neural Networks

nnetGrid <- expand.grid(decay = c(0.001, .01, .1), 
                        size = seq(1, 24, by = 4), 
                        bag = FALSE)
set.seed(669)
nnetFit <- train(CompressiveStrength ~ .,
                 data = trainData,
                 method = "avNNet",
                 tuneGrid = nnetGrid,
                 preProc = c("center", "scale"),
                 linout = TRUE,
                 trace = FALSE,
                 maxit = 1000,
                 allowParallel = FALSE,
                 trControl = ctrl)

nnetFit
plot(nnetFit)

# single tree 
set.seed(669)
rpartTune <- train(CompressiveStrength ~., data = trainData,
                   method = "rpart",
                   tuneLength = 10,
                   trControl = ctrl)

rpartTune
plot(rpartTune)

# Bagged Tree
set.seed(669)
treebagTune <- train(CompressiveStrength ~ .,
                    data = trainData,
                    method = "treebag",
                    trControl = ctrl)

treebagTune


# Conditional Inference
set.seed(669)
ctreeTune <- train(CompressiveStrength ~ .,
                  data = trainData,
                  method = "ctree",
                  tuneLength = 10,
                  trControl = ctrl)

ctreeTune
plot(ctreeTune)

# Random Forest
set.seed(669)
rfTune <- train(CompressiveStrength ~ .,
               data = trainData,
               method = "rf",
               tuneLength = 10,
               ntrees = 1000,
               importance = TRUE,
               trControl = ctrl)

rfTune
plot(rfTune)

# Gradient Boosting
gbmGrid <- expand.grid(interaction.depth = seq(1, 7, by = 2),
                       shrinkage = c(.1, .01),
                       n.trees = seq(400, 1000, by = 100),
                       n.minobsinnode = 20)

set.seed(669)
gbmTune <- train(CompressiveStrength ~., data = trainData,
                 method = "gbm",
                 tuneGrid = gbmGrid,
                 trControl = ctrl,
                 verbose = FALSE)

gbmTune
plot(gbmTune)

# Model Tree
registerDoSEQ()

set.seed(669)
modeltreeTune <- train(CompressiveStrength ~ .,
                       data = trainData,
                       method = "M5",
                       trControl = ctrl)

modeltreeTune
plot(modeltreeTune)

# collect the resampling statistics across the models
?resamples

rs <- resamples(list("linearReg" = lmTune,
                     "PLS" = plsTune,
                     MARS = earthTune,
                     SVM = svmTuneRad,
                     "NeuralNet" = nnetFit,
                     "CondInfTree" = ctreeTune,
                     "BaggedTree" = treebagTune,
                     "BoostedTree" = gbmTune,
                     "RandomForest" = rfTune))
parallelplot(rs)
parallelplot(rs, metric = "Rsquared")

# get several model performance and compare them
nnetPred <- predict(nnetFit, newdata = testData)
gbmPred <- predict(gbmTune, newdata = testData)
rfPred <- predict(rfTune, newdata = testData)

testResults <- as.data.frame(rbind(postResample(nnetPred, testData$CompressiveStrength),
                                   postResample(gbmPred, testData$CompressiveStrength),
                                   postResample(rfPred, testData$CompressiveStrength)))

testResults <- testResults %>%
  mutate(Model = c("Neural Net", "Boosted Tree", "Random Forest")) %>%
  arrange(RMSE)

testResults  

ggplot(testResults, aes(x = Model, y = RMSE)) +
  theme_bw() +
  geom_point(alpha = 1, size = 5, aes(color = Model)) +
  labs(title = "Model Performance Comparison with respect to RMSE",
       xlab = "Model",
       ylab = "RMSE",
       subtitle = "prediction accuracy of 3 different models")

Scores <- data.frame(original = testData$CompressiveStrength,
                     nnet = nnetPred,
                     randomForest = rfPred,
                     boostedTree = gbmPred )

trellis.par.set(theme1)

xyplot(Scores$original ~ Scores$nnet| "nnet predictions and test values",
       grid = TRUE,
       type = c("p", "g"),
       xlab = "predictions",
       ylab = "test values",
       color = "gray")

Scores <- Scores %>%
  mutate(nnetRes = nnet - original)

xyplot(Scores$nnetRes ~ Scores$nnet| "nnet predictions and test values",
       grid = TRUE,
       type = c("p", "g"),
       xlab = "predictions",
       ylab = "residuals",
       col = "navy",
       panel = function(x, y, ...){
         panel.xyplot(x, y, ...)
         panel.abline(h = 0, lty = 2, col = "red")
       })

##########################################################3
# Optimizing Compressive Strength

library(proxy)

### Create a function to maximize compressive strength* while keeping
### the predictor values as mixtures. Water (in x[7]) is used as the 
### 'slack variable'. 

### * We are actually minimizing the negative compressive strength

# check the ranges and see if there is inconsistencies

map(trainData[1:6], ~ range(.))
modelPrediction <- function(x, mod, limit = 2500)
{
  if(x[1] < 0 | x[1] > 1) return(10^38)
  if(x[2] < 0 | x[2] > 1) return(10^38)
  if(x[3] < 0 | x[3] > 1) return(10^38)
  if(x[4] < 0 | x[4] > 1) return(10^38)
  if(x[5] < 0 | x[5] > 1) return(10^38)
  if(x[6] < 0 | x[6] > 1) return(10^38)
  
  #predictors add up to 1, check it
  x <- c(x, 1 - sum(x))
  
  #check the water range
  if(x[7] < 0.05) return(10^38)
  
  tmp <- as.data.frame(t(x))
  names(tmp) <- c('Cement','BlastFurnaceSlag','FlyAsh',
                  'Superplasticizer','CoarseAggregate',
                  'FineAggregate', 'Water')
  tmp$Age <- 28
  -predict(mod, tmp)
}

# get mixtures at 28 days
subTrain <- trainData %>%
  filter(Age == 28)

# center and scale the data 
# to use dissimilarity sampling
pp1 <- preProcess(subTrain[,-(8:9),],
                  c("center", "scale"))

scaledTrain <- predict(pp1, newdata = subTrain[1:7])

# Randomly select a few mixtures as a starting pool

set.seed(91)
startMixture <- sample(1:nrow(subTrain), 1)
starters <- scaledTrain[startMixture, 1:7]
pool <- scaledTrain

?maxDissim
index <- maxDissim(starters, pool, 14)
startPoints <- c(startMixture, index)

starters <- subTrain[startPoints, 1:7]
startingValues <- starters[, -4] #remove water

# for each starting mixture optimize the gbm model
# usng a simplex search routine

gbmResults <- startingValues
gbmResults$Water <- NA
gbmResults$Prediction <- NA

for(i in 1:nrow(gbmResults)){
  results <- optim(unlist(gbmResults[i, 1:6]),
                          modelPrediction,
                          method = "Nelder-Mead",
                          control = list(maxit = 5000),
                          mod = gbmTune)
  gbmResults$Prediction[i] <- -results$value
  gbmResults[i, 1:6] <- results$par
}

gbmResults$Water <- 1 - apply(gbmResults[, 1:6], 1, sum)

gbmResults <- gbmResults %>%
  filter(Prediction > 0,  Water > .02)

# take top 3 with respect to Prediction column
gbmResults <- gbmResults %>%
  arrange(desc(Prediction)) %>%
  top_n(3)

gbmResults$Model <- "BoostingTree"

#######################################
# Run the same process for Neural Network
nnetResults <- startingValues
nnetResults$Water <- NA
nnetResults$Prediction <- NA

for(i in 1:nrow(nnetResults))
{
  results <- optim(unlist(nnetResults[i, 1:6,]),
                   modelPrediction,
                   method = "Nelder-Mead",
                   control=list(maxit=5000),
                   mod = nnetFit)
  nnetResults$Prediction[i] <- -results$value
  nnetResults[i,1:6] <- results$par
}
nnetResults$Water <- 1 - apply(nnetResults[,1:6], 1, sum)

nnetResults <- nnetResults %>%
  filter(Prediction > 0, Water > .02)

nnetResults <- nnetResults %>%
  arrange(desc(Prediction)) %>%
  top_n(3)

nnetResults$Model <- "NNet"




#############################################
# Convert the  predicted mixtures to PCA space and plot

pp2 <- preProcess(subTrain[, 1:7],  method = "pca")
pca1 <- predict(pp2, newdata = subTrain[, 1:7])

pca1$Data <- "Training Set"
pca1$Data[startPoints] <- "Starting Values"

pca3 <- predict(pp2, newdata = gbmResults[, names(subTrain[1:7])])
pca3$Data <- "Boosted Tree"

pca4 <- predict(pp2, nnetResults[, names(subTrain[1:7])])
pca4$Data <- "Neural Network"

pcaData <- rbind(pca1, pca3, pca4)


