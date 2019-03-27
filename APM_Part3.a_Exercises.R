

#####################################################
# Over Fitting Chapter
# Exercise

require(tidyverse)
require(caret)
require(AppliedPredictiveModeling)

data(oil)
table(oilType)

oil <- cbind(fattyAcids, oilType)

oil[!complete.cases(oil),] #no missing data
nearZeroVar(oil) #no predictor with near-zero-var

head(oil)
oilCor <- cor(oil[,-8])
library(corrplot)
?corrplot
corrplot(oilCor)

highCor <- findCorrelation(oilCor, cutoff = .8) #column 4 has linearly dependent, but keep it
glimpse(oil)

# let's see the frequency of original data vs random sample
histogram(~oilType, data = oil, type = "count",
          col = "gray",
          main = "original data")


oilSamples <- sample(oil$oilType, 60)

histogram(~oilSamples, type = "count",
          col = "gray",
          main = "random 60 sample from original set")

# create a stratified split
set.seed(321)
index <- createDataPartition(oil$oilType, p = .75, list = FALSE)
oilTrain <- oil[index,]
oilTest <- oil[-index,]

#let's try some models, first complicated then simple ones
library(kernlab)
sigVals <- sigest(oilType ~., data = oilTrain,
                  frac = 1, na.action = na.fail)

sigVals

svmTuneGrid <- data.frame(sigma = rep(as.vector(sigVals)[1], n =10), C = 2^(-2:7)) 

ctrl <-trainControl(method = "repeatedcv", n = 10, 
                    repeats = 5)
library(doMC)
registerDoMC(cores = 3)

set.seed(1989)
svmFit <- train(oilType ~., data = oilTrain,
                method = "svmRadial",
                trControl = ctrl,
                preProc = c("center", "scale"),
                tuneGrid = svmTuneGrid,
                verbose = FALSE)

svmFit
plot(svmFit, scales = list(x = list(log = 2)),
     main = "CV Accuracy and Cost Values",
     grid = TRUE)

svmFit$resample

predictedSvm <- predict(svmFit, newdata = oilTest)
confusionMatrix(predictedSvm, oilTest$oilType)

# an LDA model
set.seed(1989)
ldaFit <- train(oilType ~., data = oilTrain,
                method = "lda",
                verbose = FALSE)

predictedLDA <- predict(ldaFit, newdata = oilTest)
confusionMatrix(predictedLDA, oilTest$oilType)

binom.test(20, 20)

############################################################
# Exercise 2 with "permeability" dataset 
require(tidyverse)
require(caret)
require(AppliedPredictiveModeling)

data("permeability")

fingerprints <- as.data.frame(fingerprints)
glimpse(head(fingerprints))

fingerprints %>%
  select_if(~ !is.numeric(.)) #no non-numeric columns (purrr::map)

#drop near-zero-var columns
zv <- nearZeroVar(fingerprints)
fingerprints <- fingerprints[, -zv]

#drop high correlation columns to avoid linear dependencies
fingerCor <- cor(fingerprints)
highCor <- findCorrelation(fingerCor, cutoff = 0.8) #drop them
fingerprints <- fingerprints[, -highCor]

fingerprints <- cbind(fingerprints, permeability)

#explore the response via visualisations
histogram(~permeability, data = fingerprints,
                       type = "count",
                       col = "gray",
                       main = "permeability frequency") #right skewed


histogram(~X6, data = fingerprints,
          type = "count",
          col = "gray")


range(fingerprints$permeability)
summary(fingerprints$permeability)

#pre-process
#all predictors are actually factors, either 0 or 1. check it.
apply(fingerprints, 2, function(x) length(unique(x)))
map(fingerprints,  ~ length(unique(.))) #same thing with purrr::map

predictors <- names(fingerprints)[1:70]
fingerprints[predictors] <- map(fingerprints[predictors], factor)

head(str(fingerprints))

#build up some models
set.seed(321)
index <- createDataPartition(fingerprints$permeability, 
                             p = 0.75,
                             list = FALSE)

fingerTrain <- fingerprints[index, ]
fingerTest <- fingerprints[-index,]

# random forest
tuneRf <- expand.grid(mtry = 1:50)
    
library(doMC)
registerDoMC(3)
set.seed(1989)
rfFit <- train(permeability ~., data = fingerTrain,
                method = "rf",
                tuneGrid = tuneRf,
                verbose = FALSE)

names(rfFit)
rfFit
rfFit$results
rfFit$bestTune
plot(rfFit)

predictedRf <- predict(rfFit, newdata = fingerTest)

RMSErf <- sqrt(sum((predictedRf - fingerTest$permeability))^2)
RMSErf
cbind(predictedRf, fingerTest$permeability)

# SVM
ctrl <- trainControl(method = "cv", n = 10,
                     savePredictions = "final")

set.seed(1989)
svmFit <- train(permeability ~., data = fingerTrain,
                method = "svmRadial",
                trControl = ctrl,
                tuneLength = 10,
                verbose = FALSE)

svmFit
svmFit$resample
svmFit$bestTune
svmFit$results
plot(svmFit, scales = list(x = (list(log = 2))))

predictedSVM <- predict(svmFit, newdata = fingerTest)            
RMSEsvm <- sqrt(sum((predictedSVM - fingerTest$permeability))^2)
RMSEsvm
cbind(predictedSVM, fingerTest$permeability)


xyplot(permeability ~ X6, data = fingerprints,
       grid = TRUE,
       type = "p",
       col = "navy")

xyplot(permeability ~ X11, data = fingerprints,
       grid = TRUE,
       type = "p",
       col = "darkgreen")
