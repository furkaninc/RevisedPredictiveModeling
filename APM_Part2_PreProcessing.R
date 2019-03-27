# Applied Predictive Modeling
# Chapter 2: Data Pre-Processing

require(tidyverse)
library(AppliedPredictiveModeling)
data(segmentationOriginal)

glimpse(segmentationOriginal)
#############################################################
# Data Transformation
# Center And Scaling

#To center a predictor variable, the average predictor
#value is subtracted from all the values. As a result of centering, the predictor
#has a zero mean. Similarly, to scale the data, each value of the predictor
#variable is divided by its standard deviation. Scaling the data coerce the
#values to have a common standard deviation of one.
#SUMMARY: (after centering, mean = 0; after scaling sd = 1 )

glimpse(segmentationOriginal)
segTrain <- segmentationOriginal %>%
  filter(Case == "Train")

segTrainX <- segTrain[, -c(1,2,3)] #remove identifier columns
segTrainClass <- segTrain$Class

## The column VarIntenCh3 measures the standard deviation of the intensity
## of the pixels in the actin filaments

max(segTrainX$VarIntenCh3)/min(segTrainX$VarIntenCh3)
# if > 20, the predictor generally has skewness, returns 870.88

library(e1071)
skewness(segTrainX$VarIntenCh3) #2.39 skewed.

library(caret)

# use PreProcess function to transform for skewness
segPP <- preProcess(segTrainX, method = "BoxCox") #transformation model
segTrainTrans <- predict(segPP, newdata = segTrainX) #apply the transformations

# Results for a single predictor
names(segPP)
segPP$bc$VarIntenCh3 #gives summary of boxcox transformation on the VarıntenCh3 variable

histogram(~segTrainX$VarIntenCh3,
          xlab = "Natural Units",
          type = "count",
          col = "gray")

histogram(~log(segTrainX$VarIntenCh3),
          xlab = "Log Units",
          ylab = "",
          type = "count",
          col = "gray")

segPP$bc$PerimCh1

histogram(~segTrainX$PerimCh1,
          xlab = "Natural Units",
          type = "count",
          col = "gray")

histogram(~PerimCh1, data = segTrainX,
          xlab = "Transformed Data (boxcox)",
          ylab = "",
          type = "count",
          col = "gray")
##############################################
# Data Transformations for Multiple Predictors
pr <- prcomp(~AvgIntenCh1 + EntropyIntenCh1,
           data = segTrainTrans,
           scale. = TRUE)

transparentTheme(pchSize =  .7, trans = .3)

xyplot(AvgIntenCh1 ~ EntropyIntenCh1,
       data = segTrainTrans,
       groups = segTrain$Class,
       xlab = "Channel 1 Fiber Width",
       ylab = "Intensity Entropy Channel 1",
       auto.key = list(columns = 2),
       type = c("p", "g"),
       main = "Original Data",
       aspect = 1)

names(pr)
head(pr$x)

xyplot(PC2 ~ PC1,
       data = as.data.frame(pr$x),
       groups = segTrain$Class,
       xlab = "Principal Component #1",
       ylab = "Principal Component #2",
       main = "Transformed Data",
       xlim = extendrange(pr$x),
       ylim = extendrange(pr$x),
       type = c("p", "g"),
       aspect = 1)

# Appliying PCA to the entire set of predictors

isZV <- apply(segTrainX, 2, function(x) length(unique(x)) == 1)
segTrainX <- segTrainX[, !isZV] #remove the predictors with a single value



segPP <- preProcess(segTrainX,
                    method = c("BoxCox", "center", "scale"))

segTrainTrans <- predict(segPP, newdata = segTrainX)

segPCA <- prcomp(segTrainTrans, center = TRUE, scale. = TRUE)
names(segPCA)
head(segPCA$x)
plot(segPCA) # 4 PC seem suitable to be retained.

# plot a scatterplot matrix of first three principal components
transparentTheme(pchsize = .8, trans = .3)

panelRange <- extendrange(segPCA$x[, 1:3])
splom(as.data.frame(segPCA$x[, 1:3]),
      groups = segTrainClass,
      type = c("p", "g"),
      as.table = TRUE,
      auto.key = list(columns = 2),
      prepanel.limits = function(x) panelRange)

####################################################
# Removing Variables
segCorr <- cor(segTrainTrans)

library(corrplot)
corrplot(segCorr, order = "hclust", tl.cex = .35)

## caret's findCorrelation function is used to identify columns to remove.
highCorr <- findCorrelation(segCorr, .75)


###############################################################
# Dummy Variables
data(cars)
type <- c("convertible", "coupe", "hatchback", "sedan", "wagon")

cars$Type <- factor(apply(cars[, 14:18], 1, function(x) type[which(x == 1)]))
carSubset <- cars[sample(1:nrow(cars), 20), c(1, 2, 19)]
head(carSubset)
levels(carSubset$Type)

simpleMod <- dummyVars(~Mileage + Type,
                      data = carSubset,
                      levelsOnly = TRUE)
simpleMod

withInteraction <- dummyVars(~Mileage + Type + Mileage:Type,
                             data = carSubset,
                             levelsOnly = TRUE)
withInteraction
predict(withInteraction, head(carSubset))




# EXERCISES #
library(tidyverse)
library(mlbench)
data(Glass)
str(Glass)

nearZeroVar(Glass)


Glass[!complete.cases(Glass),] #no missing data
Glass <- Glass[,-10]
maxVal <- as.numeric(apply(Glass, 2, function(x) max(x)))
minVal <- as.numeric(apply(Glass, 2, function(x) min(x)))
skewControl <- maxVal/minVal
skewControl #check histograms

skewDetect <- data.frame(Preditors = names(Glass), SkewControl = skewControl)

#Histograms
histogram(~RI, data = Glass,
          type = "count",
          col = "gray") #looks fine

histogram(~Na, data = Glass,
          type = "count",
          col = "gray") #fine

histogram(~Mg, data = Glass,
          type = "count",
          col = "gray") # a bit left-skewed

histogram(~Al, data = Glass,
          type = "count",
          col = "gray") # just a bit right skewed

histogram(~Si, data = Glass,
          type = "count",
          col = "gray") # fine

histogram(~K, data = Glass,
          type = "count",
          col = "gray") # outlier detected

histogram(~Ca, data = Glass,
          type = "count",
          col = "gray") #fine

histogram(~Ba, data = Glass,
          type = "count",
          col = "gray") # severely right skewed

histogram(~Fe, data = Glass,
          type = "count",
          col = "gray") # severely right skewed

# looks like Ba, Fe right skewed...

pairs(Glass)

#BoxCox transformation for Ba and K parameters
isZV <- apply(Glass, 2, function(x) length(unique(x)) == 1)
Glass <- Glass[,!isZV] #no zero-var column detected
library(caret)
glasspp <- preProcess(Glass[,c("K","Mg")],
                      method = c("YeoJohnson", "center", "scale"))

glasstrsf <- predict(glasspp, newdata = Glass)


histogram(~K, data = glasstrsf,
          col = "gray",
          type = "count",
          main = "transformed")

histogram(~Mg, data = glasstrsf,
          col = "gray",
          type = "count",
          main = "transformed")


glcor <- cor(glasstrsf)
library(corrplot)
corrplot(glcor,  order = "hclust", tl.cex = .35)
highcor <- findCorrelation(glcor, cutoff = 0.8)
names(Glass)
highcor # Ca can be discarded since it displays high in-between predictor correlations!

#####################################################################################
