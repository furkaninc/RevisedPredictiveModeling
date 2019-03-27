
# Applied Predictive Modeling by Max Kuhn et al.
# Chapter 1 Intro
##############################################
# Chapter 2: A Short Tour
require(tidyverse)
require(AppliedPredictiveModeling)
data(FuelEconomy)

glimpse(cars2010)

cars2010 <- cars2010 %>%
  arrange(desc(EngDispl))

cars2011 <- cars2011 %>%
  arrange(desc(EngDispl))

# Combine data in one df
cars2010a <- cars2010
cars2010a <- cars2010a %>%
  mutate(Year = "2010 Model Year")

cars2011a <- cars2011
cars2011a <- cars2011a %>%
  mutate(Year = "2011 Model Year")

plotData <- rbind(cars2010a, cars2011a)

library(lattice)
?xyplot
xyplot(FE ~ EngDispl|Year, data = plotData,
       xlab = "Engine Displacement",
       ylab = "Fuel Efficiency (MPG)",
       between = list(x = 1.2))



# Linear Model
require(caret)
set.seed(1)

linmodel <- train(FE ~ EngDispl, data = cars2010,
                  method = "lm",
                  trControl = trainControl(method = "cv", n = 10))
linmodel #see the summary

# Quadratic Model
cars2010$ED2 <- cars2010$EngDispl^2
cars2011$ED2 <- cars2011$EngDispl^2

set.seed(1)
qlmmodel <- train(FE ~ EngDispl + ED2, data = cars2010,
                  method = "lm",
                  trControl = trainControl(method = "cv", n = 10))
qlmmodel




# Mars Model
library(earth)
set.seed(1)
marsFit <- train(FE ~ EngDispl, 
                 data = cars2010,
                 method = "earth",
                 tuneLength = 15,
                 trControl = trainControl(method= "cv"))
marsFit
plot(marsFit)

#Predictions
cars2011$lm <- predict(linmodel, newdata = cars2011)
cars2011$qlm <- predict(qlmmodel, newdata = cars2011) 
cars2011$mars <- predict(marsFit, newdata = cars2011) 

#Get the test performance
?postResample
postResample(pred = cars2011$lm, obs = cars2011$FE) #worst performance
postResample(pred = cars2011$qlm, obs = cars2011$FE) #mediocre performance
postResample(pred = cars2011$mars, obs = cars2011$FE) #best performance

sessionInfo()


xyplot(FE ~ EngDispl|"Linear Model", data = cars2010,
       xlab = "Engine Displacement",
       ylab = "Fuel Efficiency (MPG)",
       panel = function(x, y, ...) {
         panel.xyplot(x , y, ...)
         panel.lines(x, fitted(linmodel), col = "red")
       })


xyplot(FE ~ EngDispl|"Quadratic Model", data = cars2010,
       xlab = "Engine Displacement",
       ylab = "Fuel Efficiency (MPG)",
       grid = TRUE,
       panel = function(x, y, ...) {
         panel.xyplot(x , y, ...)
         panel.lines(x, fitted(qlmmodel), col = "red")
       })

xyplot(FE ~ EngDispl|"MARS", data = cars2010,
       xlab = "Engine Displacement",
       ylab = "Fuel Efficiency (MPG)",
       main = "Multivariate Adaptive Spline",
       grid = TRUE,
       panel = function(x, y, ...) {
         panel.xyplot(x , y, ...)
         panel.lines(x, fitted(marsFit), col = "red")
       })
