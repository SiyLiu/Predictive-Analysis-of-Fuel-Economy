library(boot)
library(glmnet)
library(ISLR)
library(MASS)
library(leaps)
library(splines)

setwd("L:\\3rdSemesterHere\\DM\\Project1")
train = read.csv('train.csv')

# Delect abnormal values
train = train[train$IntakeValvePerCyl>0,]
train = train[train$ExhaustValvesPerCyl>0,]
train = train[train$NumCyl!=16,]
test = read.csv('test.csv')
test.result =read.csv('test_result.csv')

# Variables' classes check
lapply(train,class)
# Exclude ID from the dataset
train = train[,-1]
test = test[,-1]
test.result = test.result[,-1]
attach(train)

# Factorize categorical variables
train[,c(3,5,7,8,9,12,13,14,15)]=lapply(train[,c(3,5,7,8,9,12,13,14,15)],as.factor)
test[,c(3,4,6,7,8,11,12,13,14)]=lapply(test[,c(3,4,6,7,8,11,12,13,14)],as.factor)
test.result[,c(3,5,7,8,9,12,13,14,15)]=lapply(test.result[,c(3,5,7,8,9,12,13,14,15)],as.factor)

# Assumption check -- If FE is normal distributed
par(mfrow= c(1,3))
hist(train$FE,20,freq = FALSE,main = "Histogram of FE",
     xlim = c(min(train$FE),max(train$FE)),xlab = "FE")
lines(density(train$FE,adjust = 1),col = "darkgreen",lwd =2 )
qqnorm(train$FE,main = "Q-Q Plot for FE")
qqline(train$FE)
boxplot(train$FE,main = "Boxplot for FE")

hist(log(train$FE),xlim = c(min(log(train$FE)),max(log(train$FE))),
     main = "Histogram for log(FE)",xlab = "log(FE)",freq = FALSE)
lines(density(log(train$FE),adjust = 2), lwd = 2,col = "darkgreen")
qqnorm(log(train$FE), main = "Q-Q Plot for log(FE)")
qqline(log(train$FE))

boxplot(log(train$FE),main = "Boxplot for log(FE)")

## Simple Linear Regression
# 10 fold CV
k = 10
len = floor(length(FE)/k)
pd.error = vector()
for (i in 1:k){
  ind = (len*i-len+1) :(len*i)
  training = train[-ind,]
  testing = train[ind,]
  #model technique
  reg.k = lm(log(FE)~., training)
  pd.error[i] = mean((testing$FE - exp(predict(reg.k, testing)))^2)
}
cv.error.SLR = sqrt(mean(pd.error)) # 3.256
fit.SLR = lm(log(FE)~.,data = train)
test.SLR= predict(fit.SLR,test.result)
error.SLR = sqrt(mean((exp(test.SLR)-test.result$FE)^2))#3.763

## Variable Selection -- Stepwise
# 10-fold CV
k = 10
len = floor(length(FE)/k)
pd.error = vector()
for (i in 1:k){
  ind = (len*i-len+1) :(len*i)
  training = train[-ind,]
  testing = train[ind,]
  #model technique
  regfit.null = lm(log(FE)~1, data=training)
  regfit.full = lm(log(FE)~., data=training)
  reg.k = step(regfit.null, scope = list(lower = regfit.null, upper = regfit.full),direction = "both")
  pd.error[i] = mean((testing$FE - exp(predict(reg.k, testing)))^2)
}
cv.error.step = sqrt(mean(pd.error)) # 3.264
regfit.null = lm(log(FE)~1, data=train)
regfit.full = lm(log(FE)~., data=train)
fit.step = step(regfit.null, scope = list(lower = regfit.null, upper = regfit.full),direction = "both")
test.step = predict(fit.step,test.result)
error.step = sqrt(mean((exp(test.step)-test.result$FE)^2))#3.779

# Best subset
# best.reg = regsubsets(log(FE)~., train, nvmax = 60)
# regfit.summary = summary(best.reg)

## Polynomial

# First exclude varibles deleted with stepwise from the dataset
train = train[,c(-6,-8)]
# Find out if higher order are necessary for EngDispl
fit.1 = lm(log(FE) ~.,data = train) 
fit.2 = lm(log(FE) ~ .-EngDispl+poly(EngDispl, 2),data = train) 
fit.3 = lm(log(FE) ~ .-EngDispl+poly(EngDispl, 3),data = train) 
fit.4 = lm(log(FE) ~ .-EngDispl+poly(EngDispl, 4),data = train) 
fit.5 = lm(log(FE) ~ .-EngDispl+poly(EngDispl, 5),data = train) 
anova(fit.1, fit.2, fit.3, fit.4, fit.5)
# Second order item is necessary

#Find out if higher order terms items for NumCyl are necessary
fit.1 = lm(log(FE) ~ .+I(EngDispl^2),data = train) 
fit.2 = lm(log(FE) ~ .-NumCyl+I(EngDispl^2)+poly(NumCyl, 2),data = train) 
fit.3 = lm(log(FE) ~ .-NumCyl+I(EngDispl^2)+poly(NumCyl, 3),data = train) 
fit.4 = lm(log(FE) ~ .-NumCyl+I(EngDispl^2)+poly(NumCyl, 4),data = train) 
fit.5 = lm(log(FE) ~ .-NumCyl+I(EngDispl^2)+poly(NumCyl, 5),data = train)
fit.6 = lm(log(FE) ~ .-NumCyl+I(EngDispl^2)+poly(NumCyl, 6),data = train)
anova(fit.1, fit.2, fit.3, fit.4, fit.5,fit.6)
# Second Order is necessary 

# Illustrate with plot 
# log(FE)~EngDispl
par(mfrow = c(1,1))
plot(EngDispl,log(FE),  cex = 0.5, col = "darkgrey",
     main = "Polynomial Regression v.s. Linear Regression (log(FE)~EngDispl)",
     cex.main = 0.9,ylab = "log(FE)") 

m.linear = lm(log(FE)~EngDispl,data = train)
abline(m.linear, col = "red")

m.poly = lm(log(FE)~poly(EngDispl,2),data=train)

EngDispl.grid=seq(from = min(EngDispl),to = max(EngDispl),length=1000)
preds.E = predict(m.poly,newdata = list(EngDispl= EngDispl.grid), se.fit = T)
se.bands = cbind(preds.E$fit + 2*preds.E$se.fit, preds.E$fit - 2*preds.E$se.fit )
lines(EngDispl.grid, preds.E$fit, col = "blue")
matlines(EngDispl.grid, se.bands, lwd = 1, col = "green",type = "l")
legend(4.5,4.2, c("Fitted value for Linear Reg.",
                  "Fitted value for Quatratic Reg.",
                  "SE.band"),
       col = c("red","blue","green"), cex = 0.8,lty = c(1,1,1,1))
# P

# log(FE)~Numcyl
plot(NumCyl,log(FE),  cex = 0.5, col = "darkgrey",
     main = "Polynomial Regression v.s Linear Regression (log(FE)~NumCyl) ",
     cex.main = 0.9,ylab = "log(FE)") 

m.linear = lm(log(FE)~NumCyl,data = train)
abline(m.linear, col = "red")

m.poly = lm(log(FE)~poly(NumCyl,2),data=train)
NumCyl.grid=seq(from = min(NumCyl),to = max(NumCyl),length=1000)
preds.N = predict(m.poly,newdata = list(NumCyl= NumCyl.grid), se.fit = T)
se.bands = cbind(preds.N$fit + 2*preds.N$se.fit, preds.N$fit - 2*preds.N$se.fit )
lines(NumCyl.grid, preds.N$fit, col = "blue")
matlines(NumCyl.grid, se.bands, lwd = 1, col = "green",type = "l")
#title("Polynomial Regression v.s First Order Regression (log(FE) vs NumCyl) ", outer = F) 
legend(7.5,4.2, c("Fitted value for Linear Reg.",
                 "Fitted value for Quatratic Reg.",
                 "SE.band"),
       col = c("red","blue","green"), cex = 0.8,lty = c(1,1,1,1))
# Polynomial regression
# 10-fold CV
k = 10
len = floor(length(FE)/k)
pd.error = vector()
for (i in 1:k){
  ind = (len*i-len+1) :(len*i)
  training = train[-ind,]
  testing = train[ind,]
  #model technique
  reg.k = lm(log(FE)~.+I(EngDispl^2)+I(NumCyl^2),data = training)
  pd.error[i] = mean((testing$FE - exp(predict(reg.k, testing)))^2)
}
cv.error.poly = sqrt(mean(pd.error)) # 3.176
fit.poly = lm(log(FE)~.+I(EngDispl^2)+I(NumCyl^2),data = train)
#RMSE = sqrt(mean((exp(poly.reg$fitted.values)-train$FE)^2))
test.poly = predict(fit.poly, test.result)
error.poly = sqrt(mean((test.result$FE-exp(test.poly))^2)) # 3.632

# Shrinkage--Ridge Regression
# 10-fold CV
train = read.csv('train.csv')
train = train[train$IntakeValvePerCyl>0,-1]
train[,c(3,5,7,8,9,12,13,14,15)]=lapply(train[,c(3,5,7,8,9,12,13,14,15)],as.factor)
k = 10
len = floor(length(FE)/k)
pd.error = vector()
for(i in 1:k){
  ind <- (len*i-len+1):(len*i)
  training <- train[-ind,]
  testing <- train[ind,]
  #model technique
  x.training<-model.matrix(log(FE)~.+I(EngDispl^2)+I(NumCyl^2),data = training)
  x.testing<-model.matrix(log(FE)~.+I(EngDispl^2)+I(NumCyl^2),data = testing)
  y<-log(training$FE)
  reg.k<-glmnet(x.training,y,alpha = 0)
  reg.k.cv<-cv.glmnet(x.training,y,alpha=0,type.measure = "mse")
  pd.error[i] <- mean((testing$FE-exp(predict.glmnet(reg.k,x.testing,s=reg.k.cv$lambda.min)))^2)
}
cv.error.ridge = sqrt(mean(pd.error)) # 3.340
x.train<-model.matrix(log(FE)~.+I(EngDispl^2)+I(NumCyl^2),data = train)
x.test.result<-model.matrix(log(FE)~.+I(EngDispl^2)+I(NumCyl^2),data = test.result)
y<-log(train$FE)
fit.ridge<-glmnet(x.train,y,alpha = 0)
fit.ridge.cv<-cv.glmnet(x.train,y,alpha=0,type.measure = "mse")
test.ridge = predict.glmnet(fit.ridge,x.test.result,s=fit.ridge.cv$lambda.min)
error.ridge = sqrt(mean((test.result$FE-exp(test.ridge))^2)) # 3.953

# Shrinkage--Lasso Regression
# 10-fold CV
k = 10
len = floor(length(FE)/k)
pd.error = vector()
for(i in 1:k){
  ind <- (len*i-len+1):(len*i)
  training <- train[-ind,]
  testing <- train[ind,]
  #model technique
  x.training<-model.matrix(log(FE)~.+I(EngDispl^2)+I(NumCyl^2),data = training)
  x.testing<-model.matrix(log(FE)~.+I(EngDispl^2)+I(NumCyl^2),data = testing)
  y<-log(training$FE)
  reg.k<-glmnet(x.training,y,alpha = 1)
  reg.k.cv<-cv.glmnet(x.training,y,alpha=1,type.measure = "mse")
  pd.error[i] <- mean((testing$FE-exp(predict.glmnet(reg.k,x.testing,s=reg.k.cv$lambda.min)))^2)
}
cv.error.lasso = sqrt(mean(pd.error)) # 3.183
x.train<-model.matrix(log(FE)~.+I(EngDispl^2)+I(NumCyl^2),data = train)
x.test.result<-model.matrix(log(FE)~.+I(EngDispl^2)+I(NumCyl^2),data = test.result)
y<-log(train$FE)
fit.lasso<-glmnet(x.train,y,alpha = 1)
fit.lasso.cv<-cv.glmnet(x.train,y,alpha=1,type.measure = "mse")
test.lasso = predict.glmnet(fit.lasso,x.test.result,s=fit.lasso.cv$lambda.min)
error.lasso = sqrt(mean((test.result$FE-exp(test.lasso))^2)) # 3.64


## Spline
# Double spline 
# 10-fold CV
train = train[,c(-6,-8)]
k = 10
len = floor(length(FE)/k)
pd.error = vector()
for (i in 1:k){
  ind = (len*i-len+1) :(len*i)
  training = train[-ind,]
  testing = train[ind,]
  #model technique
  knots_Eng = quantile(training$EngDispl, prob = seq(0.2,0.8,length = 4))
  knots_NCyl = quantile(training$NumCyl, prob = seq(0.2,0.8,length = 4))
  reg.k = lm(log(FE)~.-EngDispl-NumCyl+bs(EngDispl, knots = knots_Eng):
               bs(NumCyl, knots = knots_NCyl),data = training )
  pd.error[i]= mean((testing$FE-exp(predict(reg.k, testing)))^2)
}
cv.error.bspline = sqrt(mean(pd.error)) #1572962321
knots_Eng = quantile(train$EngDispl, prob = seq(0.2,0.8,length = 4))
knots_NCyl = quantile(train$NumCyl, prob = seq(0.2,0.8,length = 4))
fit.bspline = lm(log(FE)~.-EngDispl-NumCyl+bs(EngDispl, knots = knots_Eng):
                  bs(NumCyl, knots = knots_NCyl),data = train )
test.bspline = predict(fit.bspline, test.result)
error.bspline = sqrt(mean((test.result$FE-exp(test.bspline))^2)) # 3.49
# bs.error of pure 2 spline is 3.31
# bs.error of 1 spline (EngDispl) is 3.201
# test.error for 1 b-spline is 3.753


## Natural Spline
# 10-fold CV
k = 10
len = floor(length(FE)/k)
pd.error = vector()
for (i in 1:k){
  ind = (len*i-len+1) :(len*i)
  training = train[-ind,]
  testing = train[ind,]
  #model technique
  knots_Eng = quantile(training$EngDispl, prob = seq(0.2,0.8,length = 4))
  knots_NCyl = quantile(training$NumCyl, prob = seq(0.2,0.8,length = 4))
  reg.k = lm(log(FE)~.-EngDispl+ns(EngDispl, knots = knots_Eng):
               ns(NumCyl, knots =knots_NCyl),data = training )
  pd.error[i]= mean((testing$FE-exp(predict(reg.k, testing)))^2)
}
cv.error.nspline = sqrt(mean(pd.error)) #3.141
#ns.error for pure 2 nspline is 3.168

fit.nspline = lm(log(FE)~.-EngDispl-NumCyl+ns(EngDispl, knots = knots_Eng):
                   ns(NumCyl, knots = knots_NCyl),data = train) 
test.nspline = predict(fit.nspline, test.result)
error.nspline = sqrt(mean((test.result$FE-exp(test.nspline))^2)) #3.509
#test.error for pure 2 nspline is 3.628

#CV for nspline to find the best number of knots
cv.error.knots = vector()
for ( j in 3:10){
  k = 10
  len = floor(length(FE)/k)
  pd.error = vector()
  for (i in 1:k){
    ind = (len*i-len+1) :(len*i)
    training = train[-ind,]
    testing = train[ind,]
    #model technique
    knots_Eng = quantile(training$EngDispl, prob = seq(0,1,length = j)[c(-1,-j)])
    knots_NCyl = quantile(training$NumCyl, prob = seq(0,1,length = j)[c(-1,-j)])
    reg.k = lm(log(FE)~.-EngDispl+ns(EngDispl, knots = knots_Eng):
                 ns(NumCyl, knots =knots_NCyl),data = training )
    pd.error[i]= mean((testing$FE-exp(predict(reg.k, testing)))^2)
  }
  cv.error.knots[j]= sqrt(mean(pd.error)) 
  
}

plot(seq(3,7),cv.error.knots[3:7],type = "l", ylab = "CV.RMSE", xlab = "# knots",
     main = "CV.RMSE for N-Spline model with knots 3-7")
minpoint = which.min(cv.error.knots)
points(minpoint,cv.error.knots[minpoint], col = 'red', pch = 20)
legend(4,3.3,c("# knots with the least CV.RMSE"),col = "red", pch = 20)
legend(4, 3.15, round(cv.error.knots[minpoint],4),cex = 0.8)


plot(sort(cv.error.knots[3:7]),type = "")
cv.error = cbind(cv.error.SLR,cv.error.step,cv.error.poly,
                 cv.error.ridge,cv.error.lasso,
                 cv.error.bspline,cv.error.nspline)
test.error = cbind(error.SLR,error.step,error.poly,
                   error.ridge,error.lasso,
                   error.bspline,error.nspline)
detach(train)
