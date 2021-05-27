# simple linear regression
# dataset: CCSS (electric enegrgy production prediction)

path="F:/aegis/4 ml/dataset/supervised/regression/ccpp/ccpps.csv"

ccpps=read.csv(path,header=T)

head(ccpps)
tail(ccpps)

dim(ccpps)

# check if data is all numeric
str(ccpps)

# functions to check Nulls/0
checkNull=function(x) return(any(is.na(x)))
checkZero=function(x) return(any(x==0))

# EDA
colnames(ccpps)[apply(ccpps,2,checkNull)]
colnames(ccpps)[apply(ccpps,2,checkZero)]

# check the distribution of X
hist(ccpps$temp,breaks=10,main="Histogram of Temperature",col="yellow")

# check for outliers
boxplot(ccpps$temp,horizontal=T,main="Boxplot for Temperature", col="red")

# split the data into train and test sets
totalrows = nrow(ccpps)
print(totalrows)

# generate 70% data
ss = sample(seq(1,totalrows),floor(0.7*totalrows))

# get the train and test data
train = ccpps[ss,]
test = ccpps[-ss,]

print(paste("train=",dim(train),"test=",dim(test)))

head(train)
head(test)

# build the linear regression model
m1 = lm(elec_energy~temp, data=train)

# m1 = lm(elec_energy ~ .)
summary(m1)

  
# predict the Y on the test data
p1 = predict(m1,test)

p1[1:10]
  
# create a dataframe to store the actual Y and predicted Y
result=data.frame('actual'= test$elec_energy,
                  'predicted' = round(p1,2))

head(result,25)
  
# calculate the error/sse
result$err=result$actual-result$predicted
head(result)  
result$se = result$err^2

head(result)

# SSE
sse = sum(result$se)

# MSE (mean squared errors)
mse = sse/nrow(test)
mse

# RMSE (root mean sq err)
rmse = sqrt(mse)
print(rmse)
