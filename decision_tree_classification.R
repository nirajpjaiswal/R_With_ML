# decision tree classification

# load libraries
library(caret)
library(rpart)
library(rpart.plot)

# read data
path="F:/work/2 myPresentation/5 training/1 imarticus/7 projects/inclass/classification/bankchurn/bankchurn.csv"

churn=read.csv(path,header=T)

# remove unwanted features
churn$custid=NULL
churn$surname=NULL

# structure of dataset
str(churn)


# country - merge the levels
churn$country[churn$country%in%c('Espanio','spain')] = "Spain"
churn$country[churn$country%in%c('Ger','germany')] = "Germany"
churn$country[churn$country%in%c('Fra','france')] = "France"

table(churn$country)
str(churn$country)
churn$country=as.factor(churn$country)

# gender- merge the levels
table(churn$gender)
churn$gender[churn$gender%in%c('m','Male')] = "M"
churn$gender[churn$gender%in%c('f','female','Female')] = "F"

churn$gender=as.factor(churn$gender)
str(churn$gender)

# convert Y-variable into a factor
churn$churn=as.factor(churn$churn)

# split the data into train and test
totalrows=nrow(churn)
ss = sample(seq(1,totalrows),floor(0.7*totalrows))
train = churn[ss,]
test = churn[-ss,]

print(dim(train))
print(dim(test))

# build the decision tree model
# without any hyperparameter tuning


cols = colnames(train)
Y = cols[9]
cols=cols[1:8]
cols

# formula = y~x 
m1=rpart(churn~., data=train, method="class")

# visualise the decision tree
rpart.plot(m1,type=4,extra=101, tweak = 1.5)

?rpart.plot

print(m1)

# predict 
p1 = predict(m1,test,type="class")
p1

# confusion matrix
confusionMatrix(test$churn,as.factor(p1),positive = "1")


table(test$churn)

# plot the complexity parameter
plotcp(m1)

# complexity parameter
m1$cptable
