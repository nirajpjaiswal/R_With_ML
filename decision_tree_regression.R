# decision tree classification

# load libraries
library(caret)
library(rpart)
library(rpart.plot)

# computer performance
# regression

file="F:/aegis/4 ml/dataset/supervised/regression/computer_performance/comp_perf.csv"
perf=read.csv(file,header=T)
str(perf)

perf$vendor=as.factor(perf$vendor)
perf$model_name=as.factor(perf$model_name)

# cols=c("vendor","model_name")
# perf[,cols]=NULL

perf$model_name=NULL

head(perf)

# do all the necessary checks

rows=nrow(perf)
s=sample(seq(1,rows), 0.7*rows)
train=perf[s,]
test=perf[-s,]
print(paste(dim(train),dim(test)))

# linear regression model
# lm1=lm(erp~.,data=train)
# cross validation
# library(DAAG)
# cvlm1=cv.lm(data=train,lm1,m=3)
# lmp1=predict(lm1,test)
# RMSE(lmp1,test$erp)^2


# decision tree regression model
dt1=rpart(erp~., data=train,method="anova")
rpart.plot(dt1,type=4, extra=101)

data.frame(score=dt1$variable.importance)

dt1$cptable
dtp1=predict(dt1,test)

# actual vs predicted values
results=data.frame("actual" = test$erp,
                   "pred_dt" = dtp1)


results[order(results$pred_dt,decreasing = T),]

