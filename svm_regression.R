# SVM Regression
# dataset: energy cool

library(e1071)
library(ggplot)
library(caret)

path="F:/aegis/4 ml/dataset/supervised/regression/energy/energy_cooling_load.csv"

energy=read.csv(path,header=T)

# EDA

# split the data into train and test
total = nrow(energy)
ss=sample(seq(1,total),floor(0.7*total))
train=energy[ss,]
test=energy[-ss,]

dim(train)
dim(test)

names(train)

# 1) regression type = 'EPS-regression'
# 4 kernels -> linear, radial, sigmoid, polynomial
r_type = 'eps-regression'

# kernel = 'linear'
m1=svm(cold_load~., data=train,
       type=r_type, 
       kernel='linear',
       cross=10)
summary(m1)
p1=predict(m1,test)

cbind(test$cold_load,p1)
rmse1=RMSE(test$cold_load,p1)

# visualise the result
ggplot(test,aes(cold_load,p1)) +
  geom_point(colour='green') +
  geom_smooth(colour='red',method=lm)+
  ggtitle('EPS/Linear/C=1')


# hyperparameter tuning - (C)cost=2
m2=svm(cold_load~., data=train,
       type=r_type, 
       kernel='radial',
       cross=10)

summary(m2)
p2=predict(m2,test)
rmse2=RMSE(test$cold_load,p2)

cat("RMSE1=",rmse1,"\nRMSE2=",rmse2)

# kernel = 'radial'
m3=svm(cold_load~., data=train,
       type=r_type, 
       kernel='radial',
       cross=10)
summary(m3)
p3=predict(m3,test)

# kernel = 'sigmoid'
m4=svm(cold_load~., data=train,
       type=r_type, 
       kernel='sigmoid',
       cross=10)
summary(m4)
p4=predict(m4,test)
rmse4=RMSE(test$cold_load,p4)
cat("RMSE1=",rmse1,"\nRMSE2=",rmse2,"\nRMSE3=",rmse3,"\nRMSE4=",rmse4)

# kernel = 'polynomial'
m5=svm(cold_load~., data=train,
       type=r_type, 
       kernel='polynomial',
       cross=10)
summary(m5)
p5=predict(m5,test)
rmse5=RMSE(test$cold_load,p5)
cat("RMSE1=",rmse1,"\nRMSE2=",rmse2,"\nRMSE3=",rmse3,"\nRMSE4=",rmse4,"\nRMSE5=",rmse5)


# 2) regression type = 'nu-regression'
r_type = 'nu-regression'

# kernel = 'linear'
m6=svm(cold_load~., data=train,
       type=r_type, 
       kernel='linear',
       cross=10)
summary(m6)
p6=predict(m6,test)
rmse6=RMSE(test$cold_load,p6)

cat("RMSE (linear kernel,C=1)\nEPS=",rmse1,"\nNU=",rmse6)

####################################################

buildSVM=function(rtype,ker,C=1,train,test,Y)
{
  f = as.formula(paste(Y,"~."))
  
  model=svm(f, data=train,cost=C,
            type=rtype,
            kernel=ker)
  
  pred = predict(model,test)
  rmse = RMSE(unlist(test[Y]),pred)
  return(rmse)
}

#################################################

# single function call to build and predict the different types of SVM regression models 

rtype=c('nu-regression','eps-regression')
ktype=c('linear','radial','sigmoid','polynomial')

df=c()

for(rt in rtype)
{
  for(kt in ktype)
  {
    err=buildSVM(rtype=rt,ker=kt,--
                train=train,test=test,Y='cold_load')
  
    # cat("Regression type=",rt,"Kernel=",kt,"RMSE=",err)
    # cat("\n")
    df=rbind(df,data.frame('regression'= rt,
                           'kernel'= kt,
                           'rmse'= err))
  }
  # cat("\n")
}

# store results in the dataframe for analysis
df = as.data.frame(df)

# 1) plot the RMSE of all models
ggplot(df,aes(x=kernel,y=rmse,fill=regression)) +
  geom_col(position=position_dodge()) +
  coord_flip() +
  labs( title='RMSE comparison', subtitle='SVM models') +
  scale_fill_brewer(palette="Set1")