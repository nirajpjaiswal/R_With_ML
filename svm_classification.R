# SVM classification
# dataset: CTG

# libraries
library(e1071)
library(caret)
library(corrplot)

# read the data
path="F:/aegis/4 ml/dataset/supervised/classification/cardiotacography/ctg_1.csv"
ctg=read.csv(path,header=T)

View(ctg)

# perform the EDA

nrow(ctg)

# distribution of Y
table(ctg$NSP)

str(ctg)

# minmax
minmax=function(x) return( (x-min(x))/(max(x)-min(x)) )

# standardize the dataset
ctg_scale = as.data.frame(lapply(ctg,minmax))

# ensure the scaling has been done properly
# data shd lie between 0 and 1
summary(ctg_scale)

# replace the scaled NSP with the original Y
ctg_scale$NSP = ctg$NSP

View(ctg_scale)

# change the Y to factor in both datasets
ctg_scale$NSP=as.factor(ctg_scale$NSP)
ctg$NSP=as.factor(ctg$NSP)

table(ctg$DS)
table(ctg$DP)

# split the data into train and test
total = nrow(ctg_scale)
ss = sample(seq(1,total), floor(0.7*total))
train=ctg_scale[ss,]
test=ctg_scale[-ss,]

dim(train)
dim(test)

# build SVM models using different kernels
# 1) linear
# 2) radial
# 3) sigmoid
# 4) polynomial
# ......................................

ker = 'linear'
c_list=c(0.001,0.01,0.1,1,10,50,100)

# do the CV to get the best C

cv1 = tune(svm,NSP~., 
           data=train,
           kernel=ker,
           ranges=list(cos=c_list)
           )
print(cv1)
optC = unlist(cv1$best.parameters)
optC

# build model with the opt C
m1=svm(NSP~., data=train, kernel=ker,
       cost=optC)

# predict
p1=predict(m1,test)

# confusion matrix and clasification report
cm1=confusionMatrix(test$NSP,p1)
table(test$NSP)

names(train)

# visualise
plot(m1,test,ASTV~Variance)
plot(m1,test,DP~UC)

# model 2: kernel = radial
ker = "radial"
c_list=c(0.001,0.01,0.1,1,10,50,100)
g_list=c(0.005,0.05,0.02,0.01,0.1,1)

# cross-validation to get the optimal C and Gamma combination

cv1=tune(svm,NSP~., data=train, 
      kernel = ker,
      ranges=list(cos=c_list, gamma=g_list))

print(cv1$best.parameters)

optC = unlist(cv1$best.parameters[1])
optG = unlist(cv1$best.parameters[2])

# RBF model
m2=svm(NSP~., data=train,kernel=ker,
       cost=optC,
       gamma=optG)

# predict
p2=predict(m2,test)

# confusion matrix
cm2=confusionMatrix(test$NSP,p2)


# model 3: kernel = sigmoid
ker = "sigmoid"
c_list=c(0.001,0.01,0.1,1,10,50,100)
g_list=c(0.005,0.05,0.02,0.01,0.1,1)

# cross-validation to get the optimal C and Gamma combination

cv1=tune(svm,NSP~., data=train, 
         kernel = ker,
         ranges=list(cos=c_list, gamma=g_list))

print(cv1$best.parameters)

optC = unlist(cv1$best.parameters[1])
optG = unlist(cv1$best.parameters[2])

# SIGMOID model
m3=svm(NSP~., data=train,kernel=ker,
       cost=optC,
       gamma=optG)

# predict
p3=predict(m3,test)

# confusion matrix
cm3=confusionMatrix(test$NSP,p3)
cm3


# model 4: kernel = polynomial
ker = "polynomial"
c_list=c(0.001,0.01,0.1,1,10,50,100)
g_list=c(0.005,0.05,0.02,0.01,0.1,1)

# cross-validation to get the optimal C and Gamma combination

cv1=tune(svm,NSP~., data=train, 
         kernel = ker,
         ranges=list(cos=c_list, gamma=g_list))

print(cv1$best.parameters)

optC = unlist(cv1$best.parameters[1])
optG = unlist(cv1$best.parameters[2])

# POLYNOMIAL model
m4=svm(NSP~., data=train,kernel=ker,
       cost=optC,
       gamma=optG)

# predict
p4=predict(m4,test)

# confusion matrix
cm4=confusionMatrix(test$NSP,p4)
cm4


# create a dataframe to store the results of all 4 models
# analyse the DF to pick the best peforming model

# M1
df1=as.data.frame(cm1$byClass)
df1$model = 'linear'
df1$accuracy = cm1$overall[1]   
   
# M2
df2=as.data.frame(cm2$byClass)
df2$model = 'radial'
df2$accuracy = cm2$overall[1]   

# M3
df3=as.data.frame(cm3$byClass)
df3$model = 'sigmoid'
df3$accuracy = cm3$overall[1]   

# M4
df4=as.data.frame(cm4$byClass)
df4$model = 'polynomial'
df4$accuracy = cm4$overall[1]   

# merge all the dataframes
svmdf=rbind(df1,df2,df3,df4)


names(svmdf)

svmdf$`Detection Prevalence`=NULL
svmdf$`Detection Rate`=NULL
svmdf$`Balanced Accuracy`=NULL
svmdf$`Neg Pred Value`=NULL
svmdf$Prevalence=NULL

print(svmdf)

# include the class name as a column
svmdf$class = rep(c(1,2,3),4)
rownames(svmdf) = NULL
svmdf$class=as.factor(svmdf$class)

# rename the columns
names(svmdf)[names(svmdf)=="Pos Pred Value"] = "positivepred"

View(svmdf)

cols=c()
svmdf=svmdf[cols]

View(svmdf)

models=c('linear','radial','sigmoid','polynomial')

df1 = data.frame(model=models,
                  accuracy=unique(svmdf[svmdf$model%in%models,"accuracy"]) )
df1

barplot(df1$accuracy,xlab=models,ylab='Accuracy')


library(ggplot2)

# 1) plot the accuracy of all models
# vertical plot
ggplot(df1,aes(x=model,y=accuracy)) +
   geom_col(fill='#0099f9') +
   labs(
      title = 'Accuracy comparison',
      subtitle = 'SVM models')
   
# horizontal plot
ggplot(df1,aes(x=model,y=accuracy)) +
   geom_col(fill='#0099f9') +
   coord_flip() +
   labs(
      title = 'Accuracy comparison',
      subtitle = 'SVM models')


# 2) Positive predictions based on Classes of all models

# method 1
ggplot(svmdf,aes(x=class,y=positivepred,fill=model)) +
   geom_col(position=position_dodge()) +
   coord_flip() +
   scale_fill_brewer(palette="Set1")


# method 2
ggplot(svmdf,aes(x=model,y=positivepred,fill=class)) +
   geom_col(position=position_dodge()) +
   coord_flip() +
   scale_fill_brewer(palette="Set2")