# PCA 
# dataset: wine
# method: SVD (singular value decomposition)
# Use this for class demo


library(caTools)
library(e1071)
library(caret)

path="F:/aegis/4 ml/dataset/unsupervised/pca/wine.csv"
wine=read.csv(path)

View(wine)

# y-variable (customer_segment) categorises customers based on various parameters
# for a given wine, predict to which customer segment this wine has to be recommended

# how to plot all these variables in a graph to show the segmentation
# use PCA to extract the 2 most (new) important features that can explain the maximum variation in the dataset
# prediction region and prediction boundary can then be viewed from this reduced dimensions

# these newly extracted features are called PRINCIPAL COMPONENTS

length(colnames(wine))

# feature scaling using the minmax()
minmax=function(x) return( (x-min(x)) / (max(x)-min(x)) )

#pos = grep('Customer_Segment', colnames(wine))
#wine_scale = wine[,c(1:pos-1)]
#wine_scale=as.data.frame(lapply(wine_scale,minmax))
#wine_scale[pos]=wine[pos]
#View(wine_scale)

# scale the dataset
winescale=as.data.frame(lapply(wine,minmax))
winescale$Customer_Segment=wine$Customer_Segment
View(winescale)

pos = grep('Customer_Segment', colnames(winescale))
pos

# apply the PCA
# --------------
pca=prcomp(winescale[-pos])
# pca
summ = summary(pca)
# look under "porportion of variance" to get the % of variance explained
print(summ)

res1=t(data.frame(summ$importance))
View(res1)
expl_var = res1[,'Proportion of Variance']

# explained variation
# expl_var = (summ$sdev^2)/sum( (summ$sdev)^2)

screeplot(pca,col="brown",main="Principal Components")


# Yes, rotation (orthogonal) is necessary because it maximizes the difference between variance captured by the component. This makes the components easier to interpret. Not to forget, that's the motive of doing PCA where, we aim to select fewer components (than features) which can explain the maximum variance in the data set. By doing rotation, the relative location of the components doesn't change, it only changes the actual coordinates of the points.

# If we don't rotate the components, the effect of PCA will diminish and we'll have to select more number of components to explain variance in the data set.


df = data.frame(PC= paste0("PC",1:13), var_explained=expl_var)
df$PC=factor(df$PC, levels=paste0("PC",1:13))
df
str(df)

ggplot(df,aes(x=PC,y=var_explained)) +
  geom_col(size=1,fill="white", colour="blue") +
  labs(title = "Scree Plot")


ggplot(df,aes(x=PC,y=var_explained)) +
  geom_bar(stat="identity", colour="black",fill="violet") +
  labs(title = "Scree Plot")



# build the PCA
wine_pca = as.data.frame(pca$x)
wine_pca = wine_pca[c(1,2)]
wine_pca$Customer_Segment = wine$Customer_Segment
View(wine_pca)


# ---------------------------------------------
# from here, build any classification model
# ----------------------------------------------

# shuffle the dataset
wine_pca = wine_pca[order(sample(seq(1,nrow(wine_pca)))),]
View(wine_pca)

# split the dataset into train and test
split=sample.split(wine_pca$Customer_Segment,SplitRatio = 0.8)
train=subset(wine_pca,split==TRUE)
test=subset(wine_pca,split==FALSE)
nrow(wine_pca); nrow(train); nrow(test)

View(train)
View(test)


# build an SVM
model=svm(Customer_Segment~., data=train, kernel='linear',
          type='C-classification')
prediction = predict(model,test[-3])
confusionMatrix(as.factor(test$Customer_Segment), as.factor(prediction))


# visualize the results
# ----------------------

# install.packages("ElemStatLearn")
library(ElemStatLearn)

set=train
X1=seq(min(set[,1])-1,max(set[,1])+1, by=0.1) 
X2=seq(min(set[,2])-1,max(set[,2])+1, by=0.1) 
grid_set = expand.grid(X1,X2)
colnames(grid_set)=c('PC1','PC2')
y_grid = predict(model,newdata = grid_set)
length(y_grid)

plot(set[,-3],
     main="SVM Classification",
     xlab='PC1', ylab='PC2',
     xlim=range(X1), ylim=range(X2))

contour(X1,X2,matrix(as.numeric(y_grid),length(X1),length(X2)),add=T)

points(grid_set,pch='.',col=ifelse(y_grid==2,'deepskyblue', ifelse(y_grid==1, 'springgreen3','tomato') ))
points(set,pch=21,bg=ifelse(set[,3]==2, 'blue3', ifelse(set[,3]==1, 'green4','red3')))

