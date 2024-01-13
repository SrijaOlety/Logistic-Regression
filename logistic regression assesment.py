# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 19:15:51 2023

@author: dell
"""
### importing the data###

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("C:\\Users\\dell\\Downloads\\bank-full (1).csv")
df
df.info()
pd.set_option('display.max_columns', None)
df
df.isnull().sum()


# EDA
#EDA----->EXPLORATORY DATA ANALYSIS

import seaborn as sns
import matplotlib.pyplot as plt
data = ['age','balance','day','duration','campaign','pdays','previous']
for column in data:
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    sns.boxplot(x=df[column])
    plt.title(f" Horizontal Box Plot of {column}")
    plt.show()
    
#so basically we have seen the ouliers at once without doing everytime for each variable using seaborn#

"""removing the ouliers"""

# List of column names with continuous variables

continuous_columns = ["age", "balance","duration","campaign","pdays","previous" ]  
# Create a new DataFrame without outliers for each continuous column
data_without_outliers = df.copy()
for df.cloumns in continuous_columns:
    Q1 = data_without_outliers[column].quantile(0.25)
    Q3 = data_without_outliers[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker_Length = Q1 - 1.5 * IQR
    upper_whisker_Length = Q3 + 1.5 * IQR
    data_without_outliers = data_without_outliers[(data_without_outliers[column] >= lower_whisker_Length) & (data_without_outliers[column]<= upper_whisker_Length)]
    
# Print the cleaned data without outliers

print(data_without_outliers)
df = data_without_outliers
print(df)

"""data division and standardizing"""

df_cont = df.iloc[:,[0,5,9,11,12,13,14]]
df_cont
df_cont.info()
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X1 = SS.fit_transform(df_cont)
X1= pd.DataFrame(X1)
X1.columns=list(df_cont)
X1
df_cat = df.iloc[:,[1,2,3,4,6,7,8,10,15]]
df_cat
"""df_cat = df.drop(df.columns[[0,5,9,11,12,13,14]],axis=1)"""
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
for i in range(0,9):
    df_cat.iloc[:,i] = LE.fit_transform(df_cat.iloc[:,i])
df_cat.head()
X = pd.concat([df_cont,df_cat],axis = 1)
X
X.info()

Y = df.iloc[:,16:17]
Y

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
Y.iloc[:,0] = LE.fit_transform(Y.iloc[:,0])
Y

#  data partition and data validation#
from sklearn.model_selection import train_test_split
#by default it will take 75% of data as training data if we donot mention in the code#
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size = 0.75,random_state = 15)
X_train.shape
X_test.shape


"""fitting the model   here we fitted all the X and Y variables data i.e., complete data"""
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X,Y)
Y_pred = logreg.predict(X)
Y_pred


logreg.predict_proba(X)[:,1]
df["Y_probabilities"] = logreg.predict_proba(X)[:,1]

#Function to change the cut off
def f1(X):
    if X<0.4:
        return 0
    elif X>=0.4:
        return 1

df["Y_prob"] = df["Y_probabilities"].apply(f1)
df

Y_p = df["Y_prob"]
Y_p
#confusion matrix and accuracy score

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y,Y_p)
cm
ac = accuracy_score(Y,Y_p)
ac  


#################################################################################################################


#Validation set approach
training_accuracy = []
test_accuracy = []

for i in range(1,101):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.75,random_state=i)
    logreg.fit(X_train,Y_train)
    Y_pred_train =logreg.predict(X_train)
    Y_pred_test =logreg.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))
print("Average Training accuracy ",np.mean(training_accuracy).round(3))
print("Average Test accuracy ",np.mean(test_accuracy).round(3))

#################################################################################################################


#Metrices
from sklearn.metrics import recall_score,precision_score,f1_score
print("Sensitivity score:",recall_score(Y,Y_pred).round(3))
print("Precision score:",precision_score(Y,Y_pred).round(3))
print("F1 score:",f1_score(Y,Y_pred).round(3))

TN =cm[0,0]
FP =cm[1,0]

TNR = TN/(TN+FP)

print("Specificity:",TNR.round(3))

############################################################################################################


#K-fold 
from sklearn.model_selection import KFold
kf = KFold(n_splits=5) 

for train_index,test_index in kf.split(range(11,143)):
    print(train_index)
    print(test_index)
    
training_accuracy = []
test_accuracy = []

for train_index,test_index in kf.split(X):
    X_train,X_test = X.iloc[train_index],X.iloc[test_index]
    Y_train,Y_test= Y.iloc[train_index],Y.iloc[test_index]
    logreg = LogisticRegression()
    logreg.fit(X_train,Y_train)
    Y_pred_train =logreg.predict(X_train)
    Y_pred_test =logreg.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))   

print("K-Fold Training accuracy ",np.mean(training_accuracy).round(3))
print("K-Fold Test accuracy ",np.mean(test_accuracy).round(3))

###################################################################################################################
# ROC Curve plotting and finding AUC value
from sklearn.metrics import roc_auc_score,roc_curve
fpr,tpr,dummy = roc_curve(Y,Y_p)

import matplotlib.pyplot as plt
plt.scatter(x=fpr,y=tpr)
plt.plot(fpr,tpr,color='red')
plt.plot([0,1],[0,1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

auc = roc_auc_score(Y,Y_p)
print("Area under curve:",(auc*100).round(3))


                  
