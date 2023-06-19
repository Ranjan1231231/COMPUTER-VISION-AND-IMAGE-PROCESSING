#SUPPORT VECTOR MACHINE VS VANILLA LINEAR CLASSIFIER

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,svm,metrics,model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score

digits=datasets.load_digits()
target=digits.target
flatten_digits=digits.images.reshape((len(digits.images),-1))
#VISUALIZING SOME HANDWRITTEN IMAGES IN THE DATASETS
_,axes=plt.subplots(nrows = 1,ncols = 5,figsize=(10,4))
for ax, image,label in zip(axes,digits.images,target):
    ax.set_axis_off()
    ax.imshow(image,cmap=plt.cm.gray_r,interpolation = "nearest")
    ax.set_title("%i"%label)
#DIVIDE IMAGES INTO TRAINING AND TEST SET
X_train,X_test,y_train,y_test=train_test_split(flatten_digits,target,test_size = 0.2)
#HAND-WRITTEN CLASSIFICATION WITH LOGISTIC REGRESSION
scaler=StandardScaler()
X_train_logistic=scaler.fit_transform(X_train)
X_test_logistic=scaler.transform(X_test)
logit=LogisticRegression(C=0.01,penalty = "l1",solver = "saga",tol=0.1,multi_class = "multinomial")
logit.fit(X_train_logistic,y_train)
y_pred_logistic=logit.predict(X_test_logistic)
# print("Acurracy:"+str(logit.score(X_test_logistic,y_test)))
label_names=[0,1,2,3,4,5,6,7,8,9]
cmx=confusion_matrix(y_test,y_pred_logistic,labels=label_names)
df_cm=pd.DataFrame(cmx)
#plt.figure(figsize=(10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm,annot = True,annot_kws = {"size":16})#font size
# title="Confusion Matrix for SVM results"
# plt.title(title)
# plt.show()
#HAND WRITTEN CLASSIFICATION WITH SVM
svm_classifier=svm.SVC(gamma = "scale")
svm_classifier.fit(X_train,y_train)
#PREDICT OUR TEST SET
y_pred_svm=svm_classifier.predict(X_test)
#GET ACCURACY FOR THE SVM MODEL WE CAN SEE WE HAVE A NEARLY PERFECT MODEL
# print("Accuracy:"+str(accuracy_score(y_test,y_pred_svm)))
label_names=[0,1,2,3,4,5,6,7,8,9]
cmx=confusion_matrix(y_test,y_pred_svm,labels=label_names)
df_cm=pd.DataFrame(cmx)
#plt.figure(figsize=(10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm,annot=True,annot_kws = {"size":16})#font sizze
title='Confusion Matrix for SVM results'
# plt.title(title)
# plt.show()

#Comparing both SVM AND LOGISTIC REGRESSION WITH K-FOLDCROSS VALIDATION
algorithm=[]
algorithm.append(("SVM",svm_classifier))
algorithm.append(("Logistic_L1",logit))
algorithm.append(('LOGISIC_L1',logit))
algorithm.append(("Logistic_L2",LogisticRegression(C=0.01,penalty = "l2",solver = "saga",tol=0.1,multi_class = "multinomial")))
results=[]
names=[]
y=digits.target
for name,algo in algorithm:
    k_fold=model_selection.KFold(n_splits = 10)
    if name=="SVM":
        X=flatten_digits
        cv_results=model_selection.cross_val_score(algo,X,y,cv=k_fold,scoring ='accuracy' )
    else:
        scaler=StandardScaler()
        X=scaler.fit_transform(flatten_digits)
        cv_results=model_selection.cross_val_score(algo,X,y,cv=k_fold,scoring = "accuracy")

    results.append(cv_results)
    names.append(name)
fig=plt.figure()
fig.suptitle("Compare Logistic and SVM Results")
ax=fig.add_subplot()
plt.boxplot(results)
plt.ylabel("Accuracy")
ax.set_xticklabels(names)
plt.show()