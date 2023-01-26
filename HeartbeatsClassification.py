# -*- coding: utf-8 -*-
"""

"""
# importing libs
import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier




# load train and test data
train_data = pd.read_csv("data/mitbih_train.csv", header=None )
test_data = pd.read_csv("data/mitbih_test.csv", header=None )

#print(train_data[187].value_counts())

# split train and test values
y_train = train_data[187]
y_test = test_data[187]
x_train  = train_data.loc[:,:186]
#x_train  = train_data.iloc[:,:-1].values
x_test  = test_data.loc[:,:186]

accu = []
prec = []
recc = []
f1   = []
models =[]

#random forest without feature selection ********************************************
print("***** Random forest without feature selection *****")
rfc= RandomForestClassifier(n_estimators=100,random_state=0, criterion='entropy',n_jobs=-1)

rfc.fit(x_train,y_train)
y3_pred =rfc.predict(x_test)

#confusion matrix
cm3=confusion_matrix(y_test,y3_pred)
print('RFC confusion matrix:')
print(cm3)

#f1 score
f1_nb3 = f1_score(y_test, y3_pred, average='macro')
print("F1 Score: ", f1_nb3)

accuracy = accuracy_score(y_test, y3_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


print('Precision Score : ' + str(precision_score(y_test,y3_pred,average='macro')))
print('Recall Score : ' + str(recall_score(y_test,y3_pred,average='macro')))

accu.append(accuracy_score(y_test, y3_pred))
prec.append(precision_score(y_test,y3_pred,average='macro'))
recc.append(recall_score(y_test,y3_pred,average='macro'))
f1.append(f1_score(y_test,y3_pred,average='macro'))
models.append("Random forest" )


# feature importance

feature_imp = np.argsort(rfc.feature_importances_)
#print(np.flip(feature_imp))

# plot
plt.figure(figsize=(20,8))
plt.title('Random forest')
plt.bar(range(len(rfc.feature_importances_)), rfc.feature_importances_)
plt.show()


#random forest with feature selection********************************************
print("\n")
print("***** Random forest with feature selection *****")
rfc_fs= RandomForestClassifier(n_estimators=100,random_state=0, criterion='entropy',n_jobs=-1)

rfc_fs = rfc_fs.fit(x_train, y_train)
print("feature importances: ")
print(rfc_fs.feature_importances_  )
model = SelectFromModel(rfc_fs, prefit=True,threshold=0.009)
x_train_new = model.transform(x_train)
x_test_new = model.transform(x_test)
print("x_train shape: ")  
print(x_train_new.shape)   

rfc_fs= RandomForestClassifier(n_estimators=100,random_state=0, criterion='entropy',n_jobs=-1)

rfc_fs.fit(x_train_new,y_train)
y3_pred =rfc_fs.predict(x_test_new)

#confusion matrix
cm3=confusion_matrix(y_test,y3_pred)
print('RFC confusion matrix: ')
print(cm3)

#f1 score
f1_nb3 = f1_score(y_test, y3_pred, average='macro')
print("F1 Score: ", f1_nb3)

accuracy = accuracy_score(y_test, y3_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


print('Precision Score : ' + str(precision_score(y_test,y3_pred,average='macro')))
print('Recall Score : ' + str(recall_score(y_test,y3_pred,average='macro')))


accu.append(accuracy_score(y_test, y3_pred))
prec.append(precision_score(y_test,y3_pred,average='macro'))
recc.append(recall_score(y_test,y3_pred,average='macro'))
f1.append(f1_score(y_test,y3_pred,average='macro'))
models.append("Random forest fs" )


# decision tree without feature selection****************************************************
print("\n")
print("***** Decision tree without feature selection *****")
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(x_train,y_train)
y_pred = dtc.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC confusion matrix: ')
print(cm)

#f1 score
f1_nb3 = f1_score(y_test, y_pred, average='macro')
print("F1 Score: ", f1_nb3)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


print('Precision Score : ' + str(precision_score(y_test,y_pred,average='macro')))
print('Recall Score : ' + str(recall_score(y_test,y_pred,average='macro')))

accu.append(accuracy_score(y_test, y_pred))
prec.append(precision_score(y_test,y_pred,average='macro'))
recc.append(recall_score(y_test,y_pred,average='macro'))
f1.append(f1_score(y_test,y_pred,average='macro'))
models.append("Decision tree" )

# feature importance

feature_imp = np.argsort(dtc.feature_importances_)
#print(np.flip(feature_imp))

# plot
plt.figure(figsize=(20,8))
plt.title('Decision tree')
plt.bar(range(len(dtc.feature_importances_)), dtc.feature_importances_)
plt.show()


# decision tree with feature selection****************************************************
print("\n")
print("***** Decision tree with feature selection *****")
dtc_fs = DecisionTreeClassifier(criterion = 'entropy')

dtc_fs.fit(x_train,y_train)
print("feature importances: ")
print(dtc_fs.feature_importances_  )
model = SelectFromModel(dtc_fs, prefit=True,threshold=0.001)
x_train_new = model.transform(x_train)
x_test_new = model.transform(x_test)
print("x_train shape: ")  
print(x_train_new.shape)   

dtc_fs = DecisionTreeClassifier(criterion = 'entropy')

dtc_fs.fit(x_train_new,y_train)
y_pred =dtc_fs.predict(x_test_new)


cm = confusion_matrix(y_test,y_pred)
print('DTC confusion matrix: ')
print(cm)

#f1 score
f1_nb3 = f1_score(y_test, y_pred, average='macro')
print("F1 Score: ", f1_nb3)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


print('Precision Score : ' + str(precision_score(y_test,y_pred,average='macro')))
print('Recall Score : ' + str(recall_score(y_test,y_pred,average='macro')))

accu.append(accuracy_score(y_test, y_pred))
prec.append(precision_score(y_test,y_pred,average='macro'))
recc.append(recall_score(y_test,y_pred,average='macro'))
f1.append(f1_score(y_test,y_pred,average='macro'))
models.append("Decision tree fs" )


# xgboost without feature selection****************************************************
print("\n")
print("***** Xgboost without feature selection *****")

classifier=XGBClassifier()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

print(cm)
#f1 score
f1_nb3 = f1_score(y_test, y_pred, average='macro')
print("F1 Score: ", f1_nb3)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


print('Precision Score : ' + str(precision_score(y_test,y_pred,average='macro')))
print('Recall Score : ' + str(recall_score(y_test,y_pred,average='macro')))

accu.append(accuracy_score(y_test, y_pred))
prec.append(precision_score(y_test,y_pred,average='macro'))
recc.append(recall_score(y_test,y_pred,average='macro'))
f1.append(f1_score(y_test,y_pred,average='macro'))
models.append("Xgboost" )

# feature importance

feature_imp = np.argsort(dtc.feature_importances_)
#print(np.flip(feature_imp))

# plot
plt.figure(figsize=(20,8))
plt.title('Xgboost')
plt.bar(range(len(classifier.feature_importances_)), classifier.feature_importances_)
plt.show()


# xgboost with feature selection****************************************************
print("\n")
print("***** Xgboost with feature selection *****")
classifier_fs=XGBClassifier()
classifier_fs.fit(x_train,y_train)

print("feature importances: ")
print(classifier_fs.feature_importances_  )
model = SelectFromModel(classifier_fs, prefit=True,threshold=0.005)
x_train_new = model.transform(x_train)
x_test_new = model.transform(x_test)
print("x_train shape: ")  
print(x_train_new.shape)   

classifier=XGBClassifier()

classifier_fs.fit(x_train_new,y_train)
y_pred =classifier_fs.predict(x_test_new)


cm = confusion_matrix(y_test,y_pred)

print(cm)
#f1 score
f1_nb3 = f1_score(y_test, y_pred, average='macro')
print("F1 Score: ", f1_nb3)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


print('Precision Score : ' + str(precision_score(y_test,y_pred,average='macro')))
print('Recall Score : ' + str(recall_score(y_test,y_pred,average='macro')))

accu.append(accuracy_score(y_test, y_pred))
prec.append(precision_score(y_test,y_pred,average='macro'))
recc.append(recall_score(y_test,y_pred,average='macro'))
f1.append(f1_score(y_test,y_pred,average='macro'))
models.append("Xgboost fs" )



x = ['accuracy','precision','recall','f1']
plt.figure(figsize=(10,10))
plt.title('score')
# plot lines
for i in range(len(accu)):
    plt.plot(x,[accu[i],prec[i],recc[i],f1[i]], label = models[i])
plt.legend()
plt.savefig('curves.png')
plt.show()


