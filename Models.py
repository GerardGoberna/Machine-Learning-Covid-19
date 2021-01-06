#Separem el dataset

X = dataset.iloc [:, 0:18]
y = dataset.iloc [:, 18]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.33)

#Dades amb només les variables més correlacionades
X = dataset.iloc [:, [0,1,2,3,13,14,15]]
y = dataset.iloc [:, 18]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.33)

#Dades sense les variables amb alts p-valors
X = dataset.iloc [:, [0,1,2,3,4,5,7,8,9,10,11,13,14,15,16]]
y = dataset.iloc [:, 18]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.33)
#Escalem les característiques

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test =sc_X.fit_transform(X_test)

#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

##Entrenem el model
#Busquem el numero de veïns
import math
math.sqrt(len(y_test))
#Definim el model
classifier = KNeighborsClassifier(n_neighbors=42, p=2, metric='euclidean')
#Ajustem el model
classifier.fit(X_train, y_train)
#Predim els resultats del test
prediccio_KNN = classifier.predict(X_test)
print(prediccio_KNN)
#Probabilitats
prob_KNN = classifier.predict_proba(X_test)
print(prob_KNN)
#Avaluem el model
cm_KNN = confusion_matrix(y_test, prediccio_KNN)
print(cm_KNN)
#Mirem la precisió
print(f1_score(y_test,prediccio_KNN))
f1_KNN = (f1_score(y_test,prediccio_KNN))*100
print(accuracy_score(y_test,prediccio_KNN))
acc_KNN = (accuracy_score(y_test,prediccio_KNN))*100

#Naive Bayes

from sklearn.naive_bayes import GaussianNB
clf_NB = GaussianNB()
clf_NB = clf_NB.fit(X_train, y_train)
prediccio_NB = clf_NB.predict(X_test)

#Avaluem el model
cm_NB = confusion_matrix(y_test, prediccio_NB)
print(cm_NB)

print(f1_score(y_test,prediccio_NB))
print(accuracy_score(y_test,prediccio_NB))

f1_NB = (f1_score(y_test,prediccio_NB))*100
acc_NB = (accuracy_score(y_test,prediccio_NB))*100

#Decision Trees

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
prediccio_DT = clf.predict(X_test)
cmDT = confusion_matrix(y_test, prediccio_DT)
print(cmDT)
print(f1_score(y_test,prediccio_DT))
print(accuracy_score(y_test,prediccio_DT))

f1_DT = (f1_score(y_test,prediccio_DT))*100
acc_DT = (accuracy_score(y_test,prediccio_DT))*100

#SVM

#SVC
from sklearn import svm
clf_SVC = svm.SVC()
clf_SVC = clf_SVC.fit(X_train, y_train)
prediccio_SVC = clf_SVC.predict(X_test)
cm_SVC = confusion_matrix(y_test,prediccio_SVC)
print(cm_SVC)
print(f1_score(y_test,prediccio_SVC)
print(accuracy_score(y_test,prediccio_SVC))
f1_SVC = (f1_score(y_test,prediccio_SVC))*100
acc_SVC = (accuracy_score(y_test,prediccio_SVC))*100


#Linear SVC
clf_linSVC = svm.LinearSVC()
clf_linSVC.fit(X_train, y_train)
prediccio_linSVC = clf_linSVC.predict(X_test)
cm_linSVC = confusion_matrix(y_test,prediccio_linSVC)
print(cm_linSVC)
print(f1_score(y_test,prediccio_linSVC))
print(accuracy_score(y_test,prediccio_linSVC))

f1_linSVC = (f1_score(y_test,prediccio_linSVC))*100
acc_linSVC = (accuracy_score(y_test,prediccio_linSVC))*100


#Probabilitats
prob_linSVC = clf_linSVC._predict_proba_lr(X_test)
print(prob_linSVC)

#Random forest

from sklearn.ensemble import RandomForestClassifier

clf_RF = RandomForestClassifier(max_depth=2, random_state=0)
clf_RF.fit(X_train, y_train)
prediccio_RF = clf_RF.predict(X_test)
print(prediccio_RF)
cm_RF = confusion_matrix(y_test,prediccio_RF)
print(cm_RF)
print(f1_score(y_test,prediccio_RF))
print(accuracy_score(y_test,prediccio_RF))
f1_RF = (f1_score(y_test,prediccio_RF))*100
acc_RF = (accuracy_score(y_test,prediccio_RF))*100

#Probabilitats

prob_RF = clf_RF.predict_proba(X_test)
print(prob_RF)

#Regressió logística

from sklearn.linear_model import LogisticRegression
clf_LR = LogisticRegression()
#Fit the model
clf_LR.fit(X_train, y_train)
prediccio_LR = clf_LR.predict(X_test)

cm_LR = confusion_matrix(y_test,prediccio_LR)
print(cm_LR)
print(f1_score(y_test,prediccio_LR))
print(accuracy_score(y_test,prediccio_LR))

f1_LR = (f1_score(y_test,prediccio_LR))*100
acc_LR = (accuracy_score(y_test,prediccio_LR))*100



Taula_Models = pd.DataFrame({
    'Model': ['KNN', 'Naive Bayes', 'Decision Tree','Support Vector Machines',
              ' Linear Support Vector Machines',
              'Random Forest', 'Logistic Regression'],
    'Exactitut': [acc_KNN, acc_NB, acc_DT,
              acc_SVC, acc_linSVC, acc_RF,
               acc_LR],
    'F1 Score': [f1_KNN, f1_NB, f1_DT,
              f1_SVC, f1_linSVC, f1_RF,
               f1_LR]})

print(Taula_Models)

