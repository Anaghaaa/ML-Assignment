from sklearn.svm import SVC
from sklearn.svm import LinearSVC


def learn_and_report(kernel):
  classifier.fit(X_train,y_train)
  Y_pred = classifier.predict(X_test)
  print('Classification with {} kernel'.format(kernel))
  print('True labels **** Predicted labels')
  print('(1=Spam; 0=Not spam)')

  for i in range(len(Y_pred)):
    print(str(y_test[i])+' ************* '+str(Y_pred[i]))

  accuracy_count=0
  for i in range(len(Y_pred)):
    if Y_pred[i]==y_test[i]:
      accuracy_count+=1

  accuracy=(accuracy_count/len(y_test))*100
  print('Accuracy of SVM using {} kernel  function for given dataset is {}% \n ------------------------------------\n'.format(kernel, accuracy))

datafile= open('shuffledspam.csv','r')  #rows of given dataset spambase.data have been manually shuffled to serve as suitable input to the learning model
rows= datafile.readlines()
datafile.close()

X=[]
y=[]
rows= [row.strip() for row in rows]
for row in rows:
  attributes=[float(val) if '.' in val else int(val) for val in row.split(',')]
  #splitting into predictors and labels
  X.append(attributes[0:57])
  y.append(attributes[57])


#splitting into training and testing datasets in the ratio 80:20
train_length=int(0.8*len(X))

X_train=X[0:train_length]
X_test=X[train_length:]
y_train=y[0:train_length]
y_test=y[train_length:]

print('training data----')
print(X_train)
print(y_train)

print('testing data----')
print(X_test)
print(y_test)

#SVM classification using RBF kernel
#best value found of C=500
classifier = SVC(C=500.0,kernel='rbf', gamma=0.001,random_state = 1)

learn_and_report('RBF')

#SVM classification using linear kernel 
# LinearSVC classifier object selects the best value of C
classifier= LinearSVC(random_state=0, tol=1e-5,dual=False)
learn_and_report('linear')

#SVM classification using quadratic kernel
#best value found of C=10000
classifier = SVC(C=10000.0,kernel='poly', degree=2, random_state = 42)
learn_and_report('quadratic')
