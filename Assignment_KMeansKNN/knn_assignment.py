from csv import reader

from collections import Counter

 

def euclidean_dist(row1, row2):

  distance = 0

  distance = sum([(x - y) ** 2 for x, y in zip(row1, row2)])

  return (distance**0.5)

 

def manhattan_dist(row1,row2):

  distance = 0

  distance = sum([(x - y) for x, y in zip(row1, row2)])

  return distance

 

def minkowski_dist(row1,row2):

  distance = 0

  distance = sum([abs(x - y) **100 for x, y in zip(row1, row2)])

  return distance **(1/100)

 

def k_nn(train, test, k, dist_fn):

  neighbours = []

  counter = -1

  list1 = []

  for row_dataset in train:

    distance = dist_fn(row_dataset[:-1], test)

    counter = counter + 1

    neighbours.append((distance, counter))

  neighbours.sort()

  k_neighbours = [neighbours[i] for i in range(0, k+1)]

  for d, i in k_neighbours:

    list1.append(train[i][8])

  return Counter(list1).most_common(1)[0][0]

 

def accuracy():

  tp, tn, fp, fn = 0, 0, 0, 0

  for i in range(0, len(predicted_list)):

    if expected_list[i] == predicted_list[i]:

      if expected_list[i] == '1':

        tp = tp + 1

      else:

        tn = tn + 1

    else:

      if expected_list[i] == '0':

        fp = fp + 1

      else:

        fn = fn + 1

  return (tp + tn) / (tp + tn + fp +fn)

 

url ='diabetes.csv'

train = []

test = []

predicted_list = []

expected_list = []

with open (url, 'r') as filename:

  file1 = reader(filename)

  count = 0

  for row in file1:

    count=  count + 1

    if count != 1 and count <= 615:

      train.append(row)

      count=count + 1

    elif count != 1 and count > 615:

      test.append(row)

      count = count + 1

for m in train:

  for n in range(0, len(m)):

    if m[n] == "":

      m[n] = float(0)

    else:

      m[n] = float(m[n])

for m in test:

  for n in range(0, len(m)):

    if m[n] == "":

      m[n] = float(0)

    else:

      m[n] = float(m[n])

for m in test:

  expected_list.append(m[-1])





#KNN classification

#euclidean distance

for m in range(0, len(test)):

  predicted_val = k_nn(train, test[m], k=31, dist_fn=euclidean_dist)

  #by trial, optimal value of k is found to be=31

  predicted_list.append(predicted_val)

 

accu=accuracy() *100

print("Accuracy of KNN classification using euclidean distance metric is " + str(accu))

 

#manhattan distance

predicted_list=[]

accu=0

for m in range(0, len(test)):

  predicted_val = k_nn(train, test[m], k=31, dist_fn=manhattan_dist)

  predicted_list.append(predicted_val)

 

accu=accuracy()*100

print("Accuracy of KNN classification using manhattan distance metric is " + str(accu))

 

# minkowski distance - large value of p=100 chosen

predicted_list=[]

accu=0

for m in range(0, len(test)):

  predicted_val = k_nn(train, test[m], k=31, dist_fn=minkowski_dist)

  predicted_list.append(predicted_val)

 

accu=accuracy()*100

print("Accuracy of KNN classification using minkowski distance metric with a value of p=100 is " + str(accu))

 

#normalisation

for j in range(0,7):

  xmin=train[0][j]

  xmax=train[0][j]

  for i in range(0, len(train)):

    if train[i][j]<xmin:

      xmin=train[i][j]

     

    elif train[i][j]>xmax:

      xmax=train[i][j]

     

  for i in range(0,len(train)):

    train[i][j]=(train[i][j]-xmin)/(xmax-xmin)

 

for j in range(0,7):

  xmin=test[0][j]

  xmax=test[0][j]

  for i in range(615, len(test)):

    if test[i][j]<xmin:

      xmin=test[i][j]

     

    elif test[i][j]>xmax:

      xmax=test[i][j]

     

  for i in range(0,len(test)):

    try:

      test[i][j]=(test[i][j]-xmin)/(xmax-xmin)

    except ZeroDivisionError:

      test[i][j]=0

 

#KNN with normalized dataset

predicted_list=[]

accu=0

for m in range(0, len(test)):

  predicted_val = k_nn(train, test[m], k=27, dist_fn=euclidean_dist)

  predicted_list.append(predicted_val)

 

accu=accuracy()*100

print("Accuracy of KNN classification after normalization of dataset is " + str(accu))

 

#feature ablation

predicted_list=[]

accu=0

#by experimentation, it is found that the feature on which predicted label is least dependent is 'Skin Thickness'. Hence that column has been removed

for row in train:

  row= row[0:3]+row[4:]

for m in range(0, len(test)):

  predicted_val = k_nn(train, test[m], k=27, dist_fn=euclidean_dist)

  predicted_list.append(predicted_val)

 

accu=accuracy()*100

print("Accuracy after performing feature ablation is " + str(accu))
