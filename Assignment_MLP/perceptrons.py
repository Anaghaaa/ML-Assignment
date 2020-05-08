#Hand-written code without using existing libraries
#Dataset is shuffled to get accurate results
import csv
import random
inputs,weights,answers=[],[],[]
epochs,l_rate=1000,0.01  
def activation(sum1): #Returns 1 if sum>0 
    if sum1>=0:
        return 1
    else:
        return 0
    
def predict(inputs1): #This calculates the sum which is passed to activation function
    sum=0.0
    for i in range(1,len(weights)):
        sum=sum +( float(weights[i]) * float(inputs1[i-1]) )
    sum=sum + float(weights[0])
    return activation(sum)
with open( 'iris.csv', 'r', newline='') as m: #This is used to read the csv file line by line
    read = csv.reader(m)
    for c, r in enumerate(read):
        if c<80:                               # 80:20 is the split ratio for training and test data
            inputs.append(r[:4])
            answers.append(*r[4:])

length=len(inputs)
weights.append(1) #bias
for i in range(1,len(inputs[0])+1):
    weights.append(random.random()) #Random weights are generated. Bias is the first weight here
   
print("Initial weights(including bias):",*weights,sep="\n")

for x in range(0,epochs):
    for y in range(0,length):
        prediction=predict(inputs[y])
        error=float(answers[y])- float(prediction) #Error= correct_output-predicted_output
        weights[0]= weights[0] + (l_rate * float(error))
        for z in range(1,len(weights)):
            weights[z]= weights[z]+ ((l_rate * float(error)  * float(inputs[y][z-1])))
print("\n")  
print("Final weights after training(used for test data):", *weights,sep="\n") #Weights that are used to test the test data
print("\n")
print("Testing process on test data(20 records):")
inp,ans,wt=[],[],[]
with open( 'iris.csv', 'r', newline='') as m:
    read = csv.reader(m)
    for c, r in enumerate(read):
        if c>=80:
            inp.append(r[:4])
            ans.append(*r[4:])
list1=[]
tp,tn,fp,fn=0,0,0,0
for g in range(len(inp)):
    predict1=predict(inp[g])
    list1.append(str(predict1))
    print("For input\t", *inp[g])
    print("Classified class is",predict1) #Predicted output
    print()
for i in range(0,len(ans)):
    if list1[i]==ans[i]:
        if list1[i]=='1':
            tp=tp+1
        else:
            tn=tn+1

print("Confusion matrix:")
print("TP=",tp,end="\t") #True positive
print("FN=",fn,end="\t") #False negative
print()
print("FP=",fp,end="\t") #Flase positive
print("TN=",tn,end="\t") #True negative
print("\n")
acc= (tp+tn) / (tp+tn+fp+fn)
rec= (tp)/(tp+fn)
pre= (tp)/(tp+fp)
f_mea= (float(2)*rec*pre)/(rec+pre)

print("Accuracy is", acc*100,"%")
print("Recall is", rec*100,"%")
print("Precision is", pre*100,"%")
print("F-score is", f_mea*100,"%")
