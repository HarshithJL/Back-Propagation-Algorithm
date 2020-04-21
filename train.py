import numpy as np
import random
import os
import cv2
from sklearn import preprocessing
import pandas as pd

def initializeW_B(n_of_input,n_of_hidden,no_of_output):
    network = {}
    # +1 for bias
    hidden1 = [[np.random.randn() for i in range(n_of_input+1)] for j in range(n_of_hidden)]
    network[0] = hidden1
    # hidden2  = [[np.random.uniform() for i in range(n_of_hidden+1)] for j in range(n_of_hidden)]
    # network[1] = hidden2
    output = [[np.random.randn() for i in range(n_of_hidden+1)] for j in range(no_of_output)]
    network[1]= output
    return network

def oneHotEncoding():
    le = preprocessing.LabelEncoder()
    label_encoded = le.fit_transform(classes)
    onehot_encode = preprocessing.OneHotEncoder(sparse = False)
    label_encoded = label_encoded.reshape( len(label_encoded),1)
    onehot_encode = onehot_encode.fit_transform(label_encoded)

    return onehot_encode

def get_key(val):
	for key,value in labels.items():
		if key == val:
			return value

def differentiate(f):
    return (np.multiply(f,(1-f)))

def activationFn(net):
    return (1/(1+np.exp(-net)))

def feedForward(x,network):
    list = [] #Contains f(yin) of each unit in a layer
    derivative = [] #Contains f'(yin) of each unit in a layer

    for i in range(hiddenLayers+1):
        temp = network[i]
        # print('feed ',np.shape(temp))
        newX = []
        f_derivative = []
        for j in temp:
            netInput = j[-1:] #bias
            netInput+=np.dot(np.transpose(x),j[:-1])
            # while( netInput >= 255):
            #     netInput = netInput/255 #normalization
            output = activationFn(netInput)
            newX.append(output)
            f_derivative.append(differentiate(output))
        x = newX
        list.append(newX)
        derivative.append(f_derivative)

    return x,list[:-1],derivative

def weightChange(n1,n2,delta,yin):
    learning_rate  = 0.25
    weight_change = []

    for i in range(n2):
        weight = []
        for j in range(n1):
            val = np.multiply(delta[i],np.multiply(yin[j],learning_rate))#delta is the error calculated and yin is the input from the previous layer
            weight.extend(val)
        bias = np.multiply(delta[i],learning_rate)
        weight.extend(bias)
        weight_change.append(weight)

    return weight_change

def updateW_B(weight_change,weights_old):
    weights_new =[]
    for i,j in zip(weight_change,weights_old):
        new = [sum(k) for k in zip(i,j)]
        weights_new.append(new)

    return weights_new

def backPropagate(network,finalLayerOutput,yin,derivative,t,x):

    #output Layer
    delta_k = []
    finalLayerDerivative = derivative[-1:][0]
    for i in range(no_of_outputunits):

        # print('Expected - obtained : ',(t[i]-finalLayerOutput[i]))
        cal = np.multiply((t[i]-finalLayerOutput[i]),finalLayerDerivative[i])
        delta_k.append( cal)

    #change in weights b/w hidden and outputLayer
    weightChange1 = weightChange(no_of_hiddenunits,no_of_outputunits,delta_k,yin[0])

    #Hidden layer
    #finding delta_inj
    delta_inj = []
    weights = network[1]
    for i in range(no_of_hiddenunits):
        cal=0
        for j in range(no_of_outputunits):
            cal+= delta_k[j] * weights[j][i]
        delta_inj.append(cal)

    delta_hidden =[]
    hiddenLayerDerivative = derivative[0]
    for i in range(no_of_hiddenunits):
        cal=np.multiply(delta_inj[i] ,hiddenLayerDerivative[i])
        delta_hidden.append(cal)

    #change in weights and biase b/w input and hidden Layer
    weightChange2 = weightChange(no_of_inputunits,no_of_hiddenunits,delta_hidden,x)

    #Upadate weights and biases b/w hidden and output
    l1 = updateW_B(weightChange1,network[1])
    network[1] = l1

    #update weights and biases b/w input and hidden
    l2 = updateW_B(weightChange2,network[0])
    network[0] = l2

    return network

def Train():
    images = []
    for i in classes:
        for f in os.listdir(r'C:\Users\LENOVO\harshith\programs\ML\Neural net project\Train'+'/'+i):
            img = cv2.imread(r'C:\Users\LENOVO\harshith\programs\ML\Neural net project\Train'+'/'+i+'/'+f)
            img = cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_AREA)
            images.append([cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),get_key(i)])
            k+=1
            # if(k==40):
                # break

    random.shuffle(images)
    #initialize network
    network = initializeW_B(no_of_inputunits,no_of_hiddenunits,no_of_outputunits)   #no of input units,no of hidden units,no of output units
    accuracy = 0
    for i in range(10):                 #Epochs
        print("\n\nEpoch : ",i+1)
        hit = 0
        for x in images:
            input = np.divide(np.reshape(x[0],-1),255)
            expected = x[1]
            outputLayer,each_yin,each_derivative = feedForward(np.transpose(input),network)
            h1 = outputLayer.index(max(outputLayer))
            h2 = expected.index(max(expected))
            if h1 == h2:
                hit+=1
            network = backPropagate(network,outputLayer,each_yin,each_derivative,expected,np.transpose(input))
            # print('\t\t\t-----------------------------------------------\t\t\n')
        # per = accuracy
        accuracy = hit/len(images)
        print(str(accuracy*100)+'%')
        # if(accuracy*100 >= 95):
            # break
        # if(per == accuracy):
            # break

    # storing weights b/w input and hidden
    table = pd.DataFrame(network[0])
    table.to_csv('x.csv',index=False)

    #storing weights b/w hidden and output layers
    table = pd.DataFrame(network[1])
    table.to_csv('y.csv',index=False)
    print("Weights saved successfully..!")

hiddenLayers = 1
classes = ['Mango','Pomegranate','Orange','Guava']
# classes = ['Guava','Pear']
img_size = 16
no_of_inputunits = img_size*img_size
no_of_hiddenunits = 25
no_of_outputunits = len(classes)

t = oneHotEncoding()

labels = {}
target =[]

for x in t:
    target.append(x)

for x,y in zip(classes,t):
    l = []
    for z in y:
        l.append(z)
    labels[x] = l

Train()
# TRAIN ACCURACY IS 89.5%
