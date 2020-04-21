import numpy as np
import random
import os
import cv2
from sklearn import preprocessing
import pandas as pd

def initializeW_B(layers):
    network = {}
    # +1 for bias
    for i in range(len(layers)-1):
    	layer = [[np.random.randn() for j in range(layers[i]+1)] for j in range(layers[i+1])]
    	network[i] = layer
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

    for i in range(len(layers)-1):
        temp = network[i]
        # print('feed ',np.shape(temp))
        newX =[]
        f_derivative = []
        for j in temp:
            netInput = j[-1:] #bias
            netInput+=np.dot(np.transpose(x),j[:-1])
            while( netInput >= 255):
                netInput = netInput/255 #normalization
            output = activationFn(netInput)
            newX.append(output)
            f_derivative.append(differentiate(output))
        x = newX
        list.append(newX)
        derivative.append(f_derivative)

    return list,derivative

def weightChange(n1,n2,delta,yin):
    learning_rate  = 0.25
    weight_change = []

    for i in range(n2):
        weight = []
        for j in range(n1):
            val = np.multiply(delta[i],np.multiply(yin[j],learning_rate))
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

def backPropagate(network,layers,fyin,derivative,t,x):

    #ouput and hidden weights change
    delta_k = []
    weight_change = []
    finalLayerDerivative = derivative[-1:][0]
    outputUnits = layers[-1:][0]
    output = fyin[-1:][0]
    for i in range(outputUnits):
        cal = np.multiply((t[i]-output[i]),finalLayerDerivative[i])
        delta_k.append(cal)
    weight_change.append(weightChange(layers[len(layers)-2],layers[len(layers)-1],delta_k,fyin[-2:-1][0]))

    #Hidden layers weight change
    k = len(layers)-2
    delta = delta_k
    while k>0:
        delta_inj = []
        weights = network[k]
        for i in range(layers[k]):
            cal=0
            for j in range(layers[k+1]):
                cal+= delta[j] * weights[j][i]
            delta_inj.append(cal)

        delta_hidden =[]
        hiddenLayerDerivative = derivative[k-1]

        for i in range(layers[k]):
            cal=np.multiply(delta_inj[i] ,hiddenLayerDerivative[i])
            delta_hidden.append(cal)

        #change in weights and biase b/w hidden Layers
        y = fyin[k-2]
        if k == 1:
            y = x
        weight_change.append(weightChange(layers[k-1],layers[k],delta_hidden,y))
        delta = delta_hidden
        k-=1

    weight_change = weight_change[::-1]

    # Update weights and biases
    for i in range(len(weight_change)):
        l1 = updateW_B(weight_change[i],network[i])
        network [i] = l1
    return network

def Train(network):
    images = []

    for i in classes:
        for f in os.listdir('Train/'+i):
            img = cv2.imread('Train/'+i+'/'+f)
            img = cv2.resize(img,(64,64),interpolation=cv2.INTER_AREA)
            images.append([cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),get_key(i)])
            # break

    random.shuffle(images)

    for i in range(10):                 #Epochs
        print("\n\nEpoch : ",i+1)
        hit = 0
        for x in images:
            input = np.divide(np.reshape(x[0],-1),255)
            expected = x[1]
            each_fyin,each_derivative = feedForward(np.transpose(input),network)
            outputLayer = each_fyin[-1:]
            h1 = outputLayer.index(max(outputLayer))
            h2 = expected.index(max(expected))
            if h1 == h2:
                hit+=1
            network = backPropagate(network,layers,each_fyin,each_derivative,expected,np.transpose(input))
            # print('\t\t\t-----------------------------------------------\t\t\n')
        print('Accuracy : ',hit/len(images))

    #storing weights b/w input and hidden
    #table = pd.DataFrame(network[0])
    #table.to_csv('input-hidden_weights.csv',index=False)

    #storing weights b/w hidden and output layers
    #table = pd.DataFrame(network[1])
    #table.to_csv('hidden-output_weights.csv',index=False)
    #print("Weights saved successfully..!")


    # TRAIN ACCURACY IS 89.5%


classes = ['Apple','Orange','Mango','Pomegranate']
layers = []

no_of_inputunits = 4096 #(64,64)
layers.append(no_of_inputunits)
print("Enter number of hidden layers")
n = int(input())
for i in range(n):
    layers.append(int(input()))
layers.append(len(classes))

#initialize network
network = initializeW_B(layers)

#One-hot encode for labeling o/p classes
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

Train(network)
