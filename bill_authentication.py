import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# 4 layered nueral network (3 hidden layer +1 output layer)

# getting the data
data = pd.read_csv("bill_authentication.csv")
x = data.drop(data.columns[-1], axis=1, inplace=False)
y = data.drop(data.columns[range(4)], axis=1, inplace=False)

# splitting into training and testing data set

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20)

nx = 4  # no. of nuerons in input layer
ny = 1  # no. of nuerons in output layer
layer_dims = np.array([4, 5, 5, 5, 1])  # no. of nuerons in each layer of network
m=X_train.shape[0] #no. of training examples

# initializing weights and bias of every layer
def initialize_parameter(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


"""# for checking the dimensions of weight and bias matrices
parameters=initialize_parameter(layer_dims)
for i in parameters:
    print(parameters[i].shape)"""


def sigmoid(z):
    s = 1 / (1 + np.exp(-1 * z))
    return s


def relu(z):
    s = np.maximum(0, z)
    return s

def relu_derivative(Z):
    Z[Z<=0]=0
    Z[Z>0]=1
    return Z



# Forward Propagation
def forward_prop(parameters, X_train):
    cache = {}
    L = len(layer_dims)
    for l in range(1, L):
        if l == 1:
            cache['Z' + str(l)] = np.dot(parameters['W' + str(l)], X_train.T) + parameters['b' + str(l)]
            cache['A' + str(l)] = relu(cache['Z' + str(l)])
        elif l == L - 1:
            cache['Z' + str(l)] = np.dot(parameters['W' + str(l)], cache['A' + str(l-1)]) + parameters['b' + str(l)]
            cache['A' + str(l)] = sigmoid(cache['Z' + str(l)])
        else:
            cache['Z' + str(l)] = np.dot(parameters['W' + str(l)], cache['A' + str(l-1)]) + parameters['b' + str(l)]
            cache['A' + str(l)] = relu(cache['Z' + str(l)])
    return cache


# for checking the dimensions of Z value
"""parameters=initialize_parameter(layer_dims)
cache=forward_prop(parameters,X_train)
for i in cache:
    print(cache[i].shape)"""

# cost function
def cost_func(cache,Y_train):
    L=len(layer_dims)
    J=np.sum(np.multiply(Y_train,np.log(cache['A'+str(L-1)]).T))+np.sum(np.multiply(1-Y_train,np.log(1-cache['A'+str(L-1)]).T))
    J=-J/m
    return J


# Back prop
def back_prop(cache,Y_train,parameters,X_train):
    L=len(layer_dims)
    grad={}
    for l in reversed(range(1,L)):
        if l==L-1:
            grad['dz'+str(l)]=cache['A'+str(l)]-Y_train.T
        else:
            grad['dz'+str(l)]=np.multiply(np.dot(parameters['W'+str(l+1)].T,grad['dz'+str(l+1)]),relu_derivative(cache['Z'+str(l)]))

        if l!=1:
            grad['dw' + str(l)] = (np.dot(grad['dz' + str(l)], cache['A' + str(l - 1)].T)) / m
            grad['db' + str(l)] = (np.sum(grad['dz' + str(l)], axis=1)) / m
            grad['db' + str(l)] = np.array(grad['db' + str(l)])
            grad['db' + str(l)] = grad['db' + str(l)].reshape(layer_dims[l],1)
        elif l==1:
            grad['dw' + str(l)] = (np.dot(grad['dz' + str(l)], X_train)) / m
            grad['db' + str(l)] = (np.sum(grad['dz' + str(l)], axis=1)) / m
            grad['db' + str(l)] = np.array(grad['db' + str(l)])
            grad['db' + str(l)] = grad['db' + str(l)].reshape(layer_dims[l], 1)


    return grad


# for checking dimension of dz,dw,db
"""grad=back_prop(cache,Y_train,parameters,X_train)
for i in grad:
    print(grad[i].shape)"""


def update(grad,parameters,alpha):
    L=len(layer_dims)
    for l in range(1,L):
        parameters['W' + str(l)]= parameters['W'+str(l)]-np.multiply(alpha,grad['dw'+str(l)])
        parameters['b' + str(l)]=parameters['b'+ str(l)]-np.multiply(alpha,grad['db'+str(l)])

    return parameters


# for checking the dimension of W and b
"""temp=update(grad,parameters,0.1)
for i in temp:
    print(temp[i].shape)"""

# for prediction on test set
def predict(X_test,parameters):
    temp=forward_prop(parameters,X_test)
    L=len(layer_dims)
    prediction=temp['A'+str(L-1)]
    prediction[prediction < 0.5] = 0
    prediction[prediction >= 0.5] = 1
    return prediction


# Model

def model(X_train,Y_train,epoch,learning_rate):

    parameters=initialize_parameter(layer_dims)
    for i in range(epoch):
        cache=forward_prop(parameters,X_train)
        print(cost_func(cache,Y_train))
        grad=back_prop(cache, Y_train, parameters, X_train)
        parameters=update(grad, parameters, learning_rate)
    return parameters



def accuracy(prediction,Y_test):
   total=len(prediction.T)
   temp=np.array(prediction-Y_test.T)
   temp=temp.T
   correct=0
   for i in temp:
       if i==0:
           correct+=1
   print((correct/total)*100)


# evaluating the nueral network
parameters=model(X_train,Y_train,10000,0.3)
prediction=predict(X_test,parameters)
#print(prediction.shape)


accuracy(prediction,Y_test)


