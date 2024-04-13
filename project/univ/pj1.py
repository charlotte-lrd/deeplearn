from scipy.io import loadmat
import numpy as np
from sklearn.metrics import classification_report
mat = loadmat(r'./data/digits.mat')

def one_hot(y):
    num_class = y.max(axis=0)[0]
    result = np.zeros((y.shape[0],num_class))
    for i in range(y.shape[0]):
        result[i,y[i][0]-1] = 1
    return result

X,y,Xtest,ytest,Xval,yval = mat['X'], one_hot(mat['y']), mat['Xtest'], one_hot(mat['ytest']), mat['Xvalid'], one_hot(mat['yvalid'])

class Model():
    def __init__(self, num_input, num_hiddens, num_output):
        self.num_input = num_input
        self.num_hiddens = num_hiddens # num_hiddens[i]: num of perceptrons in hidden layer[i]
        self.num_output = num_output
        
        # initialize weight;bias
        self.weights = []
        self.bias = []
        nums = [num_input] + num_hiddens + [num_output]
        for i in range(len(nums) - 1):
            self.weights.append(np.random.randn(nums[i], nums[i+1]))
            self.bias.append(np.random.randn(1,nums[i+1]))

        # initialize weight1,2, bias1,2
        self.weights2 = None
        self.bias2 = None
        self.weights1 = self.weights
        self.bias1 =self.bias

    def activate(self, x):
        return np.tanh(x)
    
    def softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self,X):
        return self.predict(X, forward=True)
    
    def predict(self, X, forward=False):
        h = X
        hidden_outputs = [h]
        num_weights = len(self.weights)
        for i in range(num_weights-1):
            h = self.activate(np.dot(h, self.weights[i])+self.bias[i])
            hidden_outputs.append(h)
        output = np.dot(h, self.weights[-1]) + self.bias[-1]
        if forward:
            return self.softmax(output),hidden_outputs
        else:
            return self.softmax(output)

    def cross_entropy_loss(self, y_pred, y_true):
        return -np.sum(y_true * np.log(y_pred)) / len(y_true)
    
    def grad_cross_entropy_loss(self, y_pred, y_true):
        return (y_pred - y_true) / len(y_true)
    
    def dropout(self, p=0.5):
        to_drop = []
        for i in range(X.shape[0]):
            root = np.random.uniform()
            if root < p:
                to_drop.append(i)
        return to_drop
    
    def backward(self,hidden_outputs,momentum_w, momentum_b,lr,beta,grad):
        grad_w = np.dot(hidden_outputs[-1].T, grad)
        grad_b = grad.sum(axis=0)
        self.weights[-1] -= (lr * grad_w - beta*momentum_w[-1])
        self.bias[-1] -= (lr * grad_b - beta*momentum_b[-1])
        for i in range(len(self.weights)-2, -1, -1):
            grad = np.dot(grad, self.weights[i+1].T) * (1 - hidden_outputs[i+1] ** 2)
            grad_w = np.dot(hidden_outputs[i].T, grad)
            grad_b = grad.sum(axis=0)
            self.weights[i] -= (lr * grad_w - beta*momentum_w[i])
            self.bias[i] -= (lr * grad_b - beta*momentum_b[i])

    def fit(self, X, y, Xval, yval, lr=0.01, beta = 0.9, epochs=1000, if_dropout=False, if_earlystop=False):
        for epoch in range(epochs):
            # forward
            Xtrain = X
            if if_dropout:
                to_drop = self.dropout()
                Xtrain = np.delete(X,to_drop, axis = 0)
                ytrain = np.delete(y,to_drop, axis = 0)
            _,hidden_outputs = self.forward(Xtrain)
            y_pred = self.predict(X)
            y_pred_val = self.predict(Xval)
    
            loss_test = self.cross_entropy_loss(y_pred, y)
            loss_val = self.cross_entropy_loss(y_pred_val,yval)
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch+1}: test loss: {loss_test} ; val loss: {loss_val}')
                if if_earlystop:
                    if epoch == 99:
                        loss_val_bf = loss_val
                    else:
                        print(loss_val,loss_val_bf)
                        if loss_val > loss_val_bf:
                            break
                        else:
                            loss_val_bf = loss_val
            # backward
            if epoch > 0:
                momentum_w = [self.weights1[i] - self.weights2[i] for i in range(len(self.weights1)) ]
                momentum_b = [self.bias1[i] - self.bias2[i] for i in range(len(self.bias1)) ]
            else:
                momentum_w = [np.zeros(self.weights1[i].shape) for i in range(len(self.weights1))]
                momentum_b = [np.zeros(self.bias1[i].shape) for i in range(len(self.bias1))]
            grad = self.grad_cross_entropy_loss(self.predict(Xtrain), ytrain)
            self.backward(hidden_outputs,momentum_w,momentum_b,lr,beta,grad)
            self.weights2 = self.weights1
            self.weights1 = self.weights
            self.bias2 = self.bias1
            self.bias1 = self.bias

num_hiddens = [16,16]
mlp = Model(X.shape[1],num_hiddens,10)
mlp.fit(X,y,Xval,yval,lr=1.0,epochs=1000,beta=0.9, if_dropout=True)
ypred1 = mlp.predict(Xtest).argmax(axis=1)
ypred2 = mlp.predict(X).argmax(axis=1)
print(classification_report(ypred2,y.argmax(axis=1)))
print(classification_report(ypred1,ytest.argmax(axis=1)))