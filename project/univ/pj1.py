from scipy.io import loadmat
import numpy as np
from sklearn.metrics import classification_report
mat = loadmat(r'project\univ\data\digits.mat')

def one_hot(y):
    num_class = y.max(axis=0)[0]
    result = np.zeros((y.shape[0],num_class))
    for i in range(y.shape[0]):
        result[i,y[i][0]-1] = 1
    return result

def resize(img, scale_factor):
    new_height = int(img.shape[0] * scale_factor)
    new_width = int(img.shape[1] * scale_factor)
    resized_img = np.zeros((new_height, new_width))
    for i in range(new_height):
        for j in range(new_width):
            resized_img[i, j] = img[int(i / scale_factor), int(j / scale_factor)]
    return resized_img

def translate(img, tx, ty):
    translated_img = np.zeros_like(img)
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            if 0 <= i + tx < rows and 0 <= j + ty < cols:
                translated_img[i + tx, j + ty] = img[i, j]
    return translated_img

def rotate(img, angle):
    rotated_img = np.zeros_like(img)
    rows, cols = img.shape
    center_x, center_y = rows // 2, cols // 2
    for i in range(rows):
        for j in range(cols):
            x = (i - center_x) * np.cos(np.deg2rad(angle)) - (j - center_y) * np.sin(np.deg2rad(angle)) + center_x
            y = (i - center_x) * np.sin(np.deg2rad(angle)) + (j - center_y) * np.cos(np.deg2rad(angle)) + center_y
            if 0 <= x < rows and 0 <= y < cols:
                rotated_img[int(x), int(y)] = img[i, j]
    return rotated_img

def transform(X,y,scale_factor=1.0,if_rotate=False,if_translate=False,p=0.2):
    '''
    resize img according to scale_factor
    if_rotate=True, p chance to rotate each img randomly in an angle within(0,360), otherwise do nothing
    if_translate=True, p chance to translate each img randomly in (tx, ty), otherwise do nothing
    '''
    num_img = X.shape[0]
    X = X.reshape(-1,16,16)
    for i in range(num_img):
        X[i,:,:] = resize(X[i,:,:],scale_factor)

    if if_rotate:
        for i in range(num_img):
            if np.random.uniform() < p:
                angle = np.random.randint(0,360)
                Xnew = rotate(X[i,:,:],angle)
                X = np.concatenate((X, np.expand_dims(Xnew, axis=0)), axis=0)
                y = np.concatenate((y, np.expand_dims(y[i], axis=0)), axis=0)
    if if_translate:
        for i in range(num_img):
            if np.random.uniform() < p:
                tx, ty = np.random.randint(0,16,size=2)
                Xnew = translate(X[i,:,:], tx, ty)
                X = np.concatenate((X, np.expand_dims(Xnew, axis=0)), axis=0)
                y = np.concatenate((y, np.expand_dims(y[i], axis=0)), axis=0)
    return X.reshape(X.shape[0],-1),y

X,y,Xtest,ytest,Xval,yval = mat['X'], one_hot(mat['y']), mat['Xtest'], one_hot(mat['ytest']), mat['Xvalid'], one_hot(mat['yvalid'])
X,y = transform(X,y,if_translate=True,if_rotate=True)

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
        self.norm2 = None
    
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

    def cross_entropy_loss(self, y_pred, y_true, c):
        return -np.sum(y_true * np.log(y_pred)) / len(y_true) + c*np.sqrt(self.norm2)
    
    def grad_cross_entropy_loss(self, y_pred, y_true):
        return (y_pred - y_true) / len(y_true)
    
    def dropout(self, p=0.5):
        to_drop = []
        for i in range(X.shape[0]):
            root = np.random.uniform()
            if root < p:
                to_drop.append(i)
        return to_drop
    
    def backward(self,hidden_outputs,momentum_w, momentum_b,lr,beta,grad,c):
        norm2 = self.norm2
        grad_w = np.dot(hidden_outputs[-1].T, grad) + c/np.sqrt(norm2)*self.weights[-1]
        grad_b = grad.sum(axis=0)+ c/np.sqrt(norm2)*self.bias[-1]
        self.weights[-1] -= (lr * grad_w - beta*momentum_w[-1])
        self.bias[-1] -= (lr * grad_b - beta*momentum_b[-1])
        for i in range(len(self.weights)-2, -1, -1):
            grad = np.dot(grad, self.weights[i+1].T) * (1 - hidden_outputs[i+1] ** 2)
            grad_w = np.dot(hidden_outputs[i].T, grad) + c/np.sqrt(norm2)*self.weights[i]
            grad_b = grad.sum(axis=0) + c/np.sqrt(norm2)*self.bias[i]
            self.weights[i] -= (lr * grad_w - beta*momentum_w[i])
            self.bias[i] -= (lr * grad_b - beta*momentum_b[i])

    def fit(self, X, y, Xval, yval, lr=0.01, beta = 0.9, c = 0.05, epochs=1000, dropout=0, if_earlystop=False):
        for epoch in range(epochs):
            # forward
            Xtrain = X
            if dropout != 0:
                to_drop = self.dropout(dropout)
                Xtrain = np.delete(X,to_drop, axis = 0)
                ytrain = np.delete(y,to_drop, axis = 0)
            _,hidden_outputs = self.forward(Xtrain)
            y_pred = self.predict(X)
            y_pred_val = self.predict(Xval)
            self.norm2 = 0
            for param in self.weights + self.bias:
                self.norm2 += np.linalg.norm(param,ord=2)**2
            loss_test = self.cross_entropy_loss(y_pred, y,c)
            loss_val = self.cross_entropy_loss(y_pred_val,yval,c)
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch+1}: test loss: {loss_test} ; val loss: {loss_val}')
                if if_earlystop:
                    if epoch == 99:
                        loss_val_bf = loss_val
                    else:
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
            self.backward(hidden_outputs,momentum_w,momentum_b,lr,beta,grad,c)
            self.weights2 = self.weights1
            self.weights1 = self.weights
            self.bias2 = self.bias1
            self.bias1 = self.bias


num_hiddens = [64,16]
mlp = Model(X.shape[1],num_hiddens,10)
mlp.fit(X,y,Xval,yval,lr=0.7,epochs=2000,beta=0.5, dropout=0.2, c=0.03, if_earlystop=True)
ypred1 = mlp.predict(Xtest).argmax(axis=1)
ypred2 = mlp.predict(X).argmax(axis=1)
print(classification_report(ypred2,y.argmax(axis=1)))
print(classification_report(ypred1,ytest.argmax(axis=1)))