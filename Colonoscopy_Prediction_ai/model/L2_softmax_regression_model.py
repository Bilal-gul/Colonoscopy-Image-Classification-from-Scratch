import numpy as np 
import matplotlib.pyplot as plt
import preprocessing as pre
import time

class L2Softmax:
    def __init__(self,learning_rate = 0.01):
        self.learning_rate = learning_rate

        self.lambda_ = None
        self.weights = None
        self.bias = None
        self.loss_history = []

    def linear(self,matrix):
        return np.dot(matrix,self.weights) + self.bias

    def softmax_matrix(self,matrix):

        matrix = np.exp(matrix - np.max(matrix,axis=1,keepdims=True))
        
        return matrix / np.sum(matrix,axis=1,keepdims=True)

    def cost_regularization(self,one_hot,softmax_predict,m,L2):
        if L2:
            return - np.sum(one_hot * np.log(softmax_predict + 1e-15))/m + (self.lambda_*np.sum(self.weights**2))/(2*m)
        else:
            return - np.sum(one_hot * np.log(softmax_predict + 1e-15))/m 

    def show_plot(self,array,xlabel = None, ylabel = None, title = None):
        plt.plot(array)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid()
        plt.show()

    def fit(self,X_train_norm,one_hot_train,epoch,L2 = True):

        m,n = X_train_norm.shape
        n_classes = one_hot_train.shape[1]

        self.weights = np.zeros((n,n_classes))
        self.bias = np.zeros((1,n_classes))

        for h in range(epoch):

            basic_matrix_train = self.linear(X_train_norm)

            softmax_predict = self.softmax_matrix(basic_matrix_train)

            error = one_hot_train - softmax_predict

            gradient = np.dot(X_train_norm.T,error)/m

            if L2:
                self.weights += self.learning_rate*(gradient + (self.lambda_*self.weights)/m)
                self.bias += self.learning_rate*(np.sum(error,axis=0,keepdims=True)/m)
            else:
                self.weights += self.learning_rate*gradient
                self.bias += self.learning_rate*(np.sum(error,axis=0,keepdims=True)/m)

            if h % 500 == 0 : 
                cost_train = self.cost_regularization(one_hot_train,softmax_predict,m,L2)
                self.loss_history.append(cost_train)
                print(cost_train)

    
class CrossValidator(L2Softmax):
    def __init__(self,k = 6,learning_rate = 0.01):
        self.k = k
        self.learning_rate = learning_rate

        self.weights = None
        self.bias = None
        self.loss_history = []
        self.total_cost = []
        self.best_lambda = None
        self.lambda_ = None

    def find_best_lambda(self,X_dev_train,y_dev_train):

        lambda_ = [0.001,0.01,0.1,1,10,100]
        indices = np.arange(X_dev_train.shape[0])
                            
        for self.lambda_ in lambda_:

            lambda_start_time = time.time()
            validation_cost = []
            indices_dev = np.arange(22)

            for i in range(self.k):

                # set data
                indices_train = np.setdiff1d(indices,indices_dev)
                
                X_dev = X_dev_train[indices_dev]
                X_train = X_dev_train[indices_train]

                y_dev = y_dev_train[indices_dev]
                y_train = y_dev_train[indices_train]

                X_train_mean = np.mean(X_train,axis=0)
                X_train_std = np.std(X_train,axis=0)
                X_train_std[X_train_std == 0] = 1

                X_train_norm = (X_train - X_train_mean)/X_train_std
                X_dev_norm = (X_dev - X_train_mean)/X_train_std

                # one hot 
                n_classes = len(np.unique(y_train[:,0]))
                
                one_hot_train = pre.hot_encoding(y_train,n_classes)
                one_hot_dev = pre.hot_encoding(y_dev,n_classes)

                # fit model
                L2Softmax.fit(self,X_train_norm,one_hot_train,1000)

                # dev
                m_dev = X_dev_norm.shape[0]

                basic_matrix_dev = L2Softmax.linear(self,X_dev_norm)

                softmax_dev = L2Softmax.softmax_matrix(self,basic_matrix_dev)

                dev_cost = L2Softmax.cost_regularization(self,one_hot_dev,softmax_dev,m_dev,L2=False)

                validation_cost.append(dev_cost)

                # next fold
                indices_dev += 22
                
            mean_validation_cost = np.mean(validation_cost)  
            self.total_cost.append(mean_validation_cost)

            lambda_end_time = time.time()

            lambda_elapsed_time = (lambda_end_time - lambda_start_time)  

            print(f"lambda : {self.lambda_} =>  Validation Cost : {mean_validation_cost} =>  Time: {lambda_elapsed_time:.2f} sec")

        self.best_lambda = lambda_[np.argmin(self.total_cost)]
        return self.best_lambda
    


    

        