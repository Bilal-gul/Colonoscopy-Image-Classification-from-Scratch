import numpy as np
import pandas as pd 
import time 
import model.preprocessing as pre
import model.L2_softmax_regression_model as L2srm

df = pd.read_csv("Colonoscopy_Prediction_ai\\data\\data.txt",header=None).T

X_test,y_test,X_dev_train,y_dev_train = pre.split_data(df) # set data

start_time = time.time()

model = L2srm.CrossValidator()
best_lambda = model.find_best_lambda(X_dev_train,y_dev_train) # find best lambda
np.save("best_lambda.npy", best_lambda) # save best lambda

end_time = time.time()

elapsed_time = end_time - start_time

print("\n" + "="*60)
print("Best lambda:", best_lambda)
print(f"\nTotal elapsed time: {elapsed_time:.2f} seconds")
print(f"Total elapsed time: {elapsed_time / 60:.2f} minutes")

















































































# names = df.iloc[:,0].values
# df = df.iloc[:,1:].astype('float64')

# X = df.iloc[:,1:].values
# y = df.iloc[:,0].values.reshape(-1,1)

# indices = []

# unique_names = np.unique(names)

# np.random.seed(42)
# np.random.shuffle(unique_names)

# for name in unique_names:
#     temp = np.where(names == name)[0]
#     indices.extend(temp)

# X = X[indices]
# y = y[indices]

# X_test = X[-20:,:]
# y_test = y[-20:,:]

# X_dev_train = X[:-20,:]
# y_dev_train = y[:-20,:]
# ###########################################################3

# k = 6
# indices = np.arange(X_dev_train.shape[0])
# lambda_ = [0.001,0.01,0.1,1,10,100]

# start_time = time.time()

# for l in lambda_:
#     lambda_start_time = time.time()
#     validation_cost = []
#     indices_dev = np.arange(22)

#     for i in range(k):
    
#         indices_train = np.setdiff1d(indices,indices_dev)
        
#         X_dev = X_dev_train[indices_dev]
#         X_train = X_dev_train[indices_train]

#         y_dev = y_dev_train[indices_dev]
#         y_train = y_dev_train[indices_train]

#         X_train_mean = np.mean(X_train,axis=0)
#         X_train_std = np.std(X_train,axis=0)
#         X_train_std[X_train_std == 0] = 1

#         X_train_norm = (X_train - X_train_mean)/X_train_std
#         X_dev_norm = (X_dev - X_train_mean)/X_train_std

#         ##############################################################

#         # one hot 

#         m = X_train.shape[0]
#         n = X_train.shape[1]
#         n_classes = len(np.unique(y_train[:,0]))
        
#         one_hot_train = np.zeros((m,n_classes))
#         one_hot_dev = np.zeros((y_dev.shape[0],n_classes))

#         for j in range(m):
#             one_hot_train[j,int(y_train[j,0])-1] = 1

#         #################################################################


#         w = np.zeros((n,n_classes))
#         b = np.zeros((1,n_classes))
#         learning_rate = 0.01
#         epoch = 100
#         loss_history = []

#         for t in range(epoch):

#             basic_matrix = np.dot(X_train_norm,w) + b

#             basic_matrix = np.exp(basic_matrix - np.max(basic_matrix,axis=1,keepdims=True))

#             softmax_predict = basic_matrix / np.sum(basic_matrix,axis=1,keepdims=True)

#             error = one_hot_train - softmax_predict

#             gradient = np.dot(X_train_norm.T,error)/m

#             w += learning_rate*(gradient + (l*w)/m)
#             b += learning_rate*(np.sum(error,axis=0,keepdims=True)/m)

#             ####cost####

#             if t % 10 == 0:
#                 cost = - np.sum(one_hot_train * np.log(softmax_predict + 1e-15))/m + (l*np.sum(w**2))/(2*m)
#                 loss_history.append(cost)
#                 #print(f"lambda: {l} =>  fold: {i+1} =>  cost descent: {cost}")

#         if l == 0.001 and i == 0:
#             plt.plot(loss_history)
#             plt.xlabel("Training Checkpoint")
#             plt.ylabel("Cost")
#             plt.title("Training Cost")
#             plt.show()
                

#         ################################################################

#         m_dev = X_dev_norm.shape[0]
        
#         basic_matrix_dev = np.dot(X_dev_norm,w) + b

#         basic_matrix_dev = np.exp(basic_matrix_dev - np.max(basic_matrix_dev,axis=1,keepdims=True))

#         softmax_dev = basic_matrix_dev / np.sum(basic_matrix_dev,axis=1,keepdims=True)

#         # one hot dev 

#         for z in range(m_dev):
#             one_hot_dev[z,int(y_dev[z,0])-1] = 1

#         dev_cost = - np.sum(one_hot_dev * np.log(softmax_dev + 1e-15))/m_dev 

#         validation_cost.append(dev_cost)

#         #################################################################
#         indices_dev += 22
        
#     mean_validation_cost = np.mean(validation_cost)  

#     lambda_end_time = time.time()

#     lambda_elapsed_time = (lambda_end_time - lambda_start_time)  

#     print(f"lambda : {l} =>  Validation Cost : {mean_validation_cost} =>  Time: {lambda_elapsed_time:.2f} sec")

# end_time = time.time()

# elapsed_time = end_time - start_time

    # print("\n" + "="*60)
    # print(f"\nTotal elapsed time: {elapsed_time:.2f} seconds")
    # print(f"Total elapsed time: {elapsed_time / 60:.2f} minutes")


# ##########################full train##########################################

# epoch2 = 15000
# loss_history2 = []

# m_dev_train = X_dev_train.shape[0]

# X_dev_train_norm = (X_dev_train - X_train_mean) / X_train_std

# for h in range(epoch2):

#     basic_matrix_dev_train =  np.dot(X_dev_train_norm,w) + b

#     basic_matrix_dev_train = np.exp(basic_matrix_dev_train - np.max(basic_matrix_dev_train,axis=1,keepdims=True))

#     softmax_dev_train = basic_matrix_dev_train / np.sum(basic_matrix_dev_train,axis=1,keepdims=True)

#     one_hot_dev_train = np.zeros((X_dev_train.shape[0],3))

#     for c in range(X_dev_train.shape[0]):
#         one_hot_dev_train[c,int(y_dev_train[c,0])-1] = 1

#     error_dev_train = one_hot_dev_train - softmax_dev_train

#     gradient_dev_train = np.dot(X_dev_train_norm.T,error_dev_train)/m_dev_train

#     w += learning_rate*(gradient_dev_train + (0.001*w)/m_dev_train)
#     b += learning_rate*(np.sum(error_dev_train,axis=0,keepdims=True)/m_dev_train)

#     if h % 10 == 0 : 
#         cost_dev_train = - np.sum(one_hot_dev_train * np.log(softmax_dev_train + 1e-15))/m_dev_train + (0.001*np.sum(w**2))/(2*m_dev_train)
#         loss_history2.append(cost_dev_train)
#         print(cost_dev_train)



# plt.plot(loss_history2)
# plt.xlabel("Training Checkpoint")
# plt.ylabel("Cost")
# plt.title("Training Cost")
# plt.show()


################################33test#############################################3


