import numpy as np
import pandas as pd
import model.L2_softmax_regression_model as l2srm
import model.metrics as met
import model.preprocessing as pre 



df = pd.read_csv("Colonoscopy_Prediction_ai\\data\\data.txt",header=None).T
best_lambda = np.load("best_lambda.npy") # load best lambda

# set data
X_test,y_test,X_train,y_train = pre.split_data(df)
X_train_norm,X_train_mean,X_train_std = pre.normalization(X_train)
one_hot_train = pre.hot_encoding(y_train,3)


# fit model
model = l2srm.L2Softmax()
model.lambda_ = best_lambda
model.fit(X_train_norm,one_hot_train,15000)


# test
X_test_norm = (X_test - X_train_mean) / X_train_std
basic_matrix_test = model.linear(X_test_norm)
softmax_test = model.softmax_matrix(basic_matrix_test)
one_hot_test = pre.hot_encoding(y_test,3)
cost_test = model.cost_regularization(one_hot_test,softmax_test,X_test.shape[0],True)


# evaluation metrics
predict_matrix = np.argmax(softmax_test,axis=1,keepdims=True) +1 

confusion_matrix,TP,FP,FN,TN,total = met.confusion_matrix(y_test,predict_matrix)

accuracy = met.accuracy(confusion_matrix,total)
precision = met.precision(TP,FP)
recall = met.recall(TP,FN)
f1 = met.f1(precision,recall)

# result
print("\n" + "="*60)
print("        SOFTMAX REGRESSION EVALUATION RESULTS")
print("="*60)

print(f"\nAccuracy          : {accuracy:.4f}")
print(f"Macro Precision   : {np.mean(precision):.4f}")
print(f"Macro Recall      : {np.mean(recall):.4f}")
print(f"Macro F1 Score    : {np.mean(f1):.4f}")
print(f"Test Cost         : {cost_test:.4f}")

print("\nConfusion Matrix")
print("-"*60)
print(confusion_matrix)
print("-"*60)

print("\nPer-Class Metrics")
print("-"*60)

for i in range(3):
    print(f"Class {i+1}")
    print(f"  TP : {int(TP[0,i])}")
    print(f"  FP : {int(FP[0,i])}")
    print(f"  FN : {int(FN[0,i])}")
    print(f"  TN : {int(TN[0,i])}")
    print(f"  Precision : {precision[0,i]:.4f}")
    print(f"  Recall    : {recall[0,i]:.4f}")
    print(f"  F1 Score  : {f1[0,i]:.4f}")
    print("-"*60)

print("Test class distribution:")
print(np.unique(y_test, return_counts=True))

print("Train class distribution:")
print(np.unique(y_train, return_counts=True))

model.show_plot(model.loss_history) 
met.confusion_matrix_chart(confusion_matrix)

























































































# names = df.iloc[:,0].values
# df = df.iloc[:,1:].astype('float64')

# X = df.iloc[:,1:].values
# y = df.iloc[:,0].values.reshape(-1,1)

# indices = []

# unique_names = np.unique(names)

# np.random.seed(7)
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

# X_dev_train_mean = np.mean(X_dev_train,axis=0,keepdims=True)
# X_dev_train_std = np.std(X_dev_train,axis=0,keepdims=True)
# X_dev_train_std[X_dev_train_std == 0] = 1

# X_dev_train_norm = (X_dev_train - X_dev_train_mean) /  X_dev_train_std

# epoch2 = 15000
# loss_history2 = []
# learning_rate = 0.01
# w = np.zeros((X_dev_train.shape[1],3))
# b = np.zeros((1,3))
# lambda_ = 0.001


# m_dev_train = X_dev_train.shape[0]


# for h in range(epoch2):

#     basic_matrix_dev_train =  np.dot(X_dev_train_norm,w) + b

#     basic_matrix_dev_train = np.exp(basic_matrix_dev_train - np.max(basic_matrix_dev_train,axis=1,keepdims=True))

#     softmax_dev_train = basic_matrix_dev_train / np.sum(basic_matrix_dev_train,axis=1,keepdims=True)

#     one_hot_dev_train = np.zeros((X_dev_train.shape[0],3))

#     for c in range(X_dev_train.shape[0]):
#         one_hot_dev_train[c,int(y_dev_train[c,0])-1] = 1

#     error_dev_train = one_hot_dev_train - softmax_dev_train

#     gradient_dev_train = np.dot(X_dev_train_norm.T,error_dev_train)/m_dev_train

#     w += learning_rate*(gradient_dev_train + (lambda_*w)/m_dev_train)
#     b += learning_rate*(np.sum(error_dev_train,axis=0,keepdims=True)/m_dev_train)

#     if h % 10 == 0 : 
#         cost_dev_train = - np.sum(one_hot_dev_train * np.log(softmax_dev_train + 1e-15))/m_dev_train + (lambda_*np.sum(w**2))/(2*m_dev_train)
#         loss_history2.append(cost_dev_train)
#         print(cost_dev_train)



# plt.plot(loss_history2)
# plt.xlabel("Training Checkpoint")
# plt.ylabel("Cost")
# plt.title("Training Cost")
# plt.show()


# ################################33test#############################################3

# X_test_norm = (X_test - X_dev_train_mean) / X_dev_train_std

# basic_matrix_test = np.dot(X_test_norm,w) + b

# basic_matrix_test = np.exp(basic_matrix_test - np.max(basic_matrix_test,axis=1,keepdims=True))

# softmax_test = basic_matrix_test / np.sum(basic_matrix_test,axis=1,keepdims=True)


# predict_matrix = np.argmax(softmax_test,axis=1,keepdims=True) +1 

# one_hot_test = np.zeros((y_test.shape[0],3))

# for p in range(y_test.shape[0]):
#     one_hot_test[p,int(y_test[p,0])-1] = 1

# m_test = X_test.shape[0]

# cost_test = - np.sum(one_hot_test * np.log(softmax_test + 1e-15))/m_test + (lambda_*np.sum(w**2))/(2*m_test)
# ######################################################33333

# confusion_matrix = np.zeros((3,3))
# TP = np.zeros((1,3))
# FP = np.zeros((1,3))
# TN = np.zeros((1,3))
# FN = np.zeros((1,3))

# for j in range(y_test.shape[0]):
#     confusion_matrix[int(y_test[j,0])-1,int(predict_matrix[j,0])-1] += 1 

# total = np.sum(confusion_matrix)

# for k in range(3):
#     TP[0,k] = confusion_matrix[k,k]

#     FP[0,k] = np.sum(confusion_matrix[:,k]) - TP[0,k]

#     FN[0,k] = np.sum(confusion_matrix[k,:]) - TP[0,k]

#     TN[0,k] = total - TP[0,k] - FP[0,k] - FN[0,k]

# accuracy = np.sum(np.diag(confusion_matrix)) / total
# recall = TP / (TP+FN+1e-10)
# precision = TP / (TP+FP+1e-10)
# f1 = 2*precision*recall / (precision+recall+1e-10)

# print("\n" + "="*60)
# print("        SOFTMAX REGRESSION EVALUATION RESULTS")
# print("="*60)

# print(f"\nAccuracy          : {accuracy:.4f}")
# print(f"Macro Precision   : {np.mean(precision):.4f}")
# print(f"Macro Recall      : {np.mean(recall):.4f}")
# print(f"Macro F1 Score    : {np.mean(f1):.4f}")
# print(f"Test Cost         : {cost_test:.4f}")

# print("\nConfusion Matrix")
# print("-"*60)
# print(confusion_matrix)
# print("-"*60)

# print("\nPer-Class Metrics")
# print("-"*60)

# for i in range(3):
#     print(f"Class {i+1}")
#     print(f"  TP : {int(TP[0,i])}")
#     print(f"  FP : {int(FP[0,i])}")
#     print(f"  FN : {int(FN[0,i])}")
#     print(f"  TN : {int(TN[0,i])}")
#     print(f"  Precision : {precision[0,i]:.4f}")
#     print(f"  Recall    : {recall[0,i]:.4f}")
#     print(f"  F1 Score  : {f1[0,i]:.4f}")
#     print("-"*60)

# print("Test class distribution:")
# print(np.unique(y_test, return_counts=True))

# print("Train class distribution:")
# print(np.unique(y_dev_train, return_counts=True))