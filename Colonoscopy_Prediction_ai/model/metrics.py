import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(y,predict):
    confusion_matrix = np.zeros((3,3))
    TP = np.zeros((1,3))
    FP = np.zeros((1,3))
    TN = np.zeros((1,3))
    FN = np.zeros((1,3))

    for j in range(y.shape[0]):
        confusion_matrix[int(y[j,0])-1,int(predict[j,0])-1] += 1 

    total = np.sum(confusion_matrix)

    for k in range(3):
        TP[0,k] = confusion_matrix[k,k]

        FP[0,k] = np.sum(confusion_matrix[:,k]) - TP[0,k]

        FN[0,k] = np.sum(confusion_matrix[k,:]) - TP[0,k]

        TN[0,k] = total - TP[0,k] - FP[0,k] - FN[0,k]


    return confusion_matrix,TP,FP,FN,TN,total

def accuracy(confusion_matrix,totalSum):
    return np.sum(np.diag(confusion_matrix)) / totalSum

def recall(TP,FN):
    return TP / (TP+FN+1e-10)

def precision(TP,FP):
    return TP / (TP+FP+1e-10)

def f1(pre,rcl):
    return 2*pre*rcl / (pre+rcl+1e-10)

def confusion_matrix_chart(confusion_matrix):

    n = confusion_matrix.shape[0]

    plt.figure(figsize=(8,6))

    plt.imshow(confusion_matrix,cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.xticks(range(n),range(1,n+1))
    plt.yticks(range(n),range(1,n+1))

    plt.colorbar()

    threshold = confusion_matrix.max()/2

    for i in range(n):
        for j in range(n):

            plt.text(
                j,
                i,
                int(confusion_matrix[i,j]),
                ha="center",
                va="center",
                color="white" if confusion_matrix[i,j] > threshold else "black"
            )

    plt.tight_layout()
    plt.show()