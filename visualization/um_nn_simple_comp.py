import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN
import numpy as np
import torch as t

data_path = 'D:/datasets/HKS/data/multimodal-plot-data.npy'
figsize = (10,9)
mode = 'SIMPLE'

def SIMPLE(X, Y, init_proto, init_proto_label, iter=3, sigma=0.5):
    def cal_assigned_logits(data, protos):
        norm = -np.log(sigma+2*np.pi)
        data_len = len(data)
        proto_len = len(protos)
        data = np.expand_dims(data, 1).repeat(proto_len,1).reshape(data_len*proto_len,-2)
        protos = np.expand_dims(protos, 0).repeat(data_len,0).reshape(data_len*proto_len,-2)
        logits = -((data-protos)**2).sum(-1)/(2*sigma) - norm
        logits = logits.reshape(data_len, proto_len)
        logits = t.softmax(t.Tensor(logits),dim=1).numpy()
        return logits

    protos = init_proto
    proto_labels = init_proto_label

    for i in range(iter):
        for x,y in zip(X,Y):
            min_proto_index = np.argmin(((x-protos)**2).sum(axis=1))
            if proto_labels[min_proto_index] != y:
                protos = np.concatenate((protos, np.expand_dims(x,0)), axis=0)
                proto_labels = np.append(proto_labels,y)

        # shape: [data, proto] -> [proto,data]
        logits_ = cal_assigned_logits(X,protos).T
        logits_ = np.expand_dims(logits_, -1).repeat(2,-1)
        weighted_data = np.expand_dims(X,0).repeat(len(protos),0)
        protos = (logits_*weighted_data).sum(1) / logits_.sum(1)

    return protos, proto_labels

def plot_decision_boundary(model, X, Y, proto=None, proto_label=None):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    # cmap = (plt.cm.Pastel1.colors[i] for i in [0,2,3,5,6])
    plt.figure(figsize=figsize)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral) #Spectral, RdYlBu
    if proto is not None:
        plt.scatter(proto[:, 0], proto[:, 1], c=proto_label, cmap=plt.cm.Spectral, marker="*",
                    s=180, edgecolors='k')
    plt.scatter(X[:, 0], X[:, 1], c=Y.ravel(), cmap=plt.cm.Spectral, edgecolors='k')
    plt.axis('off')
    # plt.ylabel('x2')
    # plt.xlabel('x1')

    plt.show()

knn = KNN(n_neighbors=1)
# x = np.array([[2,2],
#               [2,1],
#               [1,3],
#               [-2,1],
#               [-1,2],
#               [-2,-3],
#               [4,4],
#               [3,6],
#               [2,-1]])
x = np.load(data_path)
class_num, item_num = x.shape[0], x.shape[1]
y = np.arange(0,class_num,1).repeat(item_num, 0)

if mode == 'um':
    proto = np.mean(x,axis=1)
    plot_proto = proto
    proto_label = np.arange(0,class_num,1)
elif mode == 'nn':
    proto = x.reshape((class_num*item_num,-1))
    plot_proto = None
    proto_label = np.arange(0,class_num,1).repeat(item_num, 0)
elif mode == 'SIMPLE':
    proto = np.mean(x, axis=1)
    proto_label = np.arange(0, class_num, 1)
    proto, proto_label = SIMPLE(x.reshape((class_num*item_num,-1)),y,proto,proto_label)
    plot_proto = proto

x = x.reshape((class_num*item_num,-1))


# y = np.array([0,0,0,1,1,1,2,3,4])
knn.fit(proto,proto_label)

plot_decision_boundary(knn, x, y, plot_proto, proto_label)