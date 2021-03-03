import matplotlib.pyplot as plt
from icecream import ic
from nwork_datasetup import get_labeled_words
import pandas as pd
import ipdb
import tensorflow as tf
import numpy as np

X,y = get_labeled_words(100)
dim = X.shape[1]

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def initp(size, variance=1.0):
    return tf.Variable(np.double(tf.random.normal(size) * variance))

def init_weights(x_shape,layer_sizes):
    weight_shapes = [(x_shape[1],layer_sizes[0])]
    bias_shapes = [layer_sizes[0]]
    for i in range(len(layer_sizes)-1):
        weight_shapes.append((layer_sizes[i],layer_sizes[i+1]))
        bias_shapes.append((layer_sizes[i+1]))
    weight_shapes.append((layer_sizes[-1],1))
    bias_shapes.append(1)
    weights = [initp(i) for i in weight_shapes]
    ic(bias_shapes)
    ic(weight_shapes)
    biases = [initp((1,j)) for j in bias_shapes]
    return weights,biases
def ff(weights, biases,xb):
    out = xb
    for i in range(len(weights)):
        out = out @ weights[i]
        out += biases[i]
        ic(out.shape)
        out = softmax(out)
    return out

def loss_func(preds, yb):
    return tf.math.reduce_mean((preds-yb)**2)

lr = tf.constant(np.double([10E-4]))
losses = []
layer_shapes = [30,40]
x = tf.Variable(X.T)
weights,biases =  init_weights(x.shape,layer_shapes)
while(len(losses) == 0 or losses[-1] > 0.1):
    with tf.GradientTape(persistent=True) as tape:
        out = x @ weights[0]
        tape.watch(out)
        out += biases[0]
        for i in range(len(weights)-1):
            out = np.exp(out) / np.sum(np.exp(out), axis=0)
            out = out @ weights[i+1]
            out += biases[i+1]
            ic(out.shape)
        out = np.exp(out) / np.sum(np.exp(out), axis=0)
        loss = tf.math.reduce_mean((out-y)**2)
    dW = tape.gradient(loss, weights)
    dB = tape.gradient(loss, biases)
    losses.append(loss)
    ic(dW)
    for i in range(len(weights)):
        weights[i].assign_sub(dW[i])
        biases[i].assign_sub(dB[i])

    
plt.plot(list(range(len(losses))), losses)
plt.ylabel('loss (MSE)')
plt.xlabel('steps')
plt.show()