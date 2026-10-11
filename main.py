import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def softmax(x):
    x -= np.max(x)
    np.exp(x, out=x)
    return x / np.sum(x)

def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

rng = np.random.default_rng()

lr = 0.01
epoch = 2
layout = np.array([28*28, 16, 16, 10])

train_data = pd.read_csv("mnist_train.csv", header=None)
labels     = train_data.iloc[:, 0].to_numpy()
data       = train_data.iloc[:, 1:]
pixels     = np.empty((len(labels), data.shape[1])) # normalized pixels

for i in range(len(labels)):
    pixels[i, :] = data.iloc[i, :].to_numpy() / 255

l0_l1_w = np.random.rand(layout[1], layout[0]) # 16x784
l1_b    = np.random.rand(layout[1])            # layer 1 biases
l1_l2_w = np.random.rand(layout[2], layout[1]) # 16x16
l2_b    = np.random.rand(layout[2])            # layer 2 biases
l2_l3_w = np.random.rand(layout[3], layout[2]) # 10x16
l3_b    = np.random.rand(layout[3])            # layer 3 biases

gradient = np.empty(l0_l1_w.size + l1_l2_w.size + l2_l3_w.size + len(l1_b) + len(l2_b) + len(l3_b))

a0_input  = np.empty(layout[0])
a1        = np.empty(layout[1])
a2        = np.empty(layout[2])
a3_output = np.empty(layout[3])

true_output = np.zeros(10)
for e in range(epoch):
    print(f"epoch: {e}/{epoch}")
    gradient.fill(0)
    for j in range(len(gradient)):
        for i, l in enumerate(labels):
            true_output.fill(0)
            true_output[l] = 1
            a0_input[:]  = pixels[i, :]
            a1[:]        = sigmoid(l0_l1_w @ a0_input + l1_b)
            a2[:]        = sigmoid(l1_l2_w @ a1 + l2_b)
            a3_output[:] = sigmoid(l2_l3_w @ a2 + l3_b)
            a3_output[:] = softmax(a3_output)
            # squared_err = np.square(true_output - a3_output)
            # gradient[i] += backpropagation(a3_output, true_output)
            gradient[j] += 2 * (rng.random() - 0.5)
    gradient *= 1 / len(labels)
    l0_l1_w_ravel = l0_l1_w.ravel()
    l0_l1_w_ravel[:] -= lr * gradient[0:l0_l1_w.size]

    l1_l2_w_ravel = l1_l2_w.ravel()
    l1_l2_w_ravel[:] -= lr * gradient[l0_l1_w.size:l0_l1_w.size + l1_l2_w.size]

    l2_l3_w_ravel = l2_l3_w.ravel()
    l2_l3_w_ravel[:] -= lr * gradient[l0_l1_w.size + l1_l2_w.size:l0_l1_w.size + l1_l2_w.size + l2_l3_w.size]

    l1_b[:] -= lr * gradient[l0_l1_w.size + l1_l2_w.size + l2_l3_w.size:l0_l1_w.size + l1_l2_w.size + l2_l3_w.size + len(l1_b)]
    l2_b[:] -= lr * gradient[l0_l1_w.size + l1_l2_w.size + l2_l3_w.size + len(l1_b):l0_l1_w.size + l1_l2_w.size + l2_l3_w.size + len(l1_b) + len(l2_b)]
    l3_b[:] -= lr * gradient[l0_l1_w.size + l1_l2_w.size + l2_l3_w.size + len(l1_b) + len(l2_b):]

test_data   = pd.read_csv("mnist_test.csv", header=None)
test_labels = test_data.iloc[:, 0].to_numpy()
tdata       = test_data.iloc[:, 1:]
test_pixels = np.empty((len(test_labels), tdata.shape[1])) # normalized pixels
n_testlabels = len(test_labels)
# Test ===========================================================================
correct_pred = 0
for i, l in enumerate(test_labels):
    true_output = np.zeros(10)
    true_output[l] = 1
    a0_input[:]  = test_pixels[i, :]
    a1[:]        = sigmoid(l0_l1_w @ a0_input + l1_b)
    a2[:]        = sigmoid(l1_l2_w @ a1 + l2_b)
    a3_output[:] = sigmoid(l2_l3_w @ a2 + l3_b)
    a3_output[:] = softmax(a3_output)
    correct_pred += np.argmax(a3_output) == np.argmax(true_output)
    print(f"{correct_pred} / {n_testlabels} : [% {100 * (correct_pred / n_testlabels)}]")
