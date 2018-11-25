#!/usr/bin/env python3.7
# no imports beyond numpy should be used in answering this question
import numpy as np

# train datapoints: 2 features and binary output
train_separable = np.array([[2.7810836,2.550537003,0],
    [1.465489372,2.362125076,0],
    [3.396561688,4.400293529,0],
    [1.38807019,1.850220317,0],
    [3.06407232,3.005305973,0],
    [7.627531214,2.759262235,1],
    [5.332441248,2.088626775,1],
    [6.922596716,1.77106367,1],
    [8.675418651,-0.242068655,1],
    [7.673756466,3.508563011,1]])

# train datapoints nonseparable: 2 features and binary output
train_nonseparable = np.array([[2.7810836,2.550537003,0],
    [1.465489372,2.362125076,0],
    [3.396561688,4.400293529,1],
    [1.38807019,1.850220317,0],
    [3.06407232,3.005305973,0],
    [7.627531214,2.759262235,1],
    [5.332441248,2.088626775,1],
    [6.922596716,1.77106367,0],
    [8.675418651,-0.242068655,1],
    [7.673756466,3.508563011,1]])

# test datapoints: 2 features and binary output
test = np.array([[1.927628496,7.200103829,0],
    [9.182983992,5.290012983,1]])

#number of epochs
n_epoch = 100

# make a prediction with weights
def predict(row, weights):
    activation = 0.0
    # Do not edit any code in this function outside the edit region
    # Edit region starts here
    #########################
    # Your code goes here
    activation = np.dot(weights[1:], row) + weights[0]
    #########################
    # Edit region ends here
    
    return 1.0 if activation >= 0.0 else 0.0

# estimate perceptron weights using stochastic gradient descent
def train_weights(train_x, train_y, n_epoch):
    weights = np.zeros(train_x.shape[1]+1)
    errorVepoch = []
    for epoch in range(n_epoch):
        sum_error = 0.0
        for i in range(train_x.shape[0]):
            row = train_x[i,:]
            prediction = predict(row, weights)
            error = train_y[i] - prediction
            sum_error += error**2

            # Update the bias term "weights[0]" and the feature-specific weights "weights[1:]" in the code block below
            # Do not edit any code in this function outside the edit region
            # Edit region starts here
            #########################
            # Your code goes here
            if abs(error) > 1e-6:
                weights += error*np.concatenate((np.array([1]), row))
            #########################
            # Edit region ends here
        
        print('>epoch=%d, error=%.3f' % (epoch, sum_error))
        errorVepoch.append([epoch, sum_error])
    return weights, errorVepoch

if __name__ == '__main__':
    # split train data into features and predictions
    train_separable_x = train_separable[:, :-1]
    train_separable_y = train_separable[:, -1]
    
    # split non-separable train data into features and predictions
    train_nonseparable_x = train_nonseparable[:,:-1]
    train_nonseparable_y = train_nonseparable[:,-1]
    
    # split test data into features and predictions
    test_x = test[:,:-1]
    test_y = test[:,-1]

    # train weights using training data
    weights, errorVepoch_sep = train_weights(train_separable_x, train_separable_y, n_epoch)

    # make predictions on test data and compare with groundtruth
    for i in range(test.shape[0]):
        row = test_x[i,:]
        prediction = predict(row, weights)
        print("Expected=%d, Predicted=%d" % (test_y[i], prediction))
    
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    plt.figure(1)
    plt.plot(train_separable[:, 0], train_separable[:, 1], "g*")
    plt.plot(test[:, 0], test[:, 1], "r*")
    
    
    sepa_line_y = np.dot(weights[1:], row) + weights[0]
    x1= np.array([1, 10])
    x2 = (-weights[0] - weights[1]*x1)/weights[2]
    plt.plot(x1, x2)
    plt.show()
    
    # train weights using non-separable training data
    weights, errorVepoch_nonsep = train_weights(train_nonseparable_x, train_nonseparable_y, n_epoch)

    # make predictions on test data and compare with groundtruth
    for i in range(test.shape[0]):
        row = test_x[i,:]
        prediction = predict(row, weights)
        print("Expected=%d, Predicted=%d" % (test_y[i], prediction))
    plt.figure(2)
    plt.plot(train_nonseparable[:, 0], train_nonseparable[:, 1], "g*")
    plt.plot(test[:, 0], test[:, 1], "r*")
    
    
    sepa_line_y = np.dot(weights[1:], row) + weights[0]
    x1= np.array([1, 10])
    x2 = (-weights[0] - weights[1]*x1)/weights[2]
    plt.plot(x1, x2)
    plt.show()
    
    err_sep = np.array(errorVepoch_sep)
    err_nonsep = np.array(errorVepoch_nonsep)
    plt.figure(3)
    plt.plot(err_sep[:,0], err_sep[:, 1], 'g')
    plt.plot(err_nonsep[:,0], err_nonsep[:, 1], 'r')
    plt.show()
    