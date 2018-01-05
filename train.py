#!/usr/bin/env python
# # -*- coding: utf-8 -*-


"""
This is a simple approach to Bike Sharing Demand competition based on MLP architecture.
"""

import os
import csv
import sys
import vlc
import math
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from dateutil import parser
from datetime import datetime
import matplotlib.pyplot as plt

# some defines of the package structure
LOGG_DIR = os.path.join(os.path.split(os.path.abspath(__file__))[0], "tensorboard")
DATA_DIR = os.path.join(os.path.split(os.path.abspath(__file__))[0], "bike_sharing_demand_data")
PARAMS_DIR = os.path.join(os.path.split(os.path.abspath(__file__))[0], "params")
ALARM_DIR = os.path.join(os.path.split(os.path.abspath(__file__))[0], "alarm")
if not os.path.exists(LOGG_DIR):
    os.mkdir(LOGG_DIR)
if not os.path.exists(PARAMS_DIR):
    os.mkdir(PARAMS_DIR)

# global vars usefull with removing outliers 
Y_SCALE = 1.0
Y_MEAN = 0.0

def read_data(filename, dev_fraction, load_y, remove_outliers):
    """
    Read X and Y data for training from csv file.
    Args:
        filename (str): Path to the csv file.
        train_dev_fraction (float): Fraction of train_dev set size.
        load_y (bool): Does file have labels?
        remove_outliers (bool): Should I remove outliers from the train dataset?
    Returns:
        X (np.array): All input data array.
        Y (np.array): All input labels array.
        X_train (np.array): Train input data array (determined by 1-dev_fraction).
        Y_train (np.array): Train labels array (determined by 1-dev_fraction).
        X_dev (np.array): Dev input data array (determined by dev_fraction).
        Y_dev (np.array): Dev labels array (determined by dev_fraction).
    """
    
    #######################################################################
    ############################# CONSTANTS ###############################
    #######################################################################

    global Y_SCALE
    global Y_MEAN
    
    TEMP_MIN = -10.
    TEMP_MAX = 60.
    HUMI_MIN = 0.
    HUMI_MAX = 100.
    WIND_MIN = 0.
    WIND_MAX = 100.

    #######################################################################
    ############################ READ FRAME ###############################
    #######################################################################
    
    frame = pd.read_csv(filename, sep=",")
    
    #######################################################################
    ######################## PREPARE INPUT DATA ###########################
    #######################################################################

    # Time
    start_time = time.time()

    # Input values
    input_year = np.array([(int(x.split(" ")[0].split("-")[0])-2011.5)*2 for x in frame["datetime"]], dtype=np.float).reshape(-1,1)
    input_mont = np.array([(int((x.split(" ")[0]).split("-")[1])-6.5)/5.5 for x in frame["datetime"]], dtype=np.float).reshape(-1,1)
    input_mday = np.array([(int((x.split(" ")[0]).split("-")[2])-16.)/15 for x in frame["datetime"]], dtype=np.float).reshape(-1,1)
    input_wday = np.array([(parser.parse(x.split(" ")[0]).weekday()-3.)/3 for x in frame["datetime"]], dtype=np.float).reshape(-1,1)
    input_hour = np.array([(int((x.split(" ")[1]).split(":")[0])-11.5)/11.5 for x in frame["datetime"]], dtype=np.float).reshape(-1,1)
    input_seas = ((frame["season"].values - 2.5)/1.5).reshape(-1, 1)
    input_holi = (frame["holiday"].values.reshape(-1, 1)-0.5)/0.5
    input_work = (frame["workingday"].values.reshape(-1, 1) - 0.5)/0.5
    input_weat = ((frame["weather"].values - 2.5)/1.5).reshape(-1, 1)
    input_temp = ((frame["temp"].values - (TEMP_MIN+TEMP_MAX)/2)*2/(TEMP_MAX+TEMP_MIN)).reshape(-1, 1)
    input_atem = ((frame["atemp"].values - (TEMP_MIN+TEMP_MAX)/2)*2/(TEMP_MAX+TEMP_MIN)).reshape(-1, 1)
    input_humi = ((frame["humidity"].values - (HUMI_MIN+HUMI_MAX)/2)*2/(HUMI_MAX+HUMI_MIN)).reshape(-1, 1)
    input_wind = ((frame["windspeed"].values - (WIND_MIN+WIND_MAX)/2)*2/(WIND_MAX+WIND_MIN)).reshape(-1, 1)
    
    # X and Y
    X = np.concatenate((input_year, input_mont, input_mday, input_wday, input_hour, input_seas, input_holi, input_work, input_weat, input_temp, input_atem, input_humi, input_wind), axis=1)
    #X = np.concatenate((input_year, input_mont, input_mday, input_wday, input_hour, input_seas, input_holi, input_work, input_weat, input_temp, input_atem, input_humi), axis=1)
    #X = np.concatenate((input_mont, input_wday, input_hour, input_seas, input_holi, input_work, input_weat, input_temp, input_atem, input_humi, input_wind), axis=1)
    X = X.T
    if load_y:
        # casual, registered, count
        Y = frame["count"].values.astype(float).reshape(-1, 1)
        Y = Y.T
    
        if remove_outliers:
            inlier = np.abs(Y - np.mean(Y)) <= (2*np.std(Y))
            Y_SCALE = np.mean(Y) + 2*np.std(Y)
            Y = (Y[inlier]/Y_SCALE).reshape(1,-1)
            Y_MEAN = np.mean(Y)
            X = X[:,inlier.reshape(-1)]
    
    # Dev set
    train_dev_indices = np.random.rand(X.shape[1]) < dev_fraction
    X_train = X[:,np.logical_not(train_dev_indices)]
    X_dev = X[:,train_dev_indices]
    if load_y:
        Y_train = Y[:,np.logical_not(train_dev_indices)]
        Y_dev = Y[:,train_dev_indices]

    # Time and info
    stop_time = time.time()
    print "Number of training examples = " + str(X_train.shape[1])
    print "Number of develop examples  = " + str(X_dev.shape[1])
    print "X_train shape: " + str(X_train.shape)
    print "X_dev shape:   " + str(X_dev.shape)
    if load_y:
        print "Y_train shape: " + str(Y_train.shape)
        print "Y_dev shape:   " + str(Y_dev.shape)
    print "Input data prepared in %s seconds" % (stop_time - start_time)
    
    # Return
    if load_y:
        return X, Y, X_train, Y_train, X_dev, Y_dev
    return X, None, X_train, None, X_dev, None

def initialize_parameters(n_x, n_y, layers_shapes, initialization_method):
    """
    Initializes parameters to build a neural network with tensorflow.
    Args:
        n_x (int): size of an input layer
        n_y (int): size of an output layer
        hidden_shapes (list of int): hidden layers sizes
        method (str): net's weights initialization method: xavier and hu are supported for now 
    
    Returns:
        parameters: dict of tensors with WX and bX keys
    """    
    parameters = {}
    
    # First layer
    if initialization_method == "xavier":
        parameters["W1"] = tf.get_variable("W1", [layers_shapes[0], n_x], initializer = tf.contrib.layers.xavier_initializer())
    elif initialization_method == "hu":
        parameters["W1"] = tf.Variable(tf.random_normal([layers_shapes[0], n_x])*tf.sqrt(2.0/n_x), name = "W1")
    else:
        raise ValueError("I don't know this method of net's weights initialization..")
    parameters["b1"] = tf.get_variable("b1", [layers_shapes[0], 1], initializer = tf.zeros_initializer())
    tf.summary.histogram("W1", parameters["W1"])
    tf.summary.histogram("b1", parameters["b1"])

    # Other layers
    for idx, val in enumerate(layers_shapes[1:]):
        leyers_shape_idx = idx + 1
        layers_param_idx = str(idx+2)
        if initialization_method == "xavier":
            parameters["W" + layers_param_idx] = tf.get_variable("W" + layers_param_idx, [layers_shapes[leyers_shape_idx], layers_shapes[leyers_shape_idx-1]], initializer = tf.contrib.layers.xavier_initializer())
        elif initialization_method == "hu":
            parameters["W" + layers_param_idx] = tf.Variable(tf.random_normal([layers_shapes[leyers_shape_idx], layers_shapes[leyers_shape_idx-1]])*tf.sqrt(2.0/layers_shapes[leyers_shape_idx-1]), name = "W" + layers_param_idx)
        parameters["b" + layers_param_idx] = tf.get_variable("b" + layers_param_idx, [layers_shapes[leyers_shape_idx], 1], initializer = tf.zeros_initializer())
        tf.summary.histogram("W" + layers_param_idx, parameters["W" + layers_param_idx])
        tf.summary.histogram("b" + layers_param_idx, parameters["b" + layers_param_idx])

    return parameters

def forward_propagation(X, parameters, hidden_activation, output_activation):
    """
    Implements the forward propagation for the model defined in parameters.
    
    Arguments:
        X (tf.placeholder): input dataset placeholder, of shape (input size, number of examples)
        parameters (dict): python dictionary containing your parameters "WX", "bX",
                    the shapes are given in initialize_parameters
        hidden_activation (str): Activation method of hidden layers, supported methods:
                    relu, sigmoid, tanh
        output_activation (str): Activation method of last layer, supported layers:
                    relu, sigmoid, sigmoid_my
    Returns:
        Y_hat: the output of the last layer (estimation)
    """
    global Y_MEAN
    
    AX = X
    
    # Hidden layers
    for idx in range(1, len(parameters)/2):
        with tf.name_scope("layer_" + str(idx)):
            ZX = tf.add(tf.matmul(parameters["W" + str(idx)], AX), parameters["b" + str(idx)])
            if hidden_activation == "sigmoid":
                AX = tf.nn.sigmoid(ZX, name="sigmoid" + str(idx))
            elif hidden_activation == "relu":
                AX = tf.nn.relu(ZX, name="relu" + str(idx))
            elif hidden_activation == "tanh":
                AX = 1.7159*tf.nn.tanh(2*ZX/3, name="tanh" + str(idx)) #LeCunn tanh
            else:
                raise ValueError("I don't know this activation method...")

    # Output layer
    idx = len(parameters)/2
    with tf.name_scope("layer_" + str(idx)):
        ZX = tf.add(tf.matmul(parameters["W" + str(idx)], AX), parameters["b" + str(idx)])
        if output_activation == "sigmoid":
            AX = tf.nn.sigmoid(ZX, name="sigmoid" + str(idx))
        elif output_activation == "sigmoid_my":
            bias = 1/Y_MEAN -1
            AX = tf.divide(tf.exp(5*ZX),(tf.exp(5*ZX) + bias), name="sigmoid_my" + str(idx))
            #AX = tf.nn.sigmoid(ZX, name="sigmoid_my" + str(idx))
        elif output_activation == "relu":
            AX = tf.nn.relu(ZX, name="relu" + str(idx))

    return AX

def compute_cost_rmsle_aligned(Y, Y_hat):
    """
    Computes the RMSLE cost aligned to the competitions metric.
    
    Arguments:
        Y : true labels vector placeholder, same shape as Y_hat
        Y_hat : output of forward propagation of shape (1, number of examples)
    
    Returns:
        cost : Tensor of the cost function.
    """
    global Y_SCALE
    return tf.sqrt(tf.reduce_mean(tf.squared_difference(tf.log(Y_SCALE*Y_hat+1),tf.log(Y_SCALE*Y+1))))

def compute_cost_rmsle(Y, Y_hat):
    """
    Computes the RMSLE cost.
    
    Arguments:
        Y : true labels vector placeholder, same shape as Y_hat
        Y_hat : output of forward propagation of shape (1, number of examples)
    
    Returns:
        cost : Tensor of the cost function.
    """
    return tf.sqrt(tf.reduce_mean(tf.squared_difference(tf.log(Y_hat+1),tf.log(Y+1))))

def random_mini_batches(X, Y, mini_batch_size = 64):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
        X : input data, of shape (input size, number of examples)
        Y : true "label" vector of shape (1, number of examples)
        mini_batch_size : size of the mini-batches, integer
    
    Returns:
        mini_batches : list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m/mini_batch_size))
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def bike_model_train(X_train_data, Y_train_data, X_dev_data, Y_dev_data,
                     layers_shapes, initialization_method,
                     hidden_activation, output_activation, cost_method,
                     learning_rate, num_epochs, minibatch_size,
                     tensorgraph_name, save_params):
    """
    Creates a training model for bike sharing demand.
    
    Arguments:
        X_train_data : training input data, of shape (input size, number of examples)
        Y_train_data : training labels, of shape (input size, number of examples)
        X_dev_data : dev input data, of shape (input size, number of dev_examples)
        Y_dev_data : dev labels, of shape (input size, number of dev_examples)
        initialization_method : initialization metod of the net's prameters weights,
            go @initialize_parameters for more info
        hidden_activation : activation method of the hidden units, go @forward_propagation
            for more info
        output_activation : activation method of the output unit, go @forward_propagation
            for more info
        cost_method : one of the following:
            compute_cost_rmsle
            tf.losses.mean_squared_error
        learning_rate : learning rate of the optimizer
        num_epochs : number of epochs to learn
        minibatch_size : size of the mini-batches
        tensorgraph_name : tensorgraph namespace
        save_params : should I save calculated params?
    """

    # Reset
    tf.reset_default_graph()

    # Params
    n_x, m = X_train_data.shape
    n_y = Y_train_data.shape[0]
    
    # Create placeholders
    X = tf.placeholder(tf.float32, shape=[n_x, None], name="X")
    Y = tf.placeholder(tf.float32, shape=[n_y, None], name="Y")
    
    # Init parameters
    parameters = initialize_parameters(X_train_data.shape[0], Y_train_data.shape[0], layers_shapes, initialization_method)

    # Build forward propagation
    Y_hat = forward_propagation(X, parameters, hidden_activation, output_activation)
    tf.summary.histogram("Y_hat", Y_hat)

    # Backpropagation: AdamOptimizer.
    with tf.name_scope("train_batch"):

        cost = cost_method(Y, Y_hat)
        tf.summary.scalar("cost_rmsle", cost)
        
        # learning rate decay
        #step = tf.Variable(0, trainable=False)
        #num_minibatches = int(m / minibatch_size)
        #rate = tf.train.exponential_decay(learning_rate, step, num_minibatches * 100, 0.96)
        #tf.summary.scalar("learning_rate", rate)
        
        # optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)#, global_step=step)

    # merge all summaries
    summary_train_batch = tf.summary.merge_all()

    # define cost
    with tf.name_scope("info"):
        cost_rmsle_train = compute_cost_rmsle_aligned(Y, Y_hat)
        summary_train = tf.summary.scalar("cost_rmsle_train", cost_rmsle_train)
        cost_rmsle_dev = compute_cost_rmsle_aligned(Y, Y_hat)
        summary_dev = tf.summary.scalar("cost_rmsle_dev", cost_rmsle_dev)
        summary_info = tf.summary.merge([summary_train, summary_dev])
    
    # Initialize all the variables
    init = tf.global_variables_initializer()
    
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        writer = tf.summary.FileWriter(os.path.join(LOGG_DIR, tensorgraph_name))
        writer.add_graph(sess.graph)

         # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            minibatches = random_mini_batches(X_train_data, Y_train_data, minibatch_size) 

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                sess.run(optimizer, feed_dict={X: minibatch_X, Y: minibatch_Y})

            # Print the cost every epoch
            if epoch % 5 == 0:
                # Train batch
                summary_train_batch_act = sess.run(summary_train_batch, feed_dict={X: minibatch_X, Y: minibatch_Y})
                writer.add_summary(summary_train_batch_act, epoch)
                # Dev
                summary_train_act = sess.run(summary_train, feed_dict={X: X_train_data, Y: Y_train_data})
                writer.add_summary(summary_train_act, epoch)
                # Dev
                summary_dev_act = sess.run(summary_dev, feed_dict={X: X_dev_data, Y: Y_dev_data})
                writer.add_summary(summary_dev_act, epoch)
        
        # Accuracy on train        
        accuracy_rmsle = sess.run(cost_rmsle_train, feed_dict={X: X_train_data, Y: Y_train_data})
        print "RMSLE ACCURACY TRAIN: ", accuracy_rmsle
        
        accuracy_rmsle_test = sess.run(cost_rmsle_train, feed_dict={X: X_dev_data, Y: Y_dev_data})
        print "RMSLE ACCURACY DEV:   ", accuracy_rmsle_test
        
        if save_params:
            params = sess.run(parameters)
            np.save(os.path.join(PARAMS_DIR, tensorgraph_name + '.npy'), params)

def bike_model_predict(X_test, filename, layers_shapes, hidden_activation, output_activation):
    """
    Predict the bike demand with model read form file
    
    Arguments:
        X_test : test input data, of shape (input size, number of test_examples)
        filename : Name of a file, from where the parameters will be loaded
        layers_shapes : Shape of the network.
        hidden_activation : activation method of the hidden units, go @forward_propagation
            for more info
        output_activation : activation method of the output unit, go @forward_propagation
            for more info
    """
    parameters = {}
    read_dictionary = np.load(filename).item()
    
    # First layer
    for idx in range(len(layers_shapes)):
        idx_str = str(idx+1)
        parameters["W" + idx_str] = tf.Variable(read_dictionary["W" + idx_str], name = "W" + idx_str)
        parameters["b" + idx_str] = tf.Variable(read_dictionary["b" + idx_str], name = "b" + idx_str)

    # Params
    n_x, m = X_test.shape
    n_y = 1
    
    # Create placeholders
    X = tf.placeholder(tf.float32, shape=[n_x, None], name="X")
    
    # Build forward propagation
    Y_hat = forward_propagation(X, parameters, hidden_activation, output_activation)
    tf.summary.histogram("Y_hat", Y_hat)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()
    
    # Predict
    with tf.Session() as sess:
        sess.run(init)
        output = sess.run(Y_hat, feed_dict={X: X_test})
        frame_in = pd.read_csv("test.csv", sep=",")
        frame_out = frame_in[["datetime"]].copy()
        frame_out["count"] = Y_SCALE*output.T
        frame_out.to_csv("submission.csv", sep=",", index=False)
        
def main(argv):

    # Load data
    train_filepath = os.path.join(DATA_DIR, "train.csv")
    X_train_all, Y_train_all, X_train, Y_train, X_dev, Y_dev = read_data(train_filepath, dev_fraction=0.025, load_y=True, remove_outliers=False)

    # Params of the NN
    initialization_method = "hu"
    hidden_activation="relu"
    output_activation="relu"
    num_epochs = 1000
    cost_method = compute_cost_rmsle
    
    layers_shapes = [X_train.shape[0]]*5 + [X_train.shape[0]/2, Y_train.shape[0]]
    #layers_shapes = list(reversed(range(1, X_train.shape[0] + 1)))
    #layers_shapes = [int(math.floor(x/2)) for x in list(reversed(range(3, (X_train.shape[0]+1)*2, 1)))]    
    
    for trial in range(1,4):
        for learning_rate in [10 ** (-i) for i in range(2, 6)]:
        #for learning_rate in [10 ** (-2-i/3) for i in range(1, 7)]:
        #for learning_rate in [10**-3]:
            #for minibatch_size in [32, 64, 128, 256]:
            for minibatch_size in [64]:
                #tensorgraph_name = ("rmsle_{0:.6f}".format(learning_rate)) + str(datetime.now()) 
                #tensorgraph_name = "rmle_cost_" + str(datetime.now())
                tensorgraph_name = ("rmsle_{0:.6f}".format(learning_rate)) + "_" + str(minibatch_size) + "_" + str(trial)
                      
                bike_model_train(X_train, Y_train, X_dev, Y_dev, 
                                 layers_shapes=layers_shapes, initialization_method = initialization_method,
                                 learning_rate=learning_rate, num_epochs=num_epochs, minibatch_size=minibatch_size,
                                 hidden_activation=hidden_activation, output_activation=output_activation, cost_method = cost_method,
                                 tensorgraph_name=tensorgraph_name, save_params=True)


#     X_test_all, _, _, _, _, _ = read_data("test.csv", dev_fraction=0., load_y=False, remove_outliers=False)
#     bike_model_predict(X_test_all, "params_last_hope.npy", layers_shapes,
#                        hidden_activation, output_activation)

    # DONE!
    alarm_file = os.path.join(ALARM_DIR, "Rick Astley - Never Gonna Give You Up.mp3")
    p = vlc.MediaPlayer(alarm_file)
    p.play()
    while p.get_state() != vlc.State.Ended:
        time.sleep(1)
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
