#!/usr/bin/env python
# coding: utf-8

# Python library imports

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import pandas as pd;




# Function definitions for reading data and training the model
def read_csv_convert_to_numpy(fileName='carSUV_normalized.csv'):

    filepath = Path('C:\\Users\\monem\\OneDrive\\Documents\\Homework2') / fileName #path of folder combined with file name
    df = pd.read_csv(filepath)

    # outputs: 
    numpy_x = df[["ZeroToSixty","PowerHP"]].to_numpy() # convert specified columns to numpy array
    numpy_x.shape #check shape of numpy_x

    labels = (df['IsCar'] - df['IsSUV']).to_numpy().reshape(-1,1) # +1 for car, -1 for SUV
    labels.shape #this should be (10,1) for the input from 'carSUV_normalized.csv'
    numpy_y = labels.reshape(-1,1) 
    numpy_y.shape 

    # functions to use: pandas.read_csv, .to_numpy from pandas dataframe
    print(labels.shape)
    return numpy_x, numpy_y

def calc_error_rate_for_single_vector_w(w, numpy_x, numpy_y):
    # compute scores for every sample
    scores = numpy_x @ w  # compute scores for every sample

    predictions = np.sign(scores)  # turn scores into predicted labels (+1 or -1)

    n_errors = np.sum(predictions != numpy_y) # count how many predictions don't match the true labels

    error_rate = n_errors / numpy_y.shape[0] # divide by total number of samples to get the rate

    return error_rate

def train_and_evaluate(numpy_x, numpy_y, n_epochs=20, c=0.01):
    print(numpy_x.shape)
    print(numpy_y.shape)

    w = np.random.randn(2, 1)

    for epoch in range(n_epochs): #loops 20 times   
        for i in range(numpy_y.shape[0]): #process all samples
            x_i = numpy_x[i]
            y_i = numpy_y[i]
            prediction_i = np.sign(x_i @ w)
            if prediction_i != y_i: #if prediction doesnt equal 
                w = w + c * y_i * x_i #update/nudge w toward correcting this misclassified sample

        current_error = calc_error_rate_for_single_vector_w(w, numpy_x, numpy_y)
        print(current_error)

    return w
        

# CMSC 606 only: Definition of functions for plotting errors for a grid of possible model weights
def function_error_rate_2D(w1_range, w2_range, numpy_x, numpy_y):
    #input: range of values w1 to inspect, as a python list, range of value w2 to inspect, as numpy 1D arrays; dataset (numpy_x, numpy_y) as above
    
    #output: a 2D numpy array, with rows corresponding to possible values of w1, columns corresponding to possible values of w2, 
    # for each cell containing the error rate for that specific w1,w2 weights, for the dataset (use: calc_error_rate_for_single_vector_w)
    
    error_rates_all_ws = np.ones((len(w1_range),len(w2_range))) #placeholder code putting 100% error on each w1,w2 combination; you should remove it
    # YOUR CODE HERE
    # use np.meshgrid instead of nested for loops

    # your code will be passed on to functions plot3D_function_on_grid and plot_function_on_grid
    # the output should look like in the slides

    return error_rates_all_ws


if __name__ == "__main__":
    # Below are some helper functions and code that may be useful to visualize your progress
    
    def plot_trained_w_and_dataset(numpy_x, numpy_y, w):
    
        samples_class1 = numpy_y.flatten()==1
        samples_class0 = numpy_y.flatten()==-1
        plt.scatter(numpy_x[samples_class1,0], numpy_x[samples_class1,1], c='red')
        plt.scatter(numpy_x[samples_class0,0], numpy_x[samples_class0,1], c='green')
        plt.xlabel('ZeroToSixty')
        plt.ylabel('PowerHP')
    
        if (w[1]==0): #weights are (something,0); feature x2 doesn't matter
            x2_line = np.linspace(-2, 2, 100)
            x1_line = 0*x2_line;
        else:
            x1_line = np.linspace(-2, 2, 100)
            x2_line = (-w[0] * x1_line) / w[1]
    
        # Create a blue line based on the equation
        plt.plot(x1_line, x2_line, c='blue')
        plt.show()
    
    
    def plot3D_function_on_grid(function_to_plot, numpy_x, numpy_y):
    
        # Create a meshgrid
        w1min,w1max = -2.0, 2.0
        w2min,w2max = -2.0, 2.0
        
        w1_range = np.arange(w1min,w1max, 0.01)
        w2_range = np.arange(w2min,w2max, 0.01)
        
        error_rates_values_for_W1W2 = function_to_plot(w1_range, w2_range, numpy_x, numpy_y)
        W1, W2 = np.meshgrid(w1_range, w2_range)
    
        # Create a figure and a 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Create the surface plot
        surface = ax.plot_surface(W1, W2, error_rates_values_for_W1W2, cmap='viridis', alpha=0.8, edgecolor='black')
    
        # Add labels and title
        ax.set_xlabel('w1')
        ax.set_ylabel('w2')
        ax.set_zlabel('error rate')
        ax.set_zlim(-0.5, 1.5)    
        return ax;
    
    def plot_function_on_grid(function_to_plot, numpy_x, numpy_y):
    
        # Create a meshgrid from -2.0 to +2.0
        range_ws=2.0
        w1min,w1max = -range_ws, range_ws
        w2min,w2max = -range_ws, range_ws
    
        
        w1_range = np.arange(w1min,w1max, 0.01)
        w2_range = np.arange(w2min,w2max, 0.01)
        
        error_rates_values_for_W1W2 = function_to_plot(w1_range, w2_range, numpy_x, numpy_y)
        W1, W2 = np.meshgrid(w1_range, w2_range)
    
        # Create a figure and a 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
        img = ax.imshow(error_rates_values_for_W1W2, origin='lower', cmap='coolwarm', extent=[w1min,w1max,w2min,w2max], aspect='auto')  # 'coolwarm' goes from blue (low) to red (high)
        ax.set_xlabel('w1')
        ax.set_ylabel('w2')
        ax.set_xlim(w1min,w1max)
        ax.set_ylim(w2min,w2max)
        cbar = fig.colorbar(img)  # Add a color bar to show the mapping of values to colors
        cbar.set_label('error rate')
    
        plt.show()
        
    
    # Testing read_csv_convert_to_numpy & calc_error_rate_for_single_vector_w
    # see HW2 slides for expected output
    numpy_x, numpy_y = read_csv_convert_to_numpy(fileName='carSUV_normalized.csv');
    np.random.seed(3) # to fix randomness
    random_w = np.random.randn(2,1)
    print("Random weights array shape",random_w.shape)
    print("Random weights values\n",random_w)
    error_rate_random_weights = calc_error_rate_for_single_vector_w(random_w, numpy_x, numpy_y)
    print("Error rate for random weights",error_rate_random_weights)
    
    
    # Testing train_and_evaluate; Running data reading, model training, and plotting the linear model over the dataset, using the functions defined above. 
    # see HW2 slides for expected output
    np.random.seed(8) # to eliminate randomness
    numpy_x, numpy_y = read_csv_convert_to_numpy(fileName='carSUV_normalized.csv');
    trained_w = train_and_evaluate(numpy_x, numpy_y, n_epochs = 20, c = 0.01);
    print(trained_w)
    plot_trained_w_and_dataset(numpy_x, numpy_y, trained_w);
    
    # CMSC 606 only: Error rate surface plot, over possible model weights, using functions defined above
    
    
    # Testing function_error_rate_2D
    # see HW2 slides for expected output
    numpy_x, numpy_y = read_csv_convert_to_numpy(fileName='carSUV_normalized.csv');
    range_ws=2.0
    w1min,w1max = -range_ws, range_ws
    w2min,w2max = -range_ws, range_ws
    w1_range = np.arange(w1min,w1max, 0.01)
    w2_range = np.arange(w2min,w2max, 0.01)
    
    error_rates_all_ws = function_error_rate_2D(w1_range,w2_range,numpy_x, numpy_y)
    
    print(error_rates_all_ws)
    ax = plot_function_on_grid(function_error_rate_2D, numpy_x, numpy_y);
    plt.show()
    ax = plot3D_function_on_grid(function_error_rate_2D, numpy_x, numpy_y);
    plt.show()
