import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def cost_func_for_ridge_regression(X, Y, theta, Lambda):
    m = X.shape[0] # size of dataset
    H = np.dot(X, theta) # hypothesis
    loss = H - Y
    no_of_parameters = len(theta)
    I =  np.eye(no_of_parameters, dtype=int)
    I[0][0] = 0
    regularization = (Lambda/(2*m)) * np.dot(np.dot(theta.T, I), theta)
    cost = 1/(2*m) * np.dot(loss.T, loss) + regularization
    return np.squeeze(cost)

def gradient_decent_of_cost_function(X, Y, theta, Lambda):
    m = X.shape[0]
    H = np.dot(X, theta)
    loss = H - Y
    
    no_of_parameters = len(theta)
    I =  np.eye(no_of_parameters, dtype=int)
    I[0][0] = 0
    regularization = Lambda * np.dot(I, theta)
    
    theta = 1/m * np.dot(X.T, loss) + regularization
    return theta

def gradient_decent_of_ridge_regression(X, Y, theta, alpha, Lambda, cost_diff_threshold):
    i = 0
    cost_diff = cost_diff_threshold + 1
    thetas = [theta]
    costs = [cost_func_for_ridge_regression(X=X, Y=Y, theta=theta, Lambda=Lambda)]
    while cost_diff > cost_diff_threshold:
        d_theta = gradient_decent_of_cost_function(X=X, Y=Y, theta=theta, Lambda=Lambda)
        theta = theta - (alpha * d_theta)
        thetas.append(theta)
        costs.append(cost_func_for_ridge_regression(X=X, Y=Y))
        cost_diff = costs[i+1] - costs[i]
        if cost_diff > 0:
            print(f"Diverging for theta : {theta}")
            break
        i += 1
    return thetas, costs

def normal_eqn_ridge_eqn(X, Y, Lambda):
    XT_X = np.dot(X.T, X)
    XT_Y = np.dot(X.T, Y)

    no_of_parameters = len(X[0])
    I =  np.eye(no_of_parameters, dtype=int)
    I[0][0] = 0
    inverse = np.linalg.pinv(XT_X, Lambda*I)
    
    result = np.dot(inverse, XT_Y)
    return result   

def MSE(predicted_Y, actual_Y):
    difference = actual_Y - predicted_Y
    m = len(predicted_Y)
    mse = (1/m)*np.dot(difference.T, difference)
    return np.squeeze(mse)

def get_initial_theta(num_parameters):
    theta = np.random.rand(num_parameters,1)
    return theta

def apply_gradient_decent(train_X, train_Y, Lambda):
    initial_theta = get_initial_theta(train_X.shape[1])
    cost_diff_threshold = 1e-6
    learning_rate = 0.1
    thetas, costs = gradient_decent_of_ridge_regression(
        X=train_X,
        Y=train_Y,
        theta=initial_theta,
        alpha=learning_rate,
        Lambda=Lambda,
        cost_diff_threshold=cost_diff_threshold
    )
    theta_GD = thetas[-1]
    return theta_GD, costs

def evaluate(data_X, data_Y, theta):
    predicted_Y = np.dot(data_X.T, theta)
    mse = MSE(predicted_Y=predicted_Y, actual_Y=data_Y)
    return mse, predicted_Y

def ridge_regression(train_X, train_Y, Lambda, split_data=True, use_normal_eqn=True):
    train_X = np.insert(train_X, 0, 1, axis=1)
    train_Y = train_Y.values.reshape((-1, 1))

    if use_normal_eqn:
        theta = normal_eqn_ridge_eqn(X=train_X, Y=train_Y, Lambda=Lambda)
    else:
        theta, costs = apply_gradient_decent(train_X=train_X, train_Y=train_Y, Lambda=Lambda)
    
    return theta