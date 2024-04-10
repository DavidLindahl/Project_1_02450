from data_loader import st_X_class, y_class
import numpy as np
import sklearn.linear_model as lm
from sklearn import model_selection
from scipy.stats import mode
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train_test_lr(model, train_data, test_data):
    # train model
    model.fit(train_data[0], train_data[1].reshape(-1))

    predictions = model.predict(test_data[0])
    return np.sum(predictions.reshape(-1)!=test_data[1].reshape(-1))/len(test_data[1])

def train_ann(hidden_nodes, data):
    # ANN parameters
    batch_size = 8
    learning_rate = 0.01
    epochs = 16
    M = 6 # inputs for the ANN

    # create model
    model = torch.nn.Sequential(
            torch.nn.Linear(M, hidden_nodes),  # M features to H hiden units
            torch.nn.ReLU(),  # torch.nn.ReLU(),
            torch.nn.Linear(hidden_nodes, 1),  # H hidden units to 1 output neuron
            torch.nn.Sigmoid(),  # final tranfer function
            )
            
    
    # convert data to tensers for ann
    X_train_inner_tensor = torch.from_numpy(data[0]).float()
    y_train_inner_tensor = torch.from_numpy(data [1]).float()
    
    # DataLoader for the training set
    train_dataset = TensorDataset(X_train_inner_tensor, y_train_inner_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return model

def test_ann(model, data):
    X_test_inner_tensor = torch.from_numpy(data[0]).float()
    # y_test_inner_tensor = torch.from_numpy(data[1]).float()
    
    # Set model to evaluation mode
    model_ann.eval()

    # Disable gradient computation for inference
    with torch.no_grad():
        predictions = model(X_test_inner_tensor)

    # Convert to NumPy array if needed (for further processing or evaluation)
    predictions_np = np.rint(np.array(predictions)).reshape(-1)
    # return np.sum(np.square(predictions_np - y_test_inner_tensor))
    return np.sum(predictions_np.reshape(-1) != data[1].reshape(-1))/len(data[1].reshape(-1))

def get_baseline(train_data, test_data):
    avg = mode(train_data.reshape(-1))[0]
    predictions = np.zeros(len(test_data.reshape(-1)))
    predictions.fill(avg)
    return np.sum(predictions.reshape(-1)!=test_data.reshape(-1))/len(test_data.reshape(-1))

# initizialize the table
table ="Outer fold,h,mse,alpha,mse,baseline\n"

# Parameters
A = 50  # Number of hyper parameters to test
outer_k = 10  # Outer K-fold
inner_k = 10  # Inner K-fold for hyperparameter tuning

# Alpha values
alphas = np.linspace(0.001, 3, A) # aplha values used for regularization of linear model

# Store scores for each outer fold
weighted_mse_estimate_ann = np.zeros(A)
weighted_mse_estimate_lr = np.zeros(A)
weighted_estimate_baseline = np.zeros(A)

outer_error_rate_lr = np.zeros(outer_k)
outer_error_rate_ann = np.zeros(outer_k)
outer_mse_baseline = np.zeros(outer_k)

# Inner Cross Validation for ANN
inner_error_rate_ann = np.zeros((A, inner_k))

# Inner Cross Validation for linear regression
inner_error_rate_lr = np.zeros((A, inner_k))

# length of test sets
length_of_inner_test_set = np.zeros(inner_k)
length_of_outer_test_set = np.zeros(outer_k)



# Prepare cross-validation (outer loop)
outer_cv = model_selection.KFold(n_splits=outer_k, shuffle=False)

# Outer Cross Validation
for outer_fold, (train_index, test_index) in enumerate(outer_cv.split(st_X_class, y_class)):
    print(f"running outer fold {outer_fold+1}/{outer_k}...")
    # Split data
    X_train_outer, X_test_outer = st_X_class[train_index], st_X_class[test_index]
    y_train_outer, y_test_outer = y_class[train_index], y_class[test_index]
    
    length_of_outer_test_set[outer_fold] = len(y_test_outer)
    
    inner_cv = model_selection.KFold(n_splits=inner_k, shuffle=False)
    
    for j, (train_index_inner, test_index_inner) in enumerate(inner_cv.split(X_train_outer, y_train_outer)):
        # create inner training and testing data
        X_train_inner, y_train_inner = X_train_outer[train_index_inner], y_train_outer[train_index_inner]
        X_test_inner, y_test_inner = X_train_outer[test_index_inner], y_train_outer[test_index_inner]
        length_of_inner_test_set[j] = len(y_test_inner)
        for i, alpha in enumerate(alphas):
            
            model_lr = lm.LogisticRegression(penalty='l2', C=alpha, solver='lbfgs', max_iter=10000)
            
            # train ANN
            model_ann = train_ann(hidden_nodes=(i*2)+1, data=(X_train_inner, y_train_inner))
            
            # test ANN
            inner_error_rate_ann[i, j] = test_ann(model=model_ann, data=(X_test_inner, y_test_inner))
        
            # train and test lr
            inner_error_rate_lr[i, j] = train_test_lr(model=model_lr, train_data=(X_train_inner, y_train_inner), test_data=(X_test_inner, y_test_inner))
        
    for c1 in range(A):
        weighted_mse_estimate_ann[c1] = np.sum([(length_of_inner_test_set[c2] * inner_error_rate_ann[c1, c2])/len(y_test_outer) for c2 in range(inner_k)])
        weighted_mse_estimate_lr[c1] = np.sum([(length_of_inner_test_set[c2] * inner_error_rate_lr[c1, c2])/len(y_test_outer) for c2 in range(inner_k)])
    
    # get the best aplha value and the best number of nodes in the network
    best_nodes_number = (np.argmin(weighted_mse_estimate_ann)*2) + 1
    best_alpha = alphas[np.argmin(weighted_mse_estimate_lr)]
    
    # create models based on best aplha value and the best number of nodes
    model_lr = lm.LogisticRegression(penalty='l2', C=best_alpha, solver='lbfgs', max_iter=10000)
    
    # train and test models
    model_ann = train_ann(hidden_nodes=(i*2)+1, data=(X_train_outer, y_train_outer))
    outer_error_rate_ann[outer_fold] = test_ann(model=model_ann, data=(X_test_outer, y_test_outer))
    
    outer_error_rate_lr[outer_fold] = train_test_lr(model=model_lr, train_data=(X_train_outer, y_train_outer), test_data=(X_test_outer, y_test_outer))
    baseline = get_baseline(train_data=y_train_outer, test_data=y_test_outer)
    table += f"{outer_fold + 1},{best_nodes_number},{round(outer_error_rate_ann[outer_fold], 2)},{round(best_alpha, 2)},{round(outer_error_rate_lr[outer_fold],2)},{round(baseline, 2)}\n"

print(table)
