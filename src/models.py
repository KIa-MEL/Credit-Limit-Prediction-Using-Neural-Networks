import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiplicativeLR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor

import matplotlib.pyplot as plt
from tqdm import *
from time import *
import math
import pandas as pd
from pycaret.regression import *


class OneLayerModel(nn.Module):
    def __init__(self, input_size, hidden_size, device):

        super(TwoLayerModel, self).__init__()
        self.dropout = nn.Dropout(0.05)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.leaky_reLu = nn.LeakyReLU()
        self.reLu = nn.ReLU()

        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.device = device
        self.tt = None
        
    def forward(self, x):
        out = self.dropout(x)

        out = self.linear1(out)
        out = self.leaky_reLu(out)
        out = self.linear2(out)
        # out = self.sigmoid(out)
        return out
    
    def fit(self, X_train, y_train, X_test, y_test, num_epochs=30000, learning_rate=.001, loss='MSE', optimizer_mode='Adam', \
            scheduler_lambda=.95, ridge_lambda=.01, show_plot=False, print_log=False):
        
        list_of_losses = []
        list_of_losses_test = []

        if loss=='MSE':
            criterion = nn.MSELoss()  
        elif loss=='BCE':
            criterion = nn.BCEWithLogitsLoss() 
        else:
            raise Exception('loss is not defined') 
        
        if optimizer_mode =='Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer_mode =='SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        else:
            raise Exception('optimizer is not defined')

        lmbda = lambda epoch: scheduler_lambda

        scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)

        loss = 0

        stt = time()

        for epoch in tqdm(range(num_epochs)):
            # Forward pass and loss
            # model.zero_grad()
            y_pred = self(X_train)
            loss = criterion(y_pred, y_train)

            # TODO Validation
            # TODO Bath
            # TODO 2 for

            # L2 regularization
            l2_reg = torch.tensor(0.)
            l2_reg = l2_reg.to(self.device)

            for param in self.parameters():
                l2_reg += torch.norm(param)
            
            loss += ridge_lambda * l2_reg

            # Backward pass and update

            loss.backward()
            optimizer.step()
            
            # zero grad 
            optimizer.zero_grad()
            

            if (epoch+1) % 1000 == 0:
                if print_log:
                    print(f"epoch: {epoch+1}, loss = {loss.item():.4f} || "
                        f"lr: {optimizer.state_dict()['param_groups'][0]['lr']:.7f}")
                list_of_losses.append(loss.item())
                list_of_losses_test.append(criterion(self(X_test), y_test).item())
            if (epoch+1) % 5000 == 0 and epoch > 5000:
                scheduler.step()
        
        if show_plot:
            epochs = [epoch * 1000 for epoch in range(1, len(list_of_losses) + 1)]

            plt.plot(epochs, list_of_losses, linestyle='-', color='b', label='Training Loss')
            plt.plot(epochs, list_of_losses_test, linestyle='-', color='r', label='Testing Loss')

            plt.title('Training Loss Over Epochs')
            plt.xlabel('Epoch (multiplied by 1000)')
            plt.ylabel('Loss')

            plt.grid(True)
            plt.legend()
            plt.show()

        ett=time()
        self.tt = ett-stt
        return None
    
    def predict(self, X_test):
        with torch.no_grad():
            y_predicted = self(X_test)            
            return y_predicted
        
    def test_report(self, X_test, y_test):
        cpu_device = torch.device('cpu')
 

        y_pred = self.predict(X_test)

        # self.to(cpu_device)
        # X_test.to(cpu_device)
        # y_test.to(cpu_device)

        y_pred = y_pred.to(cpu_device)
        y_test = y_test.to(cpu_device)

        y_pred_np = y_pred.numpy()
        y_test_np = y_test.numpy()

        

        MSE = mean_squared_error(y_pred, y_test)
        MAE = mean_absolute_error(y_pred, y_test)
        RMSE = math.sqrt(MSE)
        R2 = r2_score(y_pred, y_test)
        df = {
            'Model' : ['TwoLayerModel'],
            'MSE' : [MSE],
            'MAE' : [MAE],
            'RMSE' : [RMSE],
            'R2' : [R2],
            'TT (Sec)' : [self.tt],
        }
        
        self.to(self.device)
        return pd.DataFrame(df)

class TwoLayerModel(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(TwoLayerModel, self).__init__()
        self.dropout = nn.Dropout(0.05)
        self.linear1 = nn.Linear(input_size, hidden_size)
        
        self.linear2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.linear3 = nn.Linear(int(hidden_size/2), 1)


        self.leaky_reLu = nn.LeakyReLU()
        self.reLu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

        self.device = device
        self.tt = None
        
    def forward(self, x):
        out = self.dropout(x)
        out = self.linear1(out)
        out = self.leaky_reLu(out)
        out = self.linear2(out)
        out = self.leaky_reLu(out)
        out = self.linear3(out)
        # out = self.leaky_reLu(out)
        # out = self.linear4(out)
        # out = self.sigmoid(out)
        return out
    
    def fit(self, X_train, y_train, X_test, y_test, num_epochs=30000, learning_rate=.001, loss='MSE', optimizer_mode='Adam', \
            scheduler_lambda=.95, ridge_lambda=.01, show_plot=False, print_log=False):
        
        list_of_losses = []
        list_of_losses_test = []

        if loss=='MSE':
            criterion = nn.MSELoss()  
        elif loss=='BCE':
            criterion = nn.BCEWithLogitsLoss() 
        else:
            raise Exception('loss is not defined') 
        
        if optimizer_mode =='Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer_mode =='SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        else:
            raise Exception('optimizer is not defined')

        lmbda = lambda epoch: scheduler_lambda

        scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)

        loss = 0

        stt = time()

        for epoch in tqdm(range(num_epochs)):
            # Forward pass and loss
            # model.zero_grad()
            y_pred = self(X_train)
            loss = criterion(y_pred, y_train)

            # TODO Validation
            # TODO Bath
            # TODO 2 for

            # L2 regularization
            l2_reg = torch.tensor(0.)
            l2_reg = l2_reg.to(self.device)

            for param in self.parameters():
                l2_reg += torch.norm(param)
            
            loss += ridge_lambda * l2_reg

            # Backward pass and update

            loss.backward()
            optimizer.step()
            
            # zero grad 
            optimizer.zero_grad()
            

            if (epoch+1) % 1000 == 0:
                if print_log:
                    print(f"epoch: {epoch+1}, loss = {loss.item():.4f} || "
                        f"lr: {optimizer.state_dict()['param_groups'][0]['lr']:.7f}")
                list_of_losses.append(loss.item())
                list_of_losses_test.append(criterion(self(X_test), y_test).item())
            if (epoch+1) % 5000 == 0 and epoch > 5000:
                scheduler.step()
        
        if show_plot:
            epochs = [epoch * 1000 for epoch in range(1, len(list_of_losses) + 1)]

            plt.plot(epochs, list_of_losses, linestyle='-', color='b', label='Training Loss')
            plt.plot(epochs, list_of_losses_test, linestyle='-', color='r', label='Testing Loss')

            plt.title('Training Loss Over Epochs')
            plt.xlabel('Epoch (multiplied by 1000)')
            plt.ylabel('Loss')

            plt.grid(True)
            plt.legend()
            plt.show()

        ett=time()
        self.tt = ett-stt
        return None
    
    def predict(self, X_test):
        with torch.no_grad():
            y_predicted = self(X_test)            
            return y_predicted
        
    def test_report(self, X_test, y_test):
        cpu_device = torch.device('cpu')
 
        self = self.to(cpu_device)
        X_test_cpu = X_test.to(cpu_device)
        y_test_cpu = y_test.to(cpu_device)

        y_pred = self.predict(X_test_cpu)


        y_pred_cpu = y_pred.to(cpu_device)
        y_test_cpu = y_test.to(cpu_device)


        

        MSE = mean_squared_error(y_pred_cpu, y_test_cpu)
        MAE = mean_absolute_error(y_pred_cpu, y_test_cpu)
        RMSE = math.sqrt(MSE)
        R2 = r2_score(y_pred_cpu, y_test_cpu)
        df = {
            'Model' : ['TwoLayerModel'],
            'MSE' : [MSE],
            'MAE' : [MAE],
            'RMSE' : [RMSE],
            'R2' : [R2],
            'TT (Sec)' : [self.tt],
        }
        
        self = self.to(self.device)
        return pd.DataFrame(df)

class ClassicModels(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ClassicModels, cls).__new__(cls)
        return cls.instance
    @staticmethod
    def classical_report(X, y):
        cpu_device = torch.device("cpu")
        
        X_cpu = X.to(cpu_device)
        y_cpu = y.to(cpu_device)
        
        X_np = X_cpu.numpy()
        y_np = y_cpu.numpy()

        y_np = y_np.flatten()
        s = setup(data = X_np, target = y_np, session_id=123)
        best = compare_models()
        evaluate_model(best)
        predict_holdout = predict_model(best)
        return predict_holdout