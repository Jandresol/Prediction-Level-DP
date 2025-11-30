import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from torch.utils.data import DataLoader, TensorDataset


class PyTorchCNNWrapper(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper for PyTorch CNN models."""
    
    def __init__(self, model_class=None, epochs=5, batch_size=32, lr=1e-3, device='auto'):
        self.model_class = model_class
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        
    def _setup_device(self):
        if self.device == 'auto':
            self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device_ = torch.device(self.device)
    
    def _prepare_data(self, X, y=None):
        # Handle different input shapes
        if len(X.shape) == 2:
            # Flattened features, assume CIFAR-10 shape
            X = X.reshape(-1, 3, 32, 32)
        
        X_tensor = torch.FloatTensor(X)
        
        if y is not None:
            y_tensor = torch.FloatTensor(y).unsqueeze(1)
            return X_tensor, y_tensor
        return X_tensor
    
    def fit(self, X, y):
        if X.ndim == 4:
            X = X.reshape(X.shape[0], -1)
        X, y = check_X_y(X, y)
        self._setup_device()
        self.classes_ = unique_labels(y)
        
        if self.model_class is None:
            from src.models.cifar_cnn import cifar_cnn
            self.model_ = cifar_cnn().to(self.device_)
        else:
            self.model_ = self.model_class().to(self.device_)
        
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        
        X_tensor, y_tensor = self._prepare_data(X, y)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model_.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device_), batch_y.to(self.device_)
                
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        if X.ndim == 4:
            X = X.reshape(X.shape[0], -1)
        X = check_array(X)
        
        self.model_.eval()
        X_tensor = self._prepare_data(X)
        X_tensor = X_tensor.to(self.device_)
        
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            predictions = (outputs > 0.5).float().cpu().numpy().flatten()
        
        return predictions.astype(int)
