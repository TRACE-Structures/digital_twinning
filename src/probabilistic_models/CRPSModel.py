from .ProbabilisticModel import ProbabilisticModel
import torch
import torch.nn as nn
from torchnaut import crps
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
from copy import deepcopy


class CRPSModel(ProbabilisticModel):

    def __init__(self, Q, QoI_names, layers, n_epsilon=30):
        self.Q = Q
        self.QoI_names = QoI_names
        self.layers = layers
        self.model = self.CRPSInner(input_size=len(self.QoI_names), output_size=self.Q.num_variables(), layers=layers, n_epsilon=n_epsilon)
        self.n_epsilon = n_epsilon

    def inference(self, features, n_samples=2000):
        Xq = torch.tensor(self.q_scaler.transform(features), dtype=torch.float32)
        samples_scaled = self.model.predict(Xq, n_samples=n_samples)
        return self.y_scaler.inverse_transform(samples_scaled[0].numpy())

    def train(self, q, y, k_fold=None, **params):
        q_train, q_val, y_train, y_val = self.train_test_split(q, y, train_test_ratio=0.8, random_seed=1997, split_type='no_shuffle')

        self.train_config = params

        self.q_scaler = StandardScaler().fit(q_train)
        self.y_scaler = StandardScaler().fit(y_train)

        q_train_tensor = torch.tensor(self.q_scaler.transform(q_train), dtype=torch.float32)
        q_val_tensor = torch.tensor(self.q_scaler.transform(q_val), dtype=torch.float32)

        y_train_tensor = torch.tensor(self.y_scaler.transform(y_train), dtype=torch.float32)
        y_val_tensor = torch.tensor(self.y_scaler.transform(y_val), dtype=torch.float32)

        print(f'----- Training started for \'MDNModel\' model -----')
        
        if k_fold is not None:
            train_losses, val_losses = self.cross_validation(y_train_tensor, q_train_tensor, k_fold, **params)
        else:
            train_losses, val_losses = self.model.train_and_validate(q_train_tensor, y_train_tensor, q_val_tensor, y_val_tensor, **params)
        # print(f"Average train loss: {np.mean(train_losses):.14f}, Average valid loss: {np.mean(val_losses):.14f}")      
        print(f'----- Training ended for \'MDNModel\' model -----')

    class CRPSInner(nn.Module):
        def __init__(self, input_size, output_size, layers, n_epsilon):
            super().__init__()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            default_neurons = 32
            default_activation = 'relu'
            default_dropout = 0.0

            # Define a mapping from string names to PyTorch activation functions
            activation_mapping = {
                'gelu': nn.GELU(),
                'relu': nn.ReLU(),
                'sigmoid': nn.Sigmoid(),
                'tanh': nn.Tanh(),
                'leakyrelu': nn.LeakyReLU(),
                'softmax': nn.Softmax(dim=1),
                'none': nn.Identity()
            }

            hidden_layers = []
            prev_size = input_size

            # Build hidden layers
            for i, layer in enumerate(layers):
                neurons = default_neurons if layer['neurons'] is None else layer['neurons']
                activation = default_activation if layer['activation'] is None else layer['activation']
                dropout = default_dropout if layer['dropout'] is None else layer['dropout']
                # Add Linear, Activation, and Dropout layers to the list
                hidden_layers.append(nn.Linear(prev_size, neurons))
                hidden_layers.append(nn.LayerNorm(neurons))
                hidden_layers.append(activation_mapping.get(activation.lower()))
                if dropout > 0:
                    hidden_layers.append(nn.Dropout(dropout))
                prev_size = neurons

                if i == 0:
                    hidden_layers.append(crps.EpsilonSampler(n_epsilon))
                    prev_size = neurons + n_epsilon


            # Create Sequential model
            self.hidden_layers = nn.Sequential(*hidden_layers)
            self.output_layer = nn.Sequential(nn.Linear(prev_size, output_size))
            
            self.all_layers = nn.Sequential(*hidden_layers, nn.Linear(prev_size, output_size))
            
            self.to(self.device)

        def forward(self, x, n_samples=100):
            with crps.EpsilonSampler.n_samples(n_samples):
                return self.all_layers(x)

        def loss_fn(self, xb, yb):
            outputs = self(xb, n_samples=100)
            return crps.crps_loss_mv(outputs, yb).mean()
        
        def get_batch_ixs(self, like_tensor, batch_size=16, permute=True):
            if like_tensor.shape[0] <= batch_size:
                return torch.arange(like_tensor.shape[0]).unsqueeze(0)
            ixs = torch.randperm(like_tensor.shape[0]) if permute else torch.arange(like_tensor.shape[0])
            return torch.tensor_split(ixs, ixs.shape[0] // batch_size)

        # def predict(self, X, n_samples=2000):
        #     with torch.no_grad():
        #         preds = self(X)
        #     raw_samples = self.mdn_mv_util.sample(preds, n=n_samples)
        #     samples_scaled = raw_samples.transpose(0, 1) if raw_samples.shape[0] == n_samples else raw_samples
        #     return samples_scaled

        def predict(self, X, n_samples=2000):
            with torch.no_grad():
                samples_scaled = self(X, n_samples=n_samples)   # [1, n_samples, output_dim]
            return samples_scaled

        def train_and_validate(self, X_train, y_train, X_val, y_val, optimizer='adam', max_epoch=5, batch_size=64, lr=1e-3, weight_decay=1e-6, max_patience=2, verbose=False):
            self.batch_size = batch_size
            
            X_train, y_train = X_train.to(self.device), y_train.to(self.device)
            X_val, y_val = X_val.to(self.device), y_val.to(self.device)            

            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=10)

            best_val, best_state, bad_epochs = float("inf"), deepcopy(self.state_dict()), 0
            history = {"epoch": [], "train_loss": [], "val_loss": []}

            for epoch in range(max_epoch):
                self.train()
                losses = []
                for ixs in self.get_batch_ixs(X_train, batch_size, permute=True):
                    optimizer.zero_grad()
                    # outputs = self(X_train[ixs])
                    loss = self.loss_fn(X_train[ixs], y_train[ixs])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10)
                    optimizer.step()
                    losses.append(loss.item())
                scheduler.step()

                self.eval()
                with torch.no_grad():
                    val_loss = self.loss_fn(X_val, y_val).item()

                history["epoch"].append(epoch)
                history["train_loss"].append(float(np.mean(losses)))
                history["val_loss"].append(val_loss)

                if verbose and epoch % 20 == 0:
                    print(f"epoch {epoch:4d} | train {np.mean(losses):.4f} | val {val_loss:.4f}")

                if val_loss < best_val - 1e-6:
                    best_val, best_state, bad_epochs = val_loss, deepcopy(self.state_dict()), 0
                else:
                    bad_epochs += 1
                if bad_epochs > max_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} (best val loss {best_val:.4f})")
                    break

            self.load_state_dict(best_state)
            self.eval()
            return self, history