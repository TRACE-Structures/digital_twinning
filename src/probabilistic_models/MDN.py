from .ProbabilisticModel import ProbabilisticModel
import torch
import torch.nn as nn
from torchnaut.mdn import MDNMV
from sklearn.preprocessing import StandardScaler
import numpy as np
from copy import deepcopy


class MDN(ProbabilisticModel):
    """ Mixture Density Network (MDN) model for probabilistic modeling.

        This class implements a probabilistic model using a mixture density network,
        which can model complex conditional distributions of the output given the input.

        Attributes
        ----------
        Q : object
            The input query object containing the features.
        QoI_names : list of str
            The names of the quantities of interest (QoIs) corresponding to the input features.
        layers : list of dict
            The configuration of the hidden layers in the MDN, where each dictionary specifies the number of neurons, activation function, and dropout rate.
        n_components : int, optional
            The number of mixture components in the MDN (default is 5). """

    def __init__(self, Q, QoI_names, layers, n_components=5):
        """ Initialize the MDN model with the given input query, QoI names, hidden layers, and number of mixture components.

            Parameters
            ----------
            Q : object
                The input query object containing the features.
            QoI_names : list of str
                The names of the quantities of interest (QoIs) corresponding to the input features.
            layers : list of dict
                The configuration of the hidden layers in the MDN, where each dictionary specifies the number of neurons, activation function, and dropout rate.
            n_components : int, optional
                The number of mixture components in the MDN (default is 5). 
        """
        self.Q = Q
        self.QoI_names = QoI_names
        self.layers = layers
        self.model = self.MDNInner(input_size=len(self.QoI_names), output_size=self.Q.num_variables(), layers=layers, n_components=n_components)
        self.n_components = n_components

    def inference(self, features, n_samples=2000):
        """ Perform inference using the trained MDN model.

            Parameters
            ----------
            features : array-like
                The input features for which to generate predictions.
            n_samples : int, optional
                The number of samples to draw from the mixture model (default is 2000).

            Returns
            -------
            array-like
                The predicted samples for the given input features.
        """
        Xq = torch.tensor(self.q_scaler.transform(features), dtype=torch.float32)
        samples_scaled = self.model.predict(Xq, n_samples=n_samples)
        return self.y_scaler.inverse_transform(samples_scaled[0].numpy())

    def train(self, q, y, k_fold=None, **params):
        """ Train the MDN model using the provided input and output data.

            Parameters
            ----------
            q : array-like
                The input features for training.
            y : array-like
                The target outputs corresponding to the input features.
            k_fold : int, optional
                The number of folds for cross-validation (default is None, meaning no cross-validation).
            **params : dict
                Additional training parameters to be passed to the training function.
        """
        q_train, q_val, y_train, y_val = self.train_test_split(q, y, train_test_ratio=0.8, random_seed=1997, split_type='no_shuffle')

        self.train_config = params

        self.q_scaler = StandardScaler().fit(q_train)
        self.y_scaler = StandardScaler().fit(y_train)

        q_train_tensor = torch.tensor(self.q_scaler.transform(q_train), dtype=torch.float32)
        q_val_tensor = torch.tensor(self.q_scaler.transform(q_val), dtype=torch.float32)

        y_train_tensor = torch.tensor(self.y_scaler.transform(y_train), dtype=torch.float32)
        y_val_tensor = torch.tensor(self.y_scaler.transform(y_val), dtype=torch.float32)

        print(f'----- Training started for \'MDN\' model -----')
        
        if k_fold is not None:
            train_losses, val_losses = self.cross_validation(y_train_tensor, q_train_tensor, k_fold, **params)
        else:
            train_losses, val_losses = self.model.train_and_validate(q_train_tensor, y_train_tensor, q_val_tensor, y_val_tensor, **params)
        # print(f"Average train loss: {np.mean(train_losses):.14f}, Average valid loss: {np.mean(val_losses):.14f}")      
        print(f'----- Training ended for \'MDN\' model -----')

    class MDNInner(nn.Module):
        """ Inner class defining the architecture of the MDN model.

            Parameters
            ----------
            input_size : int
                The number of input features.
            output_size : int
                The number of output features.
            layers : list of dict
                A list specifying the configuration of each hidden layer. Each dictionary should contain keys 'neurons', 'activation', and 'dropout'.
            n_components : int
                The number of mixture components in the MDN.
        """
        def __init__(self, input_size, output_size, layers, n_components):
            """ Initialize the MDNInner model.

                Parameters
                ----------
                input_size : int
                    The number of input features.
                output_size : int
                    The number of output features.
                layers : list of dict
                    A list specifying the configuration of each hidden layer. Each dictionary should contain keys 'neurons', 'activation', and 'dropout'.
                n_components : int
                    The number of mixture components in the MDN.
            """
            super().__init__()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.mdn_mv_util = MDNMV(n_components, output_size)

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
            for layer in layers:
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

            # Create Sequential model
            self.hidden_layers = nn.Sequential(*hidden_layers)
            self.output_layer = nn.Sequential(nn.Linear(prev_size, self.mdn_mv_util.network_output_dim))
            
            self.all_layers = nn.Sequential(*hidden_layers, nn.Linear(prev_size, self.mdn_mv_util.network_output_dim))
            
            self.to(self.device)

        def forward(self, x):
            """ Forward pass through the MDNInner model.

                Parameters
                ----------
                x : torch.Tensor
                    The input tensor.

                Returns
                -------
                torch.Tensor
                    The output tensor after passing through all layers.
            """
            return self.all_layers(x)

        def loss_fn(self, outputs, labels):
            """ Compute the negative log-likelihood loss for the MDN model.

                Parameters
                ----------
                outputs : torch.Tensor
                    The output tensor from the forward pass.
                labels : torch.Tensor
                    The ground truth labels.

                Returns
                -------
                torch.Tensor
                    The computed loss value.
            """ 
            return -self.mdn_mv_util.log_likelihood(outputs, labels, min_log_proba=-20).mean()

        def get_batch_ixs(self, like_tensor, batch_size=16, permute=True):
            """ Get batch indices for mini-batch training.

                Parameters
                ----------
                like_tensor : torch.Tensor
                    A tensor whose first dimension determines the number of samples.
                batch_size : int, optional
                    The desired batch size (default is 16).
                permute : bool, optional
                    Whether to permute the indices before splitting into batches (default is True).

                Returns
                -------
                list of torch.Tensor
                    A list of tensors, each containing the indices for a mini-batch.
            """
            if like_tensor.shape[0] <= batch_size:
                return torch.arange(like_tensor.shape[0]).unsqueeze(0)
            ixs = torch.randperm(like_tensor.shape[0]) if permute else torch.arange(like_tensor.shape[0])
            return torch.tensor_split(ixs, ixs.shape[0] // batch_size)

        def predict(self, X, n_samples=2000):
            """ Generate predictions from the MDN model.

                Parameters
                ----------
                X : torch.Tensor
                    The input tensor for which predictions are to be made.
                n_samples : int, optional
                    The number of samples to draw from the mixture model (default is 2000).

                Returns
                -------
                torch.Tensor
                    The generated samples from the MDN model.
            """
            with torch.no_grad():
                preds = self(X)
            raw_samples = self.mdn_mv_util.sample(preds, n=n_samples)
            samples_scaled = raw_samples.transpose(0, 1) if raw_samples.shape[0] == n_samples else raw_samples
            return samples_scaled

        def train_and_validate(self, X_train, y_train, X_val, y_val, optimizer='adam', max_epoch=5, batch_size=64, lr=1e-3, weight_decay=1e-6, max_patience=2, verbose=False):
            """ Train and validate the MDN model.

                Parameters
                ----------
                X_train : torch.Tensor
                    The input tensor for the training set.
                y_train : torch.Tensor
                    The target tensor for the training set.
                X_val : torch.Tensor
                    The input tensor for the validation set.
                y_val : torch.Tensor
                    The target tensor for the validation set.
                optimizer : str, optional
                    The optimizer to use (default is 'adam').
                max_epoch : int, optional
                    The maximum number of training epochs (default is 5).
                batch_size : int, optional
                    The mini-batch size (default is 64).
                lr : float, optional
                    The learning rate (default is 1e-3).
                weight_decay : float, optional
                    The weight decay for the optimizer (default is 1e-6).
                max_patience : int, optional
                    The maximum number of epochs to wait for improvement before early stopping (default is 2).
                verbose : bool, optional
                    Whether to print training progress (default is False).

                Returns
                -------
                self
                    The trained MDN model.
                dict
                    A dictionary containing the training history, including epoch, train_loss, and val_loss.
            """
            self.batch_size = batch_size
            
            X_train, y_train = X_train.to(self.device), y_train.to(self.device)
            X_val, y_val = X_val.to(self.device), y_val.to(self.device)            

            optimizer_mapping = {
                'adamw': torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay),
                'adam': torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay),
                'RMSprop': torch.optim.RMSprop(self.parameters(), lr=lr, weight_decay=weight_decay)
            }
            optimizer = optimizer_mapping.get(optimizer.lower())

            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=10)

            best_val, best_state, bad_epochs = float("inf"), deepcopy(self.state_dict()), 0
            history = {"epoch": [], "train_loss": [], "val_loss": []}

            for epoch in range(max_epoch):
                self.train()
                losses = []
                for ixs in self.get_batch_ixs(X_train, batch_size, permute=True):
                    optimizer.zero_grad()
                    outputs = self(X_train[ixs])
                    loss = self.loss_fn(outputs, y_train[ixs])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10)
                    optimizer.step()
                    losses.append(loss.item())
                scheduler.step()

                self.eval()
                with torch.no_grad():
                    val_loss = self.loss_fn(self(X_val), y_val).item()

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