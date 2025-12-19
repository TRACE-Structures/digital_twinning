'''Deep Neural Network (DNN) model for regression tasks.'''
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import shap
from SALib.analyze import sobol

class DNNModel(nn.Module):
    '''
    Deep Neural Network (DNN) model for regression tasks.
    
    Attributes
    ----------
    input_size : int
        Number of input features.

    output_size : int
        Number of output targets.

    layers : list of dict
        Configuration of hidden layers, each specified as a dictionary with keys:
        'neurons': int
            Number of neurons in the layer.
        'activation': str
            Activation function for the layer ('relu', 'sigmoid',
            'tanh', 'gelu', 'leakyrelu', 'softmax', 'none').
        'dropout': float
            Dropout rate for the layer (0.0 to 1.0).

    outputAF : str
        Activation function for the output layer ('relu', 'sigmoid', 'tanh', 'gelu',
        'leakyrelu', 'softmax', 'none').

    Methods
    -------
    forward(self, x)
        Defines the forward pass of the DNN.

    train_and_validate(self, q_train, y_train, q_val, y_val, loss='mse', optimizer='adam',
                       epochs=5, batch_size=64, early_stopping=None)
        Trains and validates the DNN model.
    
    evaluate(self, test_dataset)
        Evaluates the model on a test dataset.
    
    predict(self, q)
        Makes predictions using the trained model.
    
    compute_partial_vars(self, model_obj, max_index)
        Computes partial variances using Sobol sensitivity analysis.
    
    get_shap_values(self, predict_fn, q, forced=False, explainer_type="kernelexplainer")
        Computes SHAP values for model interpretability.
    
    to_jsonld(self, model_id=None)
        Converts the model to JSON-LD format.    
    '''

    def __init__(self, input_size, output_size, layers, outputAF):
        '''
        Initializes the DNNModel with the specified architecture.

        Parameters:
        ----------
        input_size : int
            Number of input features.
        output_size : int
            Number of output targets.
        layers : list of dict
            Configuration of hidden layers.
        outputAF : str
            Activation function for the output layer.
        '''

        super(DNNModel, self).__init__()
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
        
        self.outputAF = activation_mapping.get(outputAF.lower()) if outputAF else nn.Identity()

        hidden_layers = []
        prev_size = input_size

        # Build hidden layers
        for layer in layers:
            neurons = default_neurons# if layer['neurons'] is None else layer['neurons']
            activation = default_activation# if layer['activation'] is None else layer['activation']
            dropout = default_dropout# if layer['dropout'] is None else layer['dropout']

            # Add Linear, Activation, and Dropout layers to the list
            hidden_layers.append(nn.Linear(prev_size, neurons))
            hidden_layers.append(activation_mapping.get(activation.lower()))
            hidden_layers.append(nn.Dropout(dropout))
            prev_size = neurons

        # Create Sequential model
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Sequential(nn.Linear(prev_size, output_size), self.outputAF)
        
        self.all_layers = nn.Sequential(*hidden_layers, nn.Linear(prev_size, output_size), self.outputAF)
        
        self.to(self.device)

    def forward(self, x):
        '''
        Defines the forward pass of the DNN.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor.

        Returns:
        -------
        out_x : torch.Tensor
            Output tensor after passing through the network.
        '''        

        out_x = self.all_layers(x)
        return out_x
    
    def train_and_validate(self, q_train, y_train, q_val, y_val, loss='mse', optimizer='adam', epochs=5, batch_size=64, early_stopping=None):
        '''
        Trains and validates the DNN model.

        Parameters
        ----------
        q_train : torch.Tensor
            Training input data.
        y_train : torch.Tensor
            Training target data.
        q_val : torch.Tensor
            Validation input data.
        y_val : torch.Tensor
            Validation target data.
        loss : str, optional
            Loss function to use ('mse', 'mae', 'crossentropy'). Default is
            'mse'.
        optimizer : str, optional
            Optimizer to use ('adam', 'RMSprop'). Default is 'adam'.
        epochs : int, optional
            Number of training epochs. Default is 5.
        batch_size : int, optional
            Batch size for training. Default is 64.
        early_stopping : dict, optional
            Early stopping configuration with keys:
            'patience': int
                Number of epochs with no improvement to wait before stopping.
            'min_delta': float
                Minimum change to qualify as an improvement.
            'monitor': str
                Metric to monitor ('val_loss', 'val_accuracy').
            'mode': str
                'min' for loss, 'max' for accuracy. Default is None (no early stopping).

        Returns
        -------
        train_loss : float
            Final training loss.
        val_loss : float
            Final validation loss.
        '''

        self.batch_size = batch_size
        train_loader = DataLoader(TensorDataset(q_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(q_val, y_val), batch_size=batch_size, shuffle=False)
        
        # Define a mapping from string names to PyTorch criterion functions
        criterion_mapping = {
            'mse': nn.MSELoss(),
            'mae': nn.L1Loss(),
            'crossentropy': nn.CrossEntropyLoss()
        }
        criterion = criterion_mapping.get(loss.lower())

        # Define a mapping from string names to PyTorch optimizer functions
        optimizer_mapping = {
            'adam': optim.Adam(self.parameters()),
            'RMSprop': optim.RMSprop(self.parameters())
        }
        optimizer = optimizer_mapping.get(optimizer.lower())

        self.hyper_params = {
            'loss': criterion.__class__.__name__,
            'optimizer': optimizer.__class__.__name__,
            'epochs': epochs,
            'batch_size': batch_size,
            'early_stopping': early_stopping
        }
        
        # Early stopping initialization
        if early_stopping is not None:
            # Extract parameters with defaults
            patience = early_stopping.get('patience', 15)
            min_delta = early_stopping.get('min_delta', 0.0)
            monitor = early_stopping.get('monitor', 'val_loss')  # e.g., 'val_loss', 'val_accuracy'
            mode = early_stopping.get('mode', 'min')  # 'min' for loss, 'max' for accuracy

            # Initialize tracking variables based on mode
            best_metric = float('inf') if mode == 'min' else -float('inf')
            epochs_no_improve = 0
            early_stop = False
            best_model_state = None  # To store the best model's state

        for epoch in range(epochs+1):
            self.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()   
            avg_train_loss = running_loss / len(train_loader)

            # Validation phase
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            
            # Calculate monitored metric (for simplicity, we only track val_loss here)
            monitored_value = avg_val_loss  # Replace with your metric (e.g., accuracy) if needed

            
            if epoch % 50 == 0:
                print(f"Epoch [{epoch}/{epochs}], train loss: {avg_train_loss:.4f}, validation loss: {avg_val_loss:.4f}")
            
            # Early stopping logic
            if early_stopping is not None:
                if mode == 'min':
                    improvement = (monitored_value < best_metric - min_delta)
                elif mode == 'max':
                    improvement = (monitored_value > best_metric + min_delta)
                else:
                    raise ValueError(f"Invalid mode: {mode}. Use 'min' or 'max'.")

                if improvement:
                    best_metric = monitored_value
                    epochs_no_improve = 0
                    best_model_state = self.state_dict().copy()  # Save best model
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f'Early stopping triggered after {epoch + 1} epochs!')
                        early_stop = True
                        break

        # Restore the best model if early stopping was triggered
        if early_stopping is not None and early_stop:
            print(f'Training stopped early. Restoring best model (monitored {monitor} = {best_metric:.4f}).')
            self.load_state_dict(best_model_state)
        else:
            print('Training completed without early stopping.')  
        return avg_train_loss, avg_val_loss

    def evaluate(self, test_dataset):
        '''
        Evaluates the model on a test dataset.

        Parameters
        ----------
        test_dataset : TensorDataset
            Test dataset containing input and target data.

        Returns
        -------
        mse : float
            Mean Squared Error on the test dataset.
        '''

        self.eval()
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        predictions = []
        targets = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                predictions.extend(outputs.cpu().numpy())
                targets.extend(labels.cpu().numpy())

        # Calculate the mean squared error for regression
        mse = mean_squared_error(targets, predictions)
        print(f"Mean Squared Error: {mse:.4f}")
        return mse
    
    def predict(self, q):
        '''
        Makes predictions using the trained model.

        Parameters
        ----------
        q : torch.Tensor
            Input data for prediction.

        Returns
        -------
        predictions : np.ndarray
            Predicted outputs as a numpy array.
        '''

        self.eval()  # Set the model to evaluation mode
        test_loader = DataLoader(q, batch_size=64, shuffle=False)  # Create DataLoader for test data
        predictions = []  # List to store the predictions

        with torch.no_grad():  # Disable gradient calculation as we only need predictions
            for inputs in test_loader:  # We don't need the labels (targets), so we use "_" to ignore them
                inputs = inputs.to(self.device)
                outputs = self(inputs)  # Get the model's predictions (outputs)
                predictions.extend(outputs.cpu().numpy())  # Add the predictions to the list (move to CPU and convert to numpy)     
        predictions = np.array(predictions)

        return predictions  # Return the list of predictions
    
    def compute_partial_vars(self, model_obj, max_index):
        '''
        Computes partial variances using Sobol sensitivity analysis.

        Parameters
        ----------
        model_obj : DNNModel
            The trained DNN model object.
        max_index : int
            Maximum Sobol index to compute (1 or 2).

        Returns
        -------
        partial_var_df : pd.DataFrame
            DataFrame containing partial variances for each parameter and QoI.
        sobol_index_df : pd.DataFrame
            DataFrame containing Sobol indices for each parameter and QoI.
        y_var : np.ndarray
            Variance of the model outputs.
        ''' 

        paramset = model_obj.Q
        QoI_names = model_obj.QoI_names
        
        problem = {
            'num_vars': paramset.num_params(), 'names': paramset.param_names(), 'dists': paramset.get_dist_types(), 'bounds': paramset.get_dist_params()
            } 
        
        d = paramset.num_params()
        q = paramset.sample(method='Sobol_saltelli', n=8192) # saltelli working only for uniform distribution # N * (2D + 2)
        # https://salib.readthedocs.io/en/latest/user_guide/advanced.html

        y = model_obj.predict(q)
        
        # Run model
        S1 = []
        S2 = []
        for i in range(y.shape[1]):
            y_i = y[:,i]

            # Sobol analysis
            Si_i = sobol.analyze(problem, y_i)
            T_Si, first_Si, (idx, second_Si) = sobol.Si_to_pandas_dict(Si_i)
            df = Si_i.to_df()
            cols_S1 = list(df[1].index)
            cols_S2 = list(df[2].index)

            S1.append(first_Si['S1'])
            S2.append(second_Si['S2'])

        S1 = np.array(S1)
        S2 = np.array(S2)

        col_names = cols_S1
        sobol_index = S1
        if max_index == 2:
            sobol_index = np.concatenate([S1, S2], axis=1)
            col_names = cols_S1 + cols_S2
            col_names = [f"{x[0]} {x[1]}" if isinstance(x, tuple) else x for x in col_names]
                    
        # Compute partial variances
        y_var = y.var(axis=0).reshape(-1, 1)
        partial_variance = sobol_index * y_var
             
        partial_var_df, sobol_index_df = pd.DataFrame(partial_variance, columns=col_names, index=QoI_names), pd.DataFrame(sobol_index, columns=col_names, index=QoI_names)

        return partial_var_df, sobol_index_df, y_var
    
    def get_shap_values(self, predict_fn, q, forced=False, explainer_type="kernelexplainer"):
        '''
        Computes SHAP values for model interpretability.

        Parameters
        ----------
        predict_fn : function
            Prediction function that takes input data and returns model predictions.
        q : np.ndarray or torch.Tensor
            Input data for SHAP value computation.
        forced : bool, optional
            If True, forces re-computation of SHAP values even if already computed.
        explainer_type : str, optional
            Type of SHAP explainer to use ('deepexplainer' or 'kernelexplainer'). Default is 'kernelexplainer'.
        
        Returns
        -------
        shap_values : np.ndarray
            Computed SHAP values for the input data.
        '''
        
        self.eval()
        if explainer_type == "deepexplainer":
            # Determine device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)
            
            # Prepare input
            if isinstance(q, pd.DataFrame):
                xi = torch.from_numpy(q.values).double().to(device)
            elif isinstance(q, np.ndarray):
                xi = torch.from_numpy(q).double().to(device)
            elif isinstance(q, torch.Tensor):
                xi = q.double().to(device)
            else:
                raise ValueError(f"Input q must be a numpy array or a torch tensor. The type of q is: {type(q)}")
            
            #xi = torch.tensor(xi, dtype=torch.float64)
            if hasattr(self, 'explainer') == False or forced == True:
                print(type(xi))
                self.eval()
                explainer = shap.DeepExplainer(self, xi)
                self.explainer = explainer
                #shap_values = explainer.shap_values(xi)
            self.eval()
            shap_values = self.explainer(xi)
        elif explainer_type == "kernelexplainer":
            if hasattr(self, 'explainer') == False or forced == True:
                explainer = shap.KernelExplainer(predict_fn, q)
                self.explainer = explainer
            shap_values = self.explainer(q)
        return shap_values
    
    def to_jsonld(self, model_id=None):
        '''
        Serializes the model to JSON-LD format.

        Parameters
        ----------
        model_id : str
            Unique identifier for the model.

        Returns
        -------
        jsonld : dict
            JSON-LD representation of the model.
        '''

        jsonld = {
            "@context": {
                "mls": "https://ml-schema.github.io/documentation/mls.html",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
            },

            "@id": f"https://example.org/models/{model_id}",
            "@type": "mls:Model",
            "mls:implementsAlgorithm": {
                "@id": "https://en.wikipedia.org/wiki/Deep_learning",
                "@type": "mls:Algorithm",
                "rdfs:label": "Deep Neural Network"
            },
            "mls:hasHyperParameter": [
                {
                    "@type": "mls:HyperParameterSetting",
                    "mls:hasParameterName": name,
                    "mls:hasParameterValue": str(value).lower() if isinstance(value, bool) else str(value)
                }
                for name, value in self.hyper_params.items()
            ],
    
            "mls:hasPart": [
                {
                    "@type": "mls:NeuralNetwork",
                    "mls:hasLayer": []
                }
            ]
        }

        layer_list = []
        index = 0
        for layer in self.all_layers:
            if isinstance(layer, nn.Linear):
                layer_entry = {
                    "@type": "mls:DenseLayer",
                    "mls:layerIndex": index,
                    "mls:units": layer.out_features,
                    "mls:inputDim": layer.in_features,
                    "mls:activationFunction": None  # Will be filled by next activation
                }
                layer_list.append(layer_entry)
                index += 1
            elif isinstance(layer, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.GELU, nn.Softmax, nn.LeakyReLU)):
                af = type(layer).__name__.lower()
                # Fill last layer's activationFunction
                layer_list[-1]["mls:activationFunction"] = af
            elif isinstance(layer, nn.Dropout):
                # Add dropout rate to last layer
                layer_list[-1]["mls:dropoutRate"] = layer.p

        jsonld["mls:hasPart"][0]["mls:hasLayer"] = layer_list
    
        return jsonld