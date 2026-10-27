'''Deep Neural Network (DNN) model for regression tasks.'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import shap
import datetime
from .PredictiveModel import PredictiveModel
from sklearn.preprocessing import MinMaxScaler

class DNN(PredictiveModel):

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
        'leakyrelu', 'softmax', 'none').   '''

    def __init__(self, Q, QoI_names, layers, outputAF):
        self.Q = Q
        self.QoI_names = QoI_names
        self.layers = layers
        self.outputAF = outputAF
        self.model = self.DNNInner(self.Q.num_variables(), len(self.QoI_names), self.layers, self.outputAF).double()


    def train(self, q_train, y_train, k_fold=None, train_test_ratio=0.8, random_seed=1997, split_type='no_shuffle',**params):
        '''
        Trains the predictive model using the provided training data.

        Parameters:
        -----------
        q_train : pd.DataFrame
            Training input features
        y_train : pd.DataFrame
            Training target values
        q_scaler : object
            Scaler for input features (default is None)
        y_scaler : object
            Scaler for target values (default is None)
        k_fold : int
            Number of folds for cross-validation (default is None)
        **params : dict
            Additional training parameters

        Returns:
        -------
        None
        '''

        self.train_config = params
        self.q_scaler = self.get_scaler(q_train)
        self.y_scaler = self.get_scaler(y_train)
        xi = self.get_scaled_q(q_train)
        yt = self.get_scaled_y(y_train) # y transformed
        print(f'----- Training started for \'DNN\' model -----')
        
        if k_fold is not None:
            train_losses, val_losses = self.cross_validation(xi, yt, k_fold, **params)
        else:
            xi_train, xi_val, yt_train, yt_val = self.train_test_split(xi, yt, train_test_ratio=train_test_ratio, random_seed=random_seed, split_type=split_type)
            train_losses, val_losses = self.train_and_validate(xi_train, yt_train, xi_val, yt_val, **params)
        print(f"Average train loss: {np.mean(train_losses):.14f}, Average valid loss: {np.mean(val_losses):.14f}")      
        print(f'----- Training ended for \'DNN\' model -----')

    def train_and_validate(self, xi_train, yt_train, xi_val, yt_val, loss='mse', optimizer='adam', epochs=5, batch_size=64, early_stopping=None):
        xi_tr,xi_vl = torch.tensor(xi_train, dtype=torch.float64), torch.tensor(xi_val, dtype=torch.float64)
        yt_tr,yt_vl = torch.tensor(yt_train, dtype=torch.float64), torch.tensor(yt_val, dtype=torch.float64)
        tr_loss, vl_loss = self.model.train_and_validate(xi_tr, yt_tr, xi_vl, yt_vl, loss=loss, optimizer=optimizer, epochs=epochs, batch_size=batch_size, early_stopping=early_stopping)
        return tr_loss, vl_loss

    def predict(self, q):
        if type(q) != pd.DataFrame:
            q = pd.DataFrame(q, columns=self.Q.variable_names())
        xi = self.get_scaled_q(q)
        xi = torch.tensor(xi, dtype=torch.float64)
        y_t = self.model.predict(xi)
        y = self.get_orig_y(y_t)
        return y

    def compute_shap_values(self, predict_fn, q, forced=False, explainer_type="kernelexplainer", silent=False):
        shap_values = self.model.compute_shap_values(q, predict_fn=predict_fn, forced=forced, explainer_type=explainer_type, silent=silent)
        self.explainer = self.model.explainer
        return shap_values

    
    def to_jsonld(self, model_id=None, building_did=None, building_ual=None, name=None, doc_type=None, description=None):

        hyperparameters = []
        hyperparameter_settings = []

        for hp_name, value in self.hyper_params.items():

            if value is None:
                continue

            hyper_id = f"#{hp_name}"

            hyperparameters.append({
                "@id": hyper_id,
                "@type": "mls:HyperParameter",
                "rdfs:label": hp_name
            })

            hyperparameter_settings.append({
                "@id": f"#{hp_name}_setting",
                "@type": "mls:HyperParameterSetting",
                "mls:specifiedBy": {
                    "@id": hyper_id
                },
                "mls:hasValue": value
            })

        mls_spec = {
            "@id": f"urn:implementation:{model_id}",
            "@type": "mls:Implementation",

            "mls:implements": {
                "@id": "https://www.wikidata.org/wiki/Q192776",
                "@type": "mls:Algorithm",
                "rdfs:label": "Deep Neural Network"
            },

            "mls:hasHyperParameter": hyperparameters
        }

        if building_did is not None:
            jsonld = {
                "@context": {
                    "schema": "https://schema.org/",
                    "dcterms": "http://purl.org/dc/terms/",
                    "dkg": "https://origintrail.io/ontology/",
                    "mls": "http://www.w3.org/ns/mls#",
                    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                    "prov": "http://www.w3.org/ns/prov#"
                },

                "@id": model_id,
                "@type": "schema:Dataset",

                "schema:name": name,
                "dcterms:type": doc_type,
                "schema:description": description,
                "schema:hasPart": {"@id": building_did, "dkg:ual": building_ual},
                "dcterms:created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "prov:wasGeneratedBy": {
                    "@type": "prov:Activity",
                    "schema:name": "Trace Structures API",
                    "schema:url": "https://buildchain.ilab.sztaki.hu/"
                },

                "mls:specifiedBy": mls_spec,
                "mls:hasPart": hyperparameter_settings
            }
        else:
            jsonld = {
                "@context": {
                    "mls": "http://www.w3.org/ns/mls#",
                    "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
                },

                "@id": f"urn:model:{model_id}",
                "@type": "mls:Model",

                "mls:specifiedBy": mls_spec,
                "mls:hasPart": hyperparameter_settings
            }

        return jsonld

    def get_scaler(self, q):

        scaler = MinMaxScaler((0.05, 0.95))
        scaler.fit(q)
        return scaler

    def get_scaled_q(self, q):
        return self.q_scaler.transform(q)

    def get_scaled_y(self, y):
        return self.y_scaler.transform(y)

    def get_orig_q(self, xi):
        '''
        Inversely transforms the scaled input features back to their original scale.

        Parameters:
        -----------
        xi : np.ndarray
            Scaled input features

        Returns:
        -------
        q : pd.DataFrame
            Original input features
        '''

        q = self.q_scaler.inverse_transform(xi)
        return q

    def get_orig_y(self, yi):
        '''
        Inversely transforms the scaled target features back to their original scale.

        Parameters:
        -----------
        yi : np.ndarray
            Scaled target features

        Returns:
        -------
        y : pd.DataFrame
            Original target features
        '''

        y = self.y_scaler.inverse_transform(yi)
        return y


    class DNNInner(nn.Module):
        
        def __init__(self, input_size, output_size, layers, outputAF):
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
            
            self.outputAF = activation_mapping.get(outputAF.lower()) if outputAF else nn.Identity()

            hidden_layers = []
            prev_size = input_size

            # Build hidden layers
            for layer in layers:
                neurons = default_neurons if layer['neurons'] is None else layer['neurons']
                activation = default_activation if layer['activation'] is None else layer['activation']
                dropout = default_dropout if layer['dropout'] is None else layer['dropout']

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


        def compute_shap_values(self, q, predict_fn, forced=False, explainer_type="kernelexplainer", silent=False):
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
                shap_values = self.explainer(xi, silent=silent)
            elif explainer_type == "kernelexplainer":
                if hasattr(self, 'explainer') == False or forced == True:
                    explainer = shap.KernelExplainer(predict_fn, q)
                    self.explainer = explainer
                shap_values = self.explainer(q, silent=silent)
            return shap_values





    

