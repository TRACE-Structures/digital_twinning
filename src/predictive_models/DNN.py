'''Deep Neural Network (DNN) model for regression tasks.'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import shap
import datetime
from .PredictiveModel import PredictiveModel
from sklearn.preprocessing import MinMaxScaler
import warnings
from ..utils.gpu_utils import use_device

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
        """ Initialize the DNN model.

            Parameters
            ----------
            Q : object
                Input feature space object
            QoI_names : list of str
                Names of the quantities of interest (output targets)
            layers : list of dict
                Configuration of hidden layers
            outputAF : str
                Activation function for the output layer """
        self.Q = Q
        self.QoI_names = QoI_names
        self.layers = layers
        self.outputAF = outputAF
        self.predict_device = 'cpu'
        self.predict_gpu_memory_fraction = None
        self.model = self._build_inner()

    def _build_inner(self):
        """ Create a freshly initialized inner network (float64, on the CPU). """
        return self.DNNInner(self.Q.num_variables(), len(self.QoI_names), self.layers, self.outputAF).double()


    # Old option names of DNN.train, translated to the options shared with MDN / CRPS
    _LEGACY_OPTIONS = {'epochs': 'max_epoch'}

    @staticmethod
    def _translate_legacy_options(params):
        """ Map the old options 'epochs' and 'early_stopping' to the shared ones, with a warning. """
        params = dict(params)
        for old, new in DNN._LEGACY_OPTIONS.items():
            if old in params:
                warnings.warn(f"DNN option '{old}' is renamed to '{new}' (same name as in MDN / CRPS).", FutureWarning,
                              stacklevel=3)
                params.setdefault(new, params.pop(old))
        if 'early_stopping' in params:
            es = params.pop('early_stopping')
            warnings.warn("DNN option 'early_stopping' is replaced by max_patience and min_delta (as in MDN / CRPS); "
                          "early stopping on the validation loss is now always on.", FutureWarning, stacklevel=3)
            if es:
                if es.get('mode', 'min') != 'min' or es.get('monitor', 'val_loss') != 'val_loss':
                    raise ValueError("Only early stopping on a decreasing validation loss is supported "
                                     "(monitor='val_loss', mode='min').")
                if 'patience' in es:
                    params.setdefault('max_patience', es['patience'])
                if 'min_delta' in es:
                    params.setdefault('min_delta', es['min_delta'])
        return params

    def train(self, q_train, y_train, k_fold=None, train_test_ratio=0.8, random_seed=1997, split_type='shuffle',
              device='cpu', **params):
        '''
        Trains the predictive model using the provided training data.

        The options and the training procedure are the same as for the MDN and CRPS
        models: early stopping on the validation loss (best weights restored),
        learning-rate warm-up, gradient clipping, cross-validation with a fresh
        network per fold followed by a final fit.

        Devices: everything runs on the CPU unless a device such as 'cuda:1' is given
        explicitly. Only the network's forward and backward passes run on that device;
        the data and scaling stay on the CPU. After training the network is back on
        the CPU and the GPU memory is released.

        Parameters:
        -----------
        q_train : pd.DataFrame
            Training input features
        y_train : pd.DataFrame
            Training target values
        k_fold : int, optional
            If given, run k-fold cross-validation on the training part first (losses
            in self.cv_train_losses / self.cv_val_losses), then fit the final model.
        train_test_ratio : float, optional
            Fraction of the rows used for training; the rest is the validation set
            for early stopping (default 0.8)
        random_seed : int, optional
            Seed of the train/validation split and the cross-validation folds (default 1997)
        split_type : {'shuffle', 'no_shuffle', 'reverse'}, optional
            How the validation set is chosen (default 'shuffle'; 'no_shuffle' takes the last rows)
        device : str, optional
            'cpu' (default) or an explicit GPU such as 'cuda:1'.
        **params : dict
            Passed to DNNInner.train_and_validate: gpu_memory_fraction, optimizer,
            max_epoch, batch_size, val_batch_size, lr, weight_decay, max_patience,
            min_delta, verbose, log_every, loss.
        '''
        params = self._translate_legacy_options(params)
        self._check_finite(q_train, y_train)
        q_tr, q_vl, y_tr, y_vl = self.train_test_split(q_train, y_train, train_test_ratio=train_test_ratio,
                                                       random_seed=random_seed, split_type=split_type)
        # Keep only the training part as data of the model (as in MDN / CRPS)
        self.samples, self.QoI = q_tr, y_tr
        self.train_config = dict(params, device=str(device), split_type=split_type, random_seed=random_seed,
                                 train_test_ratio=train_test_ratio)

        # Scalers are fitted on the training part only, so the validation rows stay unseen
        self.q_scaler = self.get_scaler(q_tr)
        self.y_scaler = self.get_scaler(y_tr)
        xi_train, xi_val = self.get_scaled_q(q_tr), self.get_scaled_q(q_vl)
        yt_train, yt_val = self.get_scaled_y(y_tr), self.get_scaled_y(y_vl)
        print(f'----- Training started for \'DNN\' model -----')

        if k_fold is not None:
            self.cv_train_losses, self.cv_val_losses = self.cross_validation(
                xi_train, yt_train, k_fold, random_seed=random_seed, device=device, **params)
        self.train_and_validate(xi_train, yt_train, xi_val, yt_val, device=device, **params)
        print(f'----- Training ended for \'DNN\' model -----')

    def train_and_validate(self, xi_train, yt_train, xi_val, yt_val, **params):
        """ Fit a freshly initialized network (one cross-validation fold, or the final fit).

            Parameters
            ----------
            xi_train, yt_train, xi_val, yt_val : array_like or torch.Tensor
                Scaled training and validation data
            **params : dict
                See DNNInner.train_and_validate.

            Returns
            -------
            tr_loss : float
                Training loss of the last epoch.
            vl_loss : float
                Best validation loss.
        """
        params = self._translate_legacy_options(params)
        as_tensor = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64))
        self.model = self._build_inner()
        _, history = self.model.train_and_validate(as_tensor(xi_train), as_tensor(yt_train),
                                                   as_tensor(xi_val), as_tensor(yt_val), **params)
        self.history = history
        self.hyper_params = self.model.hyper_params
        return history["train_loss"][-1], float(np.nanmin(history["val_loss"]))

    def predict(self, q, device=None, gpu_memory_fraction=None, batch_size=1024):
        """ Predict the target values for the given input features.

            Parameters
            ----------
            q : array_like or pandas.DataFrame
                Input features for prediction
            device : str, optional
                Device of the forward pass: 'cpu' or e.g. 'cuda:1' (defaults to
                self.predict_device, i.e. 'cpu')
            gpu_memory_fraction : float, optional
                GPU memory limit, see utils.gpu_utils.resolve_device
            batch_size : int, optional
                Rows per forward pass (default 1024)

            Returns
            -------
            y : array_like
                Predicted target values
        """
        if device is None:
            device = getattr(self, 'predict_device', 'cpu')
            if gpu_memory_fraction is None:
                gpu_memory_fraction = getattr(self, 'predict_gpu_memory_fraction', None)
        if type(q) != pd.DataFrame:
            q = pd.DataFrame(q, columns=self.Q.variable_names())
        xi = self.get_scaled_q(q)
        xi = torch.tensor(xi, dtype=torch.float64)
        with use_device(self.model, device, gpu_memory_fraction):
            y_t = self.model.predict(xi, batch_size=batch_size)
        y = self.get_orig_y(y_t)
        return y

    def compute_shap_values(self, predict_fn, q, forced=False, explainer_type="kernelexplainer", silent=False):
        """ Compute SHAP values for the given input features.

            Parameters
            ----------
            predict_fn : callable
                Prediction function to use for SHAP value computation
            q : array_like or pandas.DataFrame
                Input features for SHAP value computation
            forced : bool, optional
                Whether to force recomputation of SHAP values (default is False)
            explainer_type : str, optional
                Type of SHAP explainer to use (default is "kernelexplainer")
            silent : bool, optional
                Whether to suppress output messages (default is False)

            Returns
            -------
            shap_values : array_like
                Computed SHAP values for the input features
        """
        shap_values = self.model.compute_shap_values(q, predict_fn=predict_fn, forced=forced, explainer_type=explainer_type, silent=silent)
        self.explainer = self.model.explainer
        return shap_values

    
    def to_jsonld(self, model_id=None, building_did=None, building_ual=None, name=None, doc_type=None, description=None):
        """ Convert the DNN model to JSON-LD format.

            Parameters
            ----------
            model_id : str, optional
                Identifier for the model (default is None)
            building_did : str, optional
                Decentralized identifier for the building (default is None)
            building_ual : str, optional
                Uniform Asset Locator for the building (default is None)
            name : str, optional
                Name of the model (default is None)
            doc_type : str, optional
                Document type for the model (default is None)
            description : str, optional
                Description of the model (default is None)

            Returns
            -------
            jsonld : dict
                JSON-LD representation of the model
        """

        hyperparameters = []
        hyperparameter_settings = []

        # Models pickled before this version keep the settings on the inner network only.
        hyper_params = getattr(self, 'hyper_params', None) or getattr(self.model, 'hyper_params', {})
        for hp_name, value in hyper_params.items():

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
        """ Get a MinMaxScaler fitted to the input features.

            Parameters
            ----------
            q : array_like or pandas.DataFrame
                Input features to fit the scaler

            Returns
            -------
            scaler : sklearn.preprocessing.MinMaxScaler
                Fitted MinMaxScaler object
        """

        scaler = MinMaxScaler((0.05, 0.95))
        scaler.fit(q)
        return scaler

    def get_scaled_q(self, q):
        """ Get the scaled input features using the fitted MinMaxScaler.

            Parameters
            ----------
            q : array_like or pandas.DataFrame
                Input features to be scaled

            Returns
            -------
            scaled_q : array_like
                Scaled input features
        """
        return self.q_scaler.transform(q)

    def get_scaled_y(self, y):
        """ Get the scaled target features using the fitted MinMaxScaler.

            Parameters
            ----------
            y : array_like or pandas.DataFrame
                Target features to be scaled

            Returns
            -------
            scaled_y : array_like
                Scaled target features
        """
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
        """ Inner class representing the DNN model architecture.

            The network lives on the CPU; training and prediction move it to the
            requested device only for the duration of the call.

            Attributes
            ----------
            hidden_layers : nn.Sequential
                Sequential container for hidden layers
            output_layer : nn.Sequential
                Sequential container for the output layer
            all_layers : nn.Sequential
                Sequential container for all layers (hidden + output)
            outputAF : nn.Module
                Activation function for the output layer
        """

        def __init__(self, input_size, output_size, layers, outputAF):
            """ Initializes the DNNInner model.

            Parameters
            ----------
            input_size : int
                Number of input features
            output_size : int
                Number of output features
            layers : list of dict
                List of dictionaries specifying the hidden layers configuration
                ('neurons', 'activation', 'dropout'; missing keys or None use the
                defaults 32, 'relu', 0.0)
            outputAF : str
                Activation function for the output layer
            """
            super().__init__()
            default_neurons = 32
            default_activation = 'relu'
            default_dropout = 0.0

            # Factories, so every layer gets its own module instance
            activation_mapping = {
                'gelu': nn.GELU,
                'relu': nn.ReLU,
                'sigmoid': nn.Sigmoid,
                'tanh': nn.Tanh,
                'leakyrelu': nn.LeakyReLU,
                'softmax': lambda: nn.Softmax(dim=1),
                'none': nn.Identity
            }

            def make_activation(name):
                if name.lower() not in activation_mapping:
                    raise ValueError(f"Unknown activation '{name}'; use one of {list(activation_mapping)}")
                return activation_mapping[name.lower()]()

            self.outputAF = make_activation(outputAF) if outputAF else nn.Identity()

            hidden_layers = []
            prev_size = input_size

            # Build hidden layers
            for layer in layers:
                neurons = layer.get('neurons')
                activation = layer.get('activation')
                dropout = layer.get('dropout')
                neurons = default_neurons if neurons is None else neurons
                activation = default_activation if activation is None else activation
                dropout = default_dropout if dropout is None else dropout

                # Add Linear, Activation, and Dropout layers to the list
                hidden_layers.append(nn.Linear(prev_size, neurons))
                hidden_layers.append(make_activation(activation))
                hidden_layers.append(nn.Dropout(dropout))
                prev_size = neurons

            # Create Sequential model
            self.hidden_layers = nn.Sequential(*hidden_layers)
            self.output_layer = nn.Sequential(nn.Linear(prev_size, output_size), self.outputAF)

            self.all_layers = nn.Sequential(*hidden_layers, nn.Linear(prev_size, output_size), self.outputAF)

        @property
        def device(self):
            """ Device the network currently lives on (the CPU between calls). """
            return next(self.parameters()).device

        def get_batch_ixs(self, like_tensor, batch_size=16, permute=True):
            if like_tensor.shape[0] <= batch_size:
                return torch.arange(like_tensor.shape[0]).unsqueeze(0)
            ixs = torch.randperm(like_tensor.shape[0]) if permute else torch.arange(like_tensor.shape[0])
            return torch.tensor_split(ixs, ixs.shape[0] // batch_size)

        def _validation_loss(self, X_val, y_val, criterion, device, batch_size):
            """ Mean validation loss, computed in batches (the data stay on the CPU). """
            total = 0.0
            for start in range(0, X_val.shape[0], batch_size):
                xb = X_val[start:start + batch_size].to(device)
                yb = y_val[start:start + batch_size].to(device)
                total += criterion(self(xb), yb).item() * xb.shape[0]
            return total / X_val.shape[0]

        def train_and_validate(self, X_train, y_train, X_val, y_val, device='cpu', gpu_memory_fraction=None,
                               optimizer='adam', max_epoch=500, batch_size=64, val_batch_size=1024, lr=1e-3,
                               weight_decay=1e-6, max_patience=20, min_delta=1e-6, verbose=False, log_every=10,
                               loss='mse'):
            '''
            Train with early stopping on the validation loss (same procedure as MDN / CRPS).

            The data stay on the CPU; only the current batch is moved to the device.

            Parameters
            ----------
            X_train, y_train, X_val, y_val : torch.Tensor
                Scaled training and validation data (float64, on the CPU).
            device : str, optional
                'cpu' (default) or an explicit GPU such as 'cuda:1'.
            gpu_memory_fraction : float, optional
                GPU memory limit, see utils.gpu_utils.resolve_device.
            optimizer : {'adam', 'adamw', 'rmsprop'}, optional
                Optimizer (default 'adam').
            max_epoch : int, optional
                Maximum number of epochs (default 500).
            batch_size : int, optional
                Training batch size (default 64).
            val_batch_size : int, optional
                Batch size of the validation pass (default 1024).
            lr : float, optional
                Learning rate (default 1e-3), warmed up linearly from lr/10 over 10 epochs.
            weight_decay : float, optional
                Weight decay (default 1e-6).
            max_patience : int, optional
                Stop after this many epochs without improvement (default 20).
            min_delta : float, optional
                Minimum decrease of the validation loss that counts as an improvement (default 1e-6).
            verbose : bool, optional
                Print the losses (default False).
            log_every : int, optional
                With verbose, print every log_every epochs (default 10).
            loss : {'mse', 'mae', 'crossentropy'}, optional
                Training loss (default 'mse'). DNN only: MDN and CRPS have fixed losses.

            Returns
            -------
            self
                The trained network (best validation state), on the CPU.
            dict
                History with keys 'epoch', 'train_loss', 'val_loss'.
            '''
            self.batch_size = batch_size

            criterion_mapping = {'mse': nn.MSELoss, 'mae': nn.L1Loss, 'crossentropy': nn.CrossEntropyLoss}
            if loss.lower() not in criterion_mapping:
                raise ValueError(f"Unknown loss '{loss}'; use one of {list(criterion_mapping)}")
            criterion = criterion_mapping[loss.lower()]()

            optimizers = {'adamw': optim.AdamW, 'adam': optim.Adam, 'rmsprop': optim.RMSprop}
            if optimizer.lower() not in optimizers:
                raise ValueError(f"Unknown optimizer '{optimizer}'; use one of {list(optimizers)}")

            self.hyper_params = {
                'loss': criterion.__class__.__name__,
                'optimizer': optimizers[optimizer.lower()].__name__,
                'max_epoch': max_epoch,
                'batch_size': batch_size,
                'lr': lr,
                'weight_decay': weight_decay,
                'max_patience': max_patience,
                'min_delta': min_delta,
            }

            history = {"epoch": [], "train_loss": [], "val_loss": []}
            # The loop runs in its own method, so all its GPU tensors (batches, optimizer
            # state) are freed before use_device releases the GPU memory.
            with use_device(self, device, gpu_memory_fraction) as dev:
                best_state = self._fit(dev, X_train, y_train, X_val, y_val, history, criterion,
                                       optimizers[optimizer.lower()], max_epoch, batch_size, val_batch_size,
                                       lr, weight_decay, max_patience, min_delta, verbose, log_every)

            self.load_state_dict(best_state)
            self.eval()
            return self, history

        def _fit(self, dev, X_train, y_train, X_val, y_val, history, criterion, optimizer_cls, max_epoch, batch_size,
                 val_batch_size, lr, weight_decay, max_patience, min_delta, verbose, log_every):
            """ Training loop; returns the best state (on the CPU). """
            opt = optimizer_cls(self.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=10)
            # Snapshots of the best weights are kept on the CPU, not on the GPU.
            best_val, best_state, bad_epochs = float("inf"), self._cpu_state(), 0

            for epoch in range(max_epoch):
                self.train()
                losses = []
                for ixs in self.get_batch_ixs(X_train, batch_size, permute=True):
                    xb, yb = X_train[ixs].to(dev), y_train[ixs].to(dev)
                    opt.zero_grad()
                    loss = criterion(self(xb), yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10)
                    opt.step()
                    losses.append(loss.item())
                scheduler.step()

                self.eval()
                with torch.no_grad():
                    val_loss = self._validation_loss(X_val, y_val, criterion, dev, val_batch_size)
                train_loss = float(np.mean(losses))

                history["epoch"].append(epoch)
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)

                if verbose and (epoch % log_every == 0 or epoch == max_epoch - 1):
                    print(f"epoch {epoch:4d} | train {train_loss:.4f} | val {val_loss:.4f}")

                if not (np.isfinite(val_loss) and np.isfinite(train_loss)):
                    if not np.isfinite(best_val):
                        raise FloatingPointError(
                            f"Loss is not finite at epoch {epoch} (train {train_loss}, val {val_loss}) and no "
                            f"earlier epoch was valid; the network did not train. Try a lower lr, other "
                            f"layers, or check the data for NaN/inf.")
                    warnings.warn(f"Loss became non-finite at epoch {epoch}; stopping and keeping the best "
                                  f"state (val loss {best_val:.4f}).")
                    break

                if val_loss < best_val - min_delta:
                    best_val, best_state, bad_epochs = val_loss, self._cpu_state(), 0
                else:
                    bad_epochs += 1
                if bad_epochs > max_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} (best val loss {best_val:.4f})")
                    break
            return best_state

        def _cpu_state(self):
            return {k: v.detach().to("cpu", copy=True) for k, v in self.state_dict().items()}

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


        def predict(self, q, batch_size=1024):
            '''
            Makes predictions on the network's current device, in batches.

            Parameters
            ----------
            q : torch.Tensor
                Input data for prediction (on the CPU).
            batch_size : int, optional
                Rows per forward pass (default 1024).

            Returns
            -------
            predictions : np.ndarray
                Predicted outputs as a numpy array.
            '''

            self.eval()
            device = self.device
            predictions = []
            with torch.no_grad():
                for start in range(0, q.shape[0], batch_size):
                    predictions.append(self(q[start:start + batch_size].to(device)).cpu().numpy())
            return np.concatenate(predictions)


        def compute_shap_values(self, q, predict_fn, forced=False, explainer_type="kernelexplainer", silent=False):
            '''
            Computes SHAP values for model interpretability, on the CPU.

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
                # The network stays on the CPU (no automatic move to a GPU)
                self.to('cpu')

                # Prepare input
                if isinstance(q, pd.DataFrame):
                    xi = torch.from_numpy(q.values).double()
                elif isinstance(q, np.ndarray):
                    xi = torch.from_numpy(q).double()
                elif isinstance(q, torch.Tensor):
                    xi = q.double().cpu()
                else:
                    raise ValueError(f"Input q must be a numpy array or a torch tensor. The type of q is: {type(q)}")

                if hasattr(self, 'explainer') == False or forced == True:
                    self.explainer = shap.DeepExplainer(self, xi)
                shap_values = self.explainer(xi)       # DeepExplainer takes no 'silent' argument
            elif explainer_type == "kernelexplainer":
                if hasattr(self, 'explainer') == False or forced == True:
                    explainer = shap.KernelExplainer(predict_fn, q)
                    self.explainer = explainer
                shap_values = self.explainer(q, silent=silent)
            else:
                raise ValueError(f"Unknown explainer_type '{explainer_type}'")
            return shap_values
