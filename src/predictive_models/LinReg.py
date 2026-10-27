'''Linear Regression Model for Predictive Modelling'''

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import datetime
from .PredictiveModel import PredictiveModel
from .utils import get_scaler

class LinReg(PredictiveModel):
    '''
    Linear Regression Model for Predictive Modelling
    
    Attributes
    ----------
    model : LinearRegression
        The linear regression model instance.

    Q : VariableSet
        The variable set used for the model.

    QoI_names : list
        List of Quantity of Interest names. '''

    def __init__(self, Q, QoI_names):
        '''
        LinRegModel Constructor

        Parameters
        ----------
        Q : VariableSet
            The variable set for the model.
        QoI_names : list
            List of Quantity of Interest names.
        '''
        
        self.Q = Q
        self.QoI_names = QoI_names
        self.model = LinearRegression()


    def train_and_validate(self, X_train, y_train, X_val, y_val):
        '''
        Trains the Linear Regression model and evaluates it on training and validation data.

        Parameters
        ----------
        X_train : array-like
            Training feature data.
        y_train : array-like
            Training target data.
        X_val : array-like
            Validation feature data.
        y_val : array-like
            Validation target data.

        Returns
        -------
        tr_loss : float
            Mean squared error on the training data.
        vl_loss : float
            Mean squared error on the validation data.
        '''

        X_train, X_val = pd.DataFrame(X_train, columns=self.Q.variable_names()), pd.DataFrame(X_val, columns=self.Q.variable_names())
        y_train, y_val = pd.DataFrame(y_train, columns=self.QoI_names), pd.DataFrame(y_val, columns=self.QoI_names)

        self.model.fit(X_train, y_train)
        y_pred_tr = self.model.predict(X_train)
        tr_loss = mean_squared_error(y_train, y_pred_tr)
        y_pred_vl = self.model.predict(X_val)
        vl_loss = mean_squared_error(y_val, y_pred_vl)
        return tr_loss, vl_loss

    def train(self, q_train, y_train, k_fold=None):
        self.scaler = get_scaler("identity")

        xi = self.get_scaled(q_train)
        yt = self.get_scaled(y_train)

        print(f'----- Training started for \'LinRegModel\' model -----')
        
        if k_fold is not None:
            train_losses, val_losses = self.cross_validation(xi, yt, k_fold)
        else:
            # if there is no crossvalidation, then split 80% data for training and 20% for validation
            xi_train, xi_val, yt_train, yt_val = self.train_test_split(xi, yt, train_test_ratio=0.8, random_seed=1997, split_type='shuffle')
            train_losses, val_losses = self.train_and_validate(xi_train, yt_train, xi_val, yt_val)
        print(f"Average train loss: {np.mean(train_losses):.14f}, Average valid loss: {np.mean(val_losses):.14f}")      
        print(f'----- Training ended for LinRegModel model -----')

    # def plot_sobol_sensitivity(self, y_train, max_index=1, param_name=None):
    #     if hasattr(self, 'max_index') == False:
    #         self.max_index = max_index
    #     if self.max_index < max_index:
    #         self.get_sobol_sensitivity(max_index)
    #     fig = utils.plot_sobol_sensitivity(self.sobol_index_df, y_train, param_name)
    #     return fig

    def predict(self, X):
        '''
        Makes predictions using the trained Linear Regression model.

        Parameters
        ----------
        X : array-like
            Input feature data for prediction.

        Returns
        -------
        predictions : array-like
            Predicted target values.
        '''

        if type(X) != pd.DataFrame:
            X = pd.DataFrame(X, columns=self.Q.variable_names())

        X = self.get_scaled(X)
        X = pd.DataFrame(X, columns=self.Q.variable_names())
        y_t = self.model.predict(X)
        y = self.get_orig(y_t)

        return y

    def get_orig(self, scaled):
        original = self.scaler.inverse_transform(scaled)
        return original

    def get_scaled(self, original):
        scaled = self.scaler.transform(original)
        return scaled
    
    def compute_shap_values(self, predict_fn, q, forced=False, explainer_type="kernelexplainer", silent=False):
        '''
        Computes SHAP values for model interpretability.
        
        Parameters
        ----------
        predict_fn : function
            The prediction function of the model.
        q : array-like
            Input data for SHAP value computation.
        forced : bool, optional
            If True, forces re-computation of the SHAP explainer. Default is False.
        explainer_type : str, optional
            Type of SHAP explainer to use. Default is "kernelexplainer".
        silent : bool, optional
            If True, suppresses output during SHAP value computation. Default is False.
        Returns
        -------
        shap_values : array-like
            Computed SHAP values.
        '''

        if explainer_type == "kernelexplainer":
            if hasattr(self, 'explainer') == False or forced == True:
                explainer = shap.KernelExplainer(predict_fn, q)
                self.explainer = explainer
        shap_values = self.explainer(q, silent=silent)
        return shap_values

    def to_jsonld(self, model_id=None, building_did=None, building_ual=None, name=None, doc_type=None, description=None):

        mls_spec = {
            "@id": f"urn:implementation:{model_id}",
            "@type": "mls:Implementation",

            "mls:implements": {
                "@id": "https://www.wikidata.org/entity/Q208042",
                "@type": "mls:Algorithm",
                "rdfs:label": "Linear Regression"
            },

            "mls:hasHyperParameter": []
        }

        mls_inputs = [
            {
                "@type": "mls:Feature",

                "rdfs:label": feat_name,

                "mls:hasQuality": {
                    "@type": "mls:FeatureCharacteristic",

                    "distributionType": dist.get_dist_type(),
                    "distributionParameters": dist.get_dist_params()
                }
            }
            for feat_name, dist in self.Q.get_variables().items()
        ]

        mls_outputs = [{"@type": "mls:Feature", "rdfs:label": lbl} for lbl in self.QoI_names]

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
                "mls:hasInput": mls_inputs,
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
                "mls:hasInput": mls_inputs,
            }

        return jsonld