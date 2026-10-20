from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split, KFold
import pandas as pd


class ProbabilisticModel(ABC):
    @abstractmethod
    def train(self, q_train, y_train, k_fold=None, **params):
        pass

    def train_test_split(self, X_data, y_data, train_test_ratio=0.8, random_seed=1997, split_type='shuffle'):
        '''
        Splits the dataset into training and testing sets.

        Parameters
        ----------
        X_data : pd.DataFrame
            Input features
        y_data : pd.DataFrame
            Target values
        train_test_ratio : float
            Ratio of training data to total data (default is 0.8)
        random_seed : int
            Random seed for reproducibility (default is 1997)
        split_type : str
            Type of split ('shuffle', 'no_shuffle', 'reverse') (default is 'shuffle')

        Returns
        -------
        X_train, X_test, y_train, y_test : pd.DataFrame
            Split training and testing datasets
        '''

        self.samples = X_data
        self.QoI = y_data        
        if split_type == 'shuffle':
            X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=train_test_ratio, random_state=random_seed, shuffle=True)
        elif split_type == 'no_shuffle':
            X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=train_test_ratio, random_state=random_seed, shuffle=False)
        elif split_type == 'reverse':
            X_test, X_train, y_test, y_train = train_test_split(X_data, y_data, test_size=train_test_ratio, random_state=random_seed, shuffle=False)
        
        return X_train, X_test, y_train, y_test

    def cross_validation(self, xi, yt, k_fold, **params):
        '''
        Performs k-fold cross-validation for model training.

        Parameters:
        -----------
        xi : np.ndarray
            Input features
        yt : np.ndarray
            Target values
        k_fold : int
            Number of folds for cross-validation
        **params : dict
            Additional training parameters

        Returns:
        -------
        train_losses : list
            List of training losses for each fold
        val_losses : list
            List of validation losses for each fold
        '''

        train_losses, val_losses = [], []
        kf = KFold(n_splits=k_fold, shuffle=True)
        for fold, (train_indices, val_indices) in enumerate(kf.split(xi)):
            print(f"Fold {fold + 1}/{k_fold}")
            xi_train, xi_val = xi[train_indices], xi[val_indices]
            yt_train, yt_val = yt[train_indices], yt[val_indices]
            tr_loss, vl_loss = self.train_and_validate(xi_train, yt_train, xi_val, yt_val, **params)
            train_losses.append(tr_loss), val_losses.append(vl_loss)
        return train_losses, val_losses

    def get_mean_and_var(self, n_sample=1000):
        q = self.Q.sample(n_sample)
        q_df = pd.DataFrame(q, columns=self.Q.variable_names())
        y_predict = self.predict(q_df)
        mean, var = y_predict.mean(axis=0), y_predict.var(axis=0)
        return mean, var
    
    def __getstate__(self):
        '''
        Prepares the instance state for pickling by removing large training data attributes.

        Returns:
        -------
        state : dict
            The instance state dictionary without large training data attributes.
        '''

        # Create a copy of the instance dictionary
        state = self.__dict__.copy()
        
        # Remove large training data attributes before pickling
        for attr in ['X_train', 'y_train', 'X_test', 'y_test']:
            if attr in state:
                del state[attr]
        
        return state

    def __setstate__(self, state):
        '''
        Restores the instance state from the pickled state dictionary.

        Parameters:
        -----------
        state : dict
            The instance state dictionary to restore.
        '''

        # Restore the instance state
        self.__dict__.update(state)
