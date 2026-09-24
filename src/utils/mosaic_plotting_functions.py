import mosaictools as mosaic

def plot_variable_space_exploration(model, mode, n_samples=1000):
    """ Plot the variable space exploration for a Mosaic model. Returns three figures:
            1. Space division plot
            2. Reference eigenvectors plot
            3. Reference correlation matrix plot
        
        Parameters
        ----------
        model : MosaicModel
            The Mosaic model instance to be visualized.
        mode : int
            The index of mode for plotting.
        n_samples : int, optional
            The number of samples to use for the space division plot (default is 1000). """
    
    assert model.method == "Mosaic", "The model method must be 'Mosaic' to use this plotting function."

    fig1 = mosaic.plot_space_division(model.model.model, mode, n_samples)
    fig2 = mosaic.plot_reference_eigenvectors(model.model.model, mode)
    fig3 = mosaic.plot_reference_correlation_matrix(model.model.model, mode)
    return fig1, fig2, fig3

def get_number_of_subdomains(model):
    """ Get the number of subdomains for a Mosaic model. 
    
        Parameters
        ----------
        model : MosaicModel
            The Mosaic model instance.

        Returns
        -------
        list_of_subdomains : list of int
            The number of subdomains for the Mosaic model. """
    
    assert model.method == "Mosaic", "The model method must be 'Mosaic' to use this function."

    list_of_subdomains = model.model.model.get_number_of_subdomains()
    return list_of_subdomains