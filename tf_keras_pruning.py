import numpy as np
from scipy.stats import rankdata

class KerasModelPurner(object):
    
    def __init__(self, sparsity, pruning_type):
        
        """
        Args
        -------
        s: Sparsity level in range(0,1).  
        pruning_type: choose b/w ('weight_pruning','neuron_pruning')
        
        """
                
        self.sparsity = sparsity
        self.pruning_type = pruning_type
        
        
    def transform(self, file):
    
        """
        Purnes All weights (except last\output layer) with given purning type for a given model(.h5) file

        Args
        -------
        file: keras model file (.h5).
        s: Sparsity level in range(0,1).  
        pruning_type: choose b/w ('weight_pruning','neuron_pruning')

        Returns
        -------
        modified model file (.h5) with given sparsity
        """
    
        for layer in list(file['model_weights'].keys())[:-1]:
            if not layer.startswith('dropout'):
                # get only main weights, ignore bias
                W = file['model_weights'][layer][layer]['kernel:0']
                # choose the purning type
                if pruning_type == 'weight_pruning':
                    W[...] = weight_pruning(W, s = s)
                elif pruning_type == 'neuron_pruning':
                    W[...] = neuron_pruning(W, s = s)
                # assert the changes
                assert(W == file['model_weights'][layer][layer]['kernel:0'])        
        return file
    
    
    def weight_pruning(self, w, s):
    
        """
        Ranks & Purnes the least absoulte (s%) values in given weight matrix

        Args
        -------
        w: weight matrix
        s: Sparsity level in range(0,1).  

        Returns
        -------
        modified weight matrix (w) with given sparsity
        """

        w = np.array(w)
        # ranks from lowest to highest
        ranks = rankdata(np.abs(w),method='dense') 
        # find the theshold for given sparsity
        threshold = np.ceil(ranks.max() * s).astype(int)
        # create mask to multuiply
        mask = ranks.reshape(w.shape)
        # set the elements in the mask
        mask[mask <= threshold] = 0
        mask[mask > threshold] = 1
        # multiply the originalweights with mask to get sparse weights
        return w*mask

    def neuron_pruning(self, w, s):

        """
        Ranks & Purnes the least L2 norm (s%) columns for a given weight matrix

        Args
        -------
        w: weight matrix
        s: Sparsity level in range(0,1).  

        Returns
        -------
        modified weight matrix (w) with given sparsity
        """
        
        w = np.array(w)
        # calculate the L2 norm
        norm = np.sqrt(np.sum(w*w, axis=0))
        # ranks from lowest to highest
        ranks = rankdata(norm,method='dense') 
        # find the threshold for given sparsity
        threshold = np.ceil(ranks.max() * s).astype(int)
        # find the indices of columns below threshold
        zero_col_indices = np.where(ranks <= threshold)[0]
        # set the entire columns to zero
        w[:, zero_col_indices] = 0

        return w