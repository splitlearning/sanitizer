#
# ron_gauss.py
# Author: Thee Chanyaswad
#
# Version 1.0
# 	- Initial implementation.
#
# Version 1.1
# 	- Bug fixes.
# 	- Add reconstruction option.
# 	- Minor stylistic update.
#
# Version 1.2
# 	- Add mean adjustment option for unsupervised and supervised algs.
#
# Version 1.3 (submitted by mlopatka 17-07-2019)
#  	- Enforce python black formatting guidelines.
#   - Add test coverage
#
# Version 1.4 (tchanyaswad 27-7-2019)
#   - Change variable names and improve readability.
#   - Add docstring
#


import numpy as np
import scipy

class RONGauss:
    """RON-Gauss: Synthesizing Data with Differential Privacy

    This module implements RON-Gauss, which is a method for non-interactive differentially-private data release based on 
    random orthornormal (RON) projection and Gaussian generative model. RON-Gauss leverages the Diaconis-Freedman-Meckes (DFM) effect,
    which states that most random projections of high-dimensional data approaches Gaussian.

    The main method of the `RONGauss` class is `_generate_dpdata()`. It takes in the original data as inputs, and, depending
    on the algorithm chosen, returns the differentially private data.
    
    Parameters
    ----------
    algorithm : string {'supervised', 'unsupervised', 'gmm'}
        supervised : 
            implements RON-Gauss for supervised learning, especially for regression. This algorithm
            requires both the feature data and the target label as inputs. In addition, it requires
            `max_y` to be specified.
        unsupervised :
            implements RON-Gauss for unsupervised learning, so only the feature data are required.
            This mode will return `None` for `dp_y`.
        gmm :
            implements RON-Gauss for classification. This algorithm requires both the feature data
            and the target label as inputs. The target label has to be categorical.
    epsilon_mean : float (default 1.0)
        The privacy budget for computing the mean. The sum of `epsilon_mean` and `epsilon_cov` is the total
        privacy budget spent.
    epsilon_cov : float (default 1.0)
        The privacy budget for computing the covariance. The sum of `epsilon_mean` and `epsilon_cov` is the total
        privacy budget spent.
    
    References
    ----------
    *Thee Chanyaswad, Changchang Liu, and Prateek Mittal. "RON-Gauss: Enhancing Utility in Non-Interactive Private
    Data Release," Proceedings on Privacy Enhancing Technologies (PETS), vol. 2019, no. 1, 2018.*
    (https://content.sciendo.com/view/journals/popets/2019/1/article-p26.xml)
    
    Examples
    --------
    >>> import ron_gauss
    >>> import numpy as np
    >>> X = np.random.normal(size=(1000,100))
    >>> dim = 10
    >>> # try unsupervised
    >>> rongauss_unsup = ron_gauss.RONGauss(algorithm='unsupervised')
    >>> dp_x, _ = rongauss_unsup.generate_dpdata(X, dim)
    >>> # try supervised
    >>> y = np.random.uniform(low=0.0, high=1.0, size=1000)
    >>> rongauss_sup = ron_gauss.RONGauss(algorithm='supervised')
    >>> dp_x, dp_y = rongauss_sup.generate_dpdata(X, dim, y, max_y = 1.0)
    >>> # try gmm
    >>> y = np.random.choice([0,1], size=1000)
    >>> rongauss_gmm = ron_gauss.RONGauss(algorithm='gmm')
    >>> dp_x, dp_y = rongauss_gmm.generate_dpdata(X, dim, y)
    """
    
    def __init__(self, algorithm="supervised", epsilon_mean=1.0, epsilon_cov=1.0):
        self.algorithm = algorithm
        self.epsilon_mean = epsilon_mean
        self.epsilon_cov = epsilon_cov

    def generate_dpdata(
        self,
        X,
        dimension,
        y=None,
        max_y=None,
        n_samples=None,
        reconstruct=True,
        centering=False,
        prng_seed=None,
    ):
        """Generate differentially-private dataset using RON-Gauss
        Parameters
        ----------
        X : numpy.ndarray, shape = [N_samples, M_features]
            Feature data.
        dimension : int < M_features
            The dimension for the data to be reduced to.
        y : numpy.ndarray, shape = [n_samples] (default None)
            Target values.
            unsupervised : this parameter is not used.
            supervised : required.
            gmm : required and the values should be categorical.
        n_samples : int (default None)
            The number of samples to be synthesized. If None is passed, the returned number of samples will
            be equal to N_samples of X.
        max_y : float (default None)
            The maximum absolute value that the target label can take. For example, if y is [0,1], then
            max_y = 1. If y is [-2,1], then max_y = 2. This is required and used by the supervised
            algorithm only.
        reconstruct : bool (default True)
            An option to reconstrut the projected synthesized data back to the original space. If True, the
            returned data will have the same dimension as X. If False, the returned data will have the dimension
            specified by the parameter `dimension`.
        centering : bool (default False)
            An option to automatically center the synthesized data. If False, the mean will be the
            differentially-private mean derived from X. This parameter is always False for 'gmm'.
        prng_seed : int (default None)
            This is to specify the seed used in randomized algorithms used.
        
        Returns
        -------
        x_dp : numpy.ndarray, shape = [n_samples, M_features] or [n_samples, dimensions]
            The differentially-private feature data. If `reconstruct` is True, this will be [n_samples, M_features].
            If `reconstruct` is False, it will be [n_samples, dimensions].
        y_dp : numpy.ndarray, shape = [n_samples]
            For `unsupervised`, this will be None.
            For `supervised` and `gmm`, this will be the differentially private target label.
        """
        (n, m) = X.shape
        if n_samples is None:
            n_samples = n

        if self.algorithm == "unsupervised":
            x_dp = self._unsupervised_rongauss(X, dimension, n_samples, reconstruct, centering, prng_seed)
            y_dp = None

        elif self.algorithm == "supervised":
            x_dp, y_dp = self._supervised_rongauss(X, dimension, y, n_samples, max_y, reconstruct, centering, prng_seed)

        elif self.algorithm == "gmm":
            x_dp, y_dp = self._gmm_rongauss(X,dimension, y, n_samples, reconstruct, prng_seed)
        
        return (x_dp, y_dp)
    
    def _unsupervised_rongauss(
        self,
        X,
        dimension,
        n_samples,
        reconstruct,
        centering,
        prng_seed,
    ):
        """Generate differentially-private dataset using the unsupervised RON-Gauss
        Parameters
        ----------
        X : numpy.ndarray, shape = [N_samples, M_features]
            Feature data.
        dimension : int < M_features
            The dimension for the data to be reduced to.
        n_samples : int (default None)
            The number of samples to be synthesized. If None is passed, the returned number of samples will
            be equal to N_samples of X.
        reconstruct : bool (default True)
            An option to reconstrut the projected synthesized data back to the original space. If True, the
            returned data will have the same dimension as X. If False, the returned data will have the dimension
            specified by the parameter `dimension`.
        centering : bool (default False)
            An option to automatically center the synthesized data. If False, the mean will be the
            differentially-private mean derived from X. This parameter is always False for 'gmm'.
        prng_seed : int (default None)
            This is to specify the seed used in randomized algorithms used.
        
        Returns
        -------
        x_dp : numpy.ndarray, shape = [n_samples, M_features] or [n_samples, dimensions]
            The differentially-private feature data. If `reconstruct` is True, this will be [n_samples, M_features].
            If `reconstruct` is False, it will be [n_samples, dimensions].
        """
        prng = np.random.RandomState(prng_seed)
        (x_bar, mu_dp) = self._data_preprocessing(X, self.epsilon_mean, prng)
        (x_tilda, proj_matrix) = self._apply_ron_projection(x_bar, dimension, prng)
        (n, p) = x_tilda.shape
        noise_var = (2.0 * np.sqrt(p)) / (n * self.epsilon_cov)
        cov_matrix = np.inner(x_tilda.T, x_tilda.T) / n
        laplace_noise = prng.laplace(scale=noise_var, size=(p, p))
        cov_dp = cov_matrix + laplace_noise
        synth_data = prng.multivariate_normal(np.zeros(p), cov_dp, n_samples)
        x_dp = synth_data
        if reconstruct:
            x_dp = self._reconstruction(x_dp, proj_matrix)
        else:
            #project the mean down to the lower dimention
            mu_dp = np.inner(mu_dp, proj_matrix)
        self._mu_dp = mu_dp

        if not centering:
            x_dp = x_dp + mu_dp
        return x_dp

    def _supervised_rongauss(
        self,
        X,
        dimension,
        y,
        n_samples,
        max_y,
        reconstruct,
        centering,
        prng_seed,
    ):  
        """Generate differentially-private dataset using the supervised RON-Gauss
        Parameters
        ----------
        X : numpy.ndarray, shape = [N_samples, M_features]
            Feature data.
        dimension : int < M_features
            The dimension for the data to be reduced to.
        y : numpy.ndarray, shape = [n_samples] (default None)
            Target values.
        n_samples : int (default None)
            The number of samples to be synthesized. If None is passed, the returned number of samples will
            be equal to N_samples of X.
        max_y : float
            The maximum absolute value that the target label can take. For example, if y is [0,1], then
            max_y = 1. If y is [-2,1], then max_y = 2.
        reconstruct : bool (default True)
            An option to reconstrut the projected synthesized data back to the original space. If True, the
            returned data will have the same dimension as X. If False, the returned data will have the dimension
            specified by the parameter `dimension`.
        centering : bool (default False)
            An option to automatically center the synthesized data. If False, the mean will be the
            differentially-private mean derived from X. This parameter is always False for 'gmm'.
        prng_seed : int (default None)
            This is to specify the seed used in randomized algorithms used.
        
        Returns
        -------
        x_dp : numpy.ndarray, shape = [n_samples, M_features] or [n_samples, dimensions]
            Differentially-private feature data. If `reconstruct` is True, this will be [n_samples, M_features].
            If `reconstruct` is False, it will be [n_samples, dimensions].
        y_dp : numpy.ndarray, shape = [n_samples]
            Differentially private target label.
        """

        prng = np.random.RandomState(prng_seed)
        (x_bar, mu_dp) = self._data_preprocessing(X, self.epsilon_mean, prng)
        (x_tilda, proj_matrix) = self._apply_ron_projection(x_bar, dimension, prng)

        (n, p) = x_tilda.shape
        noise_var = (2.0 * np.sqrt(p) + 4.0 * np.sqrt(p) * max_y + max_y ** 2) / (
            n * self.epsilon_cov
        )
        y_reshaped = y.reshape(len(y), 1)
        augmented_mat = np.hstack((x_tilda, y_reshaped))
        cov_matrix = np.inner(augmented_mat.T, augmented_mat.T) / n
        laplace_noise = prng.laplace(scale=noise_var, size=cov_matrix.shape)
        cov_dp = cov_matrix + laplace_noise

        synth_data = prng.multivariate_normal(np.zeros(p + 1), cov_dp, n_samples)
        x_dp = synth_data[:, 0:-1]
        y_dp = synth_data[:, -1]
        if reconstruct:
            x_dp = self._reconstruction(x_dp, proj_matrix)
        else:
            #project the mean down to the lower dimention
            mu_dp = np.inner(mu_dp, proj_matrix)
        self._mu_dp = mu_dp

        if not centering:
            x_dp = x_dp + mu_dp
        
        return (x_dp, y_dp)

    def _gmm_rongauss(
        self,
        X,
        dimension,
        y,
        n_samples,
        reconstruct,
        prng_seed,
    ):
        """Generate differentially-private dataset using the GMM RON-Gauss
        Parameters
        ----------
        X : numpy.ndarray, shape = [N_samples, M_features]
            Feature data.
        dimension : int < M_features
            The dimension for the data to be reduced to.
        y : numpy.ndarray, shape = [n_samples] (default None)
            Target values, which hould be categorical.
        n_samples : int (default None)
            The number of samples to be synthesized. If None is passed, the returned number of samples will
            be equal to N_samples of X.
        reconstruct : bool (default True)
            An option to reconstrut the projected synthesized data back to the original space. If True, the
            returned data will have the same dimension as X. If False, the returned data will have the dimension
            specified by the parameter `dimension`.
        centering : bool (default False)
            An option to automatically center the synthesized data. If False, the mean will be the
            differentially-private mean derived from X. This parameter is always False for 'gmm'.
        prng_seed : int (default None)
            This is to specify the seed used in randomized algorithms used.
        
        Returns
        -------
        x_dp : numpy.ndarray, shape = [n_samples, M_features] or [n_samples, dimensions]
            Differentially-private feature data. If `reconstruct` is True, this will be [n_samples, M_features].
            If `reconstruct` is False, it will be [n_samples, dimensions].
        y_dp : numpy.ndarray, shape = [n_samples]
            Differentially private target label.
        """
        prng = np.random.RandomState(prng_seed)
        syn_x = None
        syn_y = np.array([])
        for label in np.unique(y):
            idx = np.where(y == label)
            x_class = X[idx]
            (x_bar, mu_dp) = self._data_preprocessing(x_class, self.epsilon_mean, prng)
            (x_tilda, proj_matrix) = self._apply_ron_projection(x_bar, dimension, prng)

            (n, p) = x_tilda.shape
            noise_var = (2.0 * np.sqrt(p)) / (n * self.epsilon_cov)
            mu_dp_tilda = np.inner(mu_dp, proj_matrix)
            cov_matrix = np.inner(x_tilda.T, x_tilda.T) / n
            laplace_noise = prng.laplace(scale=noise_var, size=(p, p))
            cov_dp = cov_matrix + laplace_noise
            synth_data = prng.multivariate_normal(mu_dp_tilda, cov_dp, n_samples)

            if reconstruct:
                synth_data = self._reconstruction(synth_data, proj_matrix)
            if syn_x is None:
                syn_x = synth_data
            else:
                syn_x = np.vstack((syn_x, synth_data))
            
            syn_y = np.append(syn_y, label * np.ones(n_samples))
        
        return syn_x, syn_y

    def learn_gmm_rongauss(self,
        X,
        dimension,
        y,
        prng_seed
    ):
        self.cov_dp = []
        self.mu_dp_tilda = []
        self.proj_matrices = []
        self.prng = np.random.RandomState(prng_seed)
        self.labels = np.unique(y)
        for label in self.labels:
            idx = np.where(y == label)
            x_class = X[idx]
            (x_bar, mu_dp) = self._data_preprocessing(x_class, self.epsilon_mean, self.prng)
            (x_tilda, proj_matrix) = self._apply_ron_projection(x_bar, dimension, self.prng)

            (n, p) = x_tilda.shape
            noise_var = (2.0 * np.sqrt(p)) / (n * self.epsilon_cov)
            self.mu_dp_tilda.append(np.inner(mu_dp, proj_matrix))
            cov_matrix = np.inner(x_tilda.T, x_tilda.T) / n
            laplace_noise = self.prng.laplace(scale=noise_var, size=(p, p))
            self.cov_dp.append(cov_matrix + laplace_noise)
            self.proj_matrices.append(proj_matrix)
        return

    def sample_dpgmm(self, n_samples):
        syn_x = None
        for idx, label in enumerate(self.labels):
            #print(self.mu_dp_tilda, self.cov_dp, n_samples)
            synth_data = self.prng.multivariate_normal(self.mu_dp_tilda[idx], self.cov_dp[idx], int(n_samples[idx]))
            synth_data = self._reconstruction(synth_data, self.proj_matrices[idx])
            if syn_x is None:
                syn_x = synth_data
                syn_y = label.astype(int) * np.ones(n_samples, dtype=np.int32)
            else:
                syn_x = np.vstack((syn_x, synth_data))
                syn_y = np.append(syn_y, label.astype(int) * np.ones(n_samples, dtype=np.int32))
        shuffler = np.random.permutation(len(syn_x))
        syn_x, syn_y = syn_x[shuffler], syn_y[shuffler]
        return syn_x, syn_y

    @staticmethod
    def _data_preprocessing(X, epsilon_mean, prng=None):
        """
        This is the DATA_PREPROCESSING algorithm based on Algo. 1 in the paper.
        Parameters
        ----------
        X : numpy.ndarray, shape = [N_samples, M_features]
            Feature data.
        epsilon_mean : float
            The privacy budget used for computing the mean.
        prng_seed : int (default None)
            This is to specify the seed used in randomized algorithms used.
        
        Returns
        -------
        x_bar : numpy.ndarray, shape = [N_samples, M_features]
            The pre-processed data which are pre-normalized, centered, and re-normalized.
        mean_dp : numpy.ndarray, shape = [M_features]
            The differentially-private mean used in the centering.
        """
        if prng is None:
            prng = np.random.RandomState()
        (n, m) = X.shape
        # pre-normalize
        x_norm = np.nan_to_num(RONGauss._normalize_sample_wise(X))
        # derive dp-mean
        mu = np.mean(x_norm, axis=0)
        noise_var_mu = np.sqrt(m) / (n * epsilon_mean)
        laplace_noise = prng.laplace(scale=noise_var_mu, size=m)
        mean_dp = mu + laplace_noise
        # centering
        x_bar = x_norm - mean_dp
        # re-normalize
        x_bar = RONGauss._normalize_sample_wise(x_bar)
        return x_bar, mean_dp

    def _apply_ron_projection(self, x_bar, dimension, prng=None):
        """
        This is the RON_PROJECTION algorithm based on Algo. 2 in the paper.
        Parameters
        ----------
        x_bar : numpy.ndarray, shape = [N_samples, M_features]
            Feature data.
        dimension : int < M_features
            The dimension for the data to be reduced to.
        prng_seed : int (default None)
            This is to specify the seed used in randomized algorithms used.
        
        Returns
        -------
        x_tilda : numpy.ndarray, shape = [N_samples, dimension]
            The dimension-reduced data.
        ron_matrix : numpy.ndarray, shape = [dimension, M_features]
            The RON projection matrix used for dimensionality reduction.
        """
        (n, m) = x_bar.shape
        full_projection_matrix = self._generate_ron_matrix(m, prng)
        ron_matrix = full_projection_matrix[0:dimension]  # take the rows
        x_tilda = np.inner(x_bar, ron_matrix)
        return x_tilda, ron_matrix

    def _reconstruction(self, x_projected, ron_matrix):
        """
        The function used to project the dimension-reduced data back to the original space.
        Parameters
        ----------
        x_projected : numpy.ndarray, shape = [N_samples, P_dimension]
            The dimension-reduced data.
        ron_matrix : numpy.ndarray, shpae = [P_dimension, M_features]
            The RON projection matrix used to produce the dimension-reduced data.
        
        Returns
        -------
        x_reconstructed : numpy.ndarray, shape = [N_samples, M_features]
            The reconstructed data.
        """
        x_reconstructed = np.inner(x_projected, ron_matrix.T)
        return x_reconstructed

    def _generate_ron_matrix(self, m, prng=None):
        """
        Generate a RON projection matrix using QR factorization.
        Parameters
        ----------
        m : int
            The dimension of the projection matrix.
        prng_seed : int (default None)
            This is to specify the seed used in randomized algorithms used.
        
        Returns
        -------
        ron_matrix : numpy.ndarray, shape = [m, m]
            The RON projection matrix.
        """
        if prng is None:
            prng = np.random.RandomState()
        # generate random matrix
        random_matrix = prng.uniform(size=(m, m))
        # QR factorization
        q_matrix, r_matrix = np.linalg.qr(random_matrix)
        ron_matrix = q_matrix
        return ron_matrix
    
    @staticmethod
    def _normalize_sample_wise(x):
        """
        Sample-wise normalization
        Parameters
        ----------
        x : numpy.ndarray, shape = [N_samples, M_features]
            Feature data.
        
        Returns
        -------
        x_normalized : numpy.ndarray, shape = [N_samples, M_features]
            The sample-wise normalized data
        """
        (n,p) = x.shape
        sample_norms = np.linalg.norm(x, axis=1) #norms of each sample
        x_normalized = x/(np.outer(sample_norms,np.ones(p)))
        return x_normalized