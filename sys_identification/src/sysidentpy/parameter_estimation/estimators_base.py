"""Base class for Basis Function."""

from warnings import warn
from abc import ABCMeta, abstractmethod

import numpy as np

from ..narmax_base import InformationMatrix


class BaseEstimator(metaclass=ABCMeta):
    """Base class for Model Structure Selection."""

    @abstractmethod
    def __init__(self, unbiased: bool = False, uiter: int = 30):
        self.unbiased = unbiased
        self.uiter = uiter

    @abstractmethod
    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Abstract method."""

    def _check_linear_dependence_rows(self, psi):
        if np.linalg.matrix_rank(psi) != psi.shape[1]:
            warn(
                "Psi matrix might have linearly dependent rows."
                "Be careful and check your data",
                stacklevel=2,
            )

    def _initial_values(self, psi: np.ndarray):
        n_theta = psi.shape[1]
        n = len(psi)
        theta = np.zeros([n_theta, n])
        xi = np.zeros([n, 1])
        return n_theta, n, theta, xi

    def unbiased_estimator(
        self, psi, y, theta, elag, max_lag, estimator, basis_function, uiter=20
    ):
        """Estimate the model parameters using Extended Least Squares method.

        Parameters
        ----------
        psi : ndarray of floats
            The information matrix of the model.
        y : ndarray of floats
            The output data.
        theta : ndarray of floats
            The estimated biased parameters of the model.
        elag : int
            The maximum lag of the residues regressors.
        max_lag : int
            Maximum lag among all the regressors.
        estimator : Estimators
            The parameter estimation method.
        basis_function : BasisFunction
            The basis function to represent the function F in the model space.
        uiter : int
            The number of iterations to be used in the Extended Least Squares method.

        Returns
        -------
        theta : ndarray of floats
            The estimated unbiased parameters of the model.

        References
        ----------
        - Manuscript: Sorenson, H. W. (1970). Least-squares estimation:
           from Gauss to Kalman. IEEE spectrum, 7(7), 63-68.
           http://pzs.dstu.dp.ua/DataMining/mls/bibl/Gauss2Kalman.pdf
        - Book (Portuguese): Aguirre, L. A. (2007). Introdução a identificação
           de sistemas: técnicas lineares e não-lineares aplicadas a sistemas
           reais. Editora da UFMG. 3a edição.
        - Manuscript: Markovsky, I., & Van Huffel, S. (2007).
           Overview of total least-squares methods.
           Signal processing, 87(10), 2283-2302.
            https://eprints.soton.ac.uk/263855/1/tls_overview.pdf
        - Wikipedia entry on Least Squares
           https://en.wikipedia.org/wiki/Least_squares

        """
        e = y - np.dot(psi, theta)
        im = InformationMatrix(ylag=elag)
        for _ in range(uiter):
            e = np.concatenate([np.zeros([max_lag, 1]), e], axis=0)

            lagged_data = im.build_output_matrix(None, e)

            e_regressors = basis_function.fit(
                lagged_data, max_lag, ylag=elag, predefined_regressors=None
            )

            psi_extended = np.concatenate([psi, e_regressors], axis=1)
            unbiased_theta = estimator.optimize(psi_extended, y)
            e = y - np.dot(psi_extended, unbiased_theta.reshape(-1, 1))

        return unbiased_theta[0 : theta.shape[0], 0].reshape(-1, 1)

    def _validate_params(self, attributes):
        """Validate input params."""
        # Define expected types and value ranges for each parameter
        param_specs = {
            "lam": (float, (0, 1)),
            "weight": (float, (0, 1)),
            "offset_covariance": (float, (0, 1)),
            "mu": (float, (0, None)),
            "eps": (float, (0, None)),
            "gama": (float, (0, None)),
            "uiter": (int, (0, None)),
            "delta": (float, (0, None)),
            "alpha": (float, (0, None)),
            "unbiased": (bool, None),
            "solver": (str, None),
        }
        valid_solvers = ["svd", "classic"]
        for attribute, value in attributes.items():
            if attribute not in param_specs:
                raise ValueError(f"Unexpected parameter: {attribute}")

            expected_type, value_range = param_specs[attribute]

            if not isinstance(value, expected_type):
                raise ValueError(
                    f"{attribute} must be of type {expected_type.__name__}. "
                    f"Got {type(value).__name__} instead."
                )

            if value_range:
                min_val, max_val = value_range
                if min_val is not None and value < min_val:
                    raise ValueError(f"{attribute} must be >= {min_val}. Got {value}.")
                if max_val is not None and value > max_val:
                    raise ValueError(f"{attribute} must be <= {max_val}. Got {value}.")

            if attribute == "solver" and value not in valid_solvers:
                raise ValueError(
                    f"{attribute} must be one of {valid_solvers}. Got {value}."
                )
