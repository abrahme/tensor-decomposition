from typing import Dict, List, Tuple

import numpy as np
from numpy.linalg import svd
from scipy.linalg import khatri_rao, inv


def CPALS(X: np.array, rank: int, max_iter: int) -> Dict:
    """

    :param rank: number of components to use
    :param X: the N- order tensor to be decomposed
    :param max_iter: maximum number of iterations to run the algorithm
    :return: Dictionary of factor matrices, objective function value, and iterations used
    """

    X_dim = X.shape
    N = len(X_dim)
    A = [np.std(X) * np.random.randn(dim, rank) + np.mean(X) for dim in X_dim]  ### initialize the factors
    lam = np.ones((rank,))  ### initialize norms
    ### normalize columns of A and store as lambda
    for k, factor_matrix in enumerate(A):
        factor_matrix_norm = np.linalg.norm(factor_matrix, axis=0)
        A[k] = factor_matrix / factor_matrix_norm
        lam *= factor_matrix_norm

    loss = np.linalg.norm(CPReconstruction(A, lam) - X)
    iterations = 0  ### initialize iterations

    while (iterations < max_iter):
        lam = np.ones((rank,))
        for i in range(N):
            A_i_norm, A_i = optimize_mode(X, A, i)
            lam *= A_i_norm  ## update the weights
            A[i] = A_i  ### update the factor matrix

        iterations += 1  ### update iteration
        loss = np.linalg.norm(CPReconstruction(A, lam) - X)
        print(f"iteration: {iterations}, loss: {loss}")
    return {"loss": loss, "lambda": lam, "iterations": iterations, "factor_matrices": A}


def compute_hadamard(factor_matrices: List[np.array]) -> np.array:
    """
    computes sequential hadamard product of list of factor matrices of size R x R
    :param factor_matrices: list of square matrices
    :return: one matrix
    """
    product = 1
    for matrix in factor_matrices:
        product *= matrix
    return product


def compute_khatri_rao(factor_matrices: List[np.array]) -> np.array:
    """
    :param factor_matrices: list of factor matrices all having columnsize R
    :return: khatri rao product of all in the list. Final matrix size is i1*i2*i3...*in X R
    """
    index = 0
    product = factor_matrices[index]
    while index < len(factor_matrices) - 1:
        product = khatri_rao(product, factor_matrices[index + 1])
        index += 1
    return product


def optimize_mode(X: np.array, factor_matrices: List[np.array], mode: int) -> Tuple[np.array, np.array]:
    """

    :param X: original tensor to reconstruct
    :param factor_matrices: list of factor matrices
    :param mode: mode to optimize
    :return: optimized factor matrix
    """
    V_list = [np.matmul(factor.T, factor) for j, factor in enumerate(factor_matrices) if
              j != mode]  #### take the square of each mat so its R by R

    V = compute_hadamard(V_list)

    inv_v = inv(V)  ### get inverse of V

    #### get khatri rao products
    khatri_list = [item[1] for item in reversed(list(enumerate(factor_matrices))) if item[0] != mode]
    khatri_product = compute_khatri_rao(khatri_list)

    X_n = mode_n_unfolding(X, mode)  ### move ith mode to first and ith mode unfolding
    A_n = np.matmul(np.matmul(X_n, khatri_product), inv_v)  ### update A

    #### normalize A and record lambda

    A_n_norm = np.linalg.norm(A_n, axis=0)
    A_n /= A_n_norm

    return A_n_norm, A_n


def mode_n_unfolding(X: np.array, mode: int) -> np.array:
    """

    :param X: N way tensor with size i1 x i2 x i3 x ... iN
    :param mode: which mode to unfold along
    :return: a matrix with size i_mode X (product of all the modes except target mode)
    """

    X_dim = X.shape
    mode_size = X_dim[mode]
    X_reorder = np.moveaxis(X, mode, 0)  ### make this axis the first order
    return np.reshape(X_reorder, (mode_size, -1))


def CPReconstruction(factor_matrices: List[np.array], lam: np.array) -> np.array:
    """
    :param factor_matrices: cp decomposition of X
    :param lam: normalizing constants
    :return: tensor of reconstruction
    """

    r = len(lam)
    N = len(factor_matrices)
    X_hat = 0

    for k in range(r):
        rank_factors = [factor[:, k] for factor in factor_matrices]
        outer_prod = lam[k] * rank_factors[0]
        for j in range(1, N):
            outer_prod = np.einsum("...j,...k->...jk", outer_prod, rank_factors[j])
        X_hat += outer_prod
    return X_hat


if __name__ == "__main__":
    from scipy.io import loadmat
    X_mat = loadmat("../tensorized_weights/cp_fc_layer.mat")
    X = X_mat["A"]
    rank = 50
    iterations = 100
    CPALS(X, rank, iterations)
