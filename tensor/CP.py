from typing import Dict, List

import numpy as np
from scipy.linalg import khatri_rao, inv
from numpy.linalg import svd

def CPALS(X: np.array, rank: int, max_iter: int) -> Dict:
    """

    :param rank: number of components to use
    :param X: the N- order tensor to be decomposed
    :param max_iter: maximum number of iterations to run the algorithm
    :return: Dictionary of factor matrices, objective function value, and iterations used
    """

    X_dim = X.shape
    N = len(X_dim)
    A = [np.std(X)*np.random.randn(dim, rank) + np.mean(X) for dim in X_dim]  ### initialize the factors
    lam = np.ones((rank,))  ### initialize norms
    ### normalize columns of A and store as lambda
    for k, factor_matrix in enumerate(A):
        factor_matrix_norm = np.linalg.norm(factor_matrix, axis=0)
        A[k] /= factor_matrix_norm
        lam *= factor_matrix_norm

    loss = CPLoss(X, A, lam)  ### initialize objective function loss
    iterations = 0  ### initialize iterations

    while (iterations < max_iter) :
        for i in range(N):

            V_list = [np.matmul(factor.T, factor) for j, factor in enumerate(A) if
                      j != i]  #### take the square of each mat so its R by R
            V = 1
            #### hadamard product of all the matrices
            for factor_matrix in V_list:
                V = V * factor_matrix

            inv_v = inv(V)  ### get inverse of V

            #### get khatri rao products
            khatri_list = [item[1] for item in reversed(list(enumerate(A))) if item[0] != i]
            while len(khatri_list) > 1:
                khatri_list[1] = khatri_rao(khatri_list[0], khatri_list[1])
                khatri_list = khatri_list[1:]

            X_n = np.reshape(np.moveaxis(X, i, 0), (X_dim[i], -1))  ### move ith mode to first and ith mode unfolding
            A[i] = np.matmul(np.matmul(X_n, khatri_list[-1]), inv_v)  ### update A

            #### normalize A[i] and record lambda

            A_i_norm = np.linalg.norm(A[i], axis=0)
            lam *= A_i_norm
            A[i] /= A_i_norm

        iterations += 1  ### update iteration
        loss = CPLoss(X, A, lam)
        print(f"iteration: {iterations}, loss: {loss}")
    return {"loss": loss, "lambda": lam, "iterations": iterations, "factor_matrices": A}


def CPLoss(X: np.array, factor_matrices: List[np.array], lam: np.array) -> float:
    """

    :param X: original tensor
    :param factor_matrices: cp decomposition of X
    :param lam: normalizing constants
    :return: float of frobenius norm of difference between X and its reconstruction
    """

    r = len(lam)
    N = len(factor_matrices)
    X_hat = 0

    for k in range(r):
        rank_factors = [factor[:, k] for factor in factor_matrices]
        outer_prod = lam[k] * rank_factors[0]
        for j in range(N - 1):
            outer_prod = np.einsum("...j,...k->...jk", outer_prod, rank_factors[j + 1])
        X_hat += outer_prod
    return np.linalg.norm(X_hat - X)


if __name__ == "__main__":
    # from scipy.io import loadmat
    # X_mat = loadmat("../tensorized_weights/cp_fc_layer.mat")
    # X = X_mat["A"]


    rank = 50
    iterations = 100
    CPALS(X,rank,iterations)