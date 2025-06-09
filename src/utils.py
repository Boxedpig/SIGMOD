

import numpy as np
import scipy.sparse as sp
import scipy
import torch
from sklearn.neighbors import NearestNeighbors

def compute_knn_gaussian_adjacency_matrix(pos, k=20, sigma=1.0):
    X = pos.to_numpy()
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = distances[:, 1:]  # Exclude self (distance zero at first position)
    indices = indices[:, 1:]      # Exclude self
    weights = np.exp(-distances**2 / (2 * sigma**2))
    N = X.shape[0]
    row_indices = np.repeat(np.arange(N), k)
    col_indices = indices.flatten()
    data = weights.flatten()
    W = sp.csr_matrix((data, (row_indices, col_indices)), shape=(N, N))
    W = W.minimum(W.transpose())  # Mutual kNN
    return W

def compute_laplacian(W):
    degrees = W.sum(axis=1).A1  # Compute degree matrix
    D = sp.diags(degrees)
    L = D - W  # Compute Laplacian
    return L

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def compute_sparse_laplacian_for_multiple_samples(pos, sample_column, k=20, sigma=1.0):
    # Split positions by sample
    unique_samples = pos[sample_column].unique()
    
    # Initialize an empty list to store adjacency matrices
    all_adjacency_matrices = []
    
    # Loop over each sample and compute the adjacency matrix
    for sample in unique_samples:
        sample_positions = pos[pos[sample_column] == sample].drop(columns=[sample_column])
        W_sample = compute_knn_gaussian_adjacency_matrix(sample_positions, k, sigma)
        print(W_sample.shape)
        all_adjacency_matrices.append(W_sample)
    # Combine all adjacency matrices into a global adjacency matrix
    W_combined = sp.block_diag(all_adjacency_matrices)  # Block matrix concatenation
    
    # Compute the Laplacian for the combined adjacency matrix
    L_combined = compute_laplacian(W_combined)
    
    # Convert to PyTorch sparse tensor if needed
    L_torch = sparse_mx_to_torch_sparse_tensor(L_combined)
    
    return L_torch


def ecdf(arr):
    return scipy.stats.rankdata(arr, method="max") / arr.size

def frex(beta, w=0.5):
    beta = np.log(beta)
    log_exclusivity = beta - scipy.special.logsumexp(beta, axis=0)
    exclusivity_ecdf = np.apply_along_axis(ecdf, 1, log_exclusivity)
    freq_ecdf = np.apply_along_axis(ecdf, 1, beta)
    out = 1.0 / (w / exclusivity_ecdf + (1 - w) / freq_ecdf)
    return out