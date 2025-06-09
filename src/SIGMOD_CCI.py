"""
spatial_utils.py

Utility functions for spatial transcriptomics analysis, including:
- Computing group‐wise means
- Differential ligand/receptor detection
- Kernel and autocorrelation functions
"""

from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from tqdm.auto import tqdm
import itertools
import re

import scipy as sp

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

def get_means_group(
    adata: AnnData,
    group_key: str,
    layer: str = None,
    gene_symbols: List[str] = None
) -> pd.DataFrame:
    """
    Compute mean expression per group for each gene.

    Parameters
    ----------
    adata
        Annotated data matrix.
    group_key
        Column in `adata.obs` to group by.
    layer
        If provided, use `adata.layers[layer]` instead of `adata.X`.
    gene_symbols
        If provided, restrict to this list of genes.

    Returns
    -------
    DataFrame of shape (n_genes, n_groups) containing group means.
    """
    # select data matrix
    get_matrix = (lambda obj: obj.layers[layer]) if layer else (lambda obj: obj.X)

    # restrict genes if requested
    genes = list(adata.var_names)
    if gene_symbols is not None:
        genes = [g for g in gene_symbols if g in adata.var_names]
    if not genes:
        raise ValueError("No matching genes found in adata.var_names.")

    # prepare output
    groups = sorted(adata.obs[group_key].unique())
    out = pd.DataFrame(
        0.0,
        index=genes,
        columns=groups,
        dtype=np.float64
    )

    # compute means
    for grp in groups:
        idx = adata.obs.index[adata.obs[group_key] == grp]
        mat = get_matrix(adata[idx, genes])
        # ensure dense
        arr = mat.A if hasattr(mat, "A") else np.asarray(mat)
        out[grp] = arr.mean(axis=0).ravel()

    return out


def get_diff_lr(
    scdata: AnnData,
    group_key:str,
    ligands_receptors: List[str]
) -> pd.DataFrame:
    """
    Identify differentially expressed ligands/receptors per cluster
    and compute their mean expression per cluster.

    Parameters
    ----------
    scdata
        AnnData with `scdata.obs['cluster']` defined.
    ligands_receptors
        List of gene names to test.

    Returns
    -------
    DataFrame of shape (n_genes, n_clusters) with means for significant genes,
    zero elsewhere.
    """
    # subset to LR genes
    adata = scdata[:, ligands_receptors].copy()

    # differential expression
    sc.tl.rank_genes_groups(
        adata, groupby=group_key, method="t-test", use_raw=False
    )
    diff_df = sc.get.rank_genes_groups_df(
        adata,
        group=None,
        log2fc_min=0,
        pval_cutoff=1.0
    )

    # compute group means
    means = get_means_group(adata, group_key=group_key)

    # mask non-significant entries
    valid = set(zip(diff_df['names'], diff_df['group']))
    mask = pd.DataFrame(
        [
            [(gene, grp) in valid for grp in means.columns]
            for gene in means.index
        ],
        index=means.index,
        columns=means.columns
    )
    return means.where(mask, 0.0)


def get_lr_effect(
    eta: np.ndarray,
    lr_niche: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute ligand–receptor effect as matrix product.

    Parameters
    ----------
    eta
        Array of shape (n_cells, n_niches).
    lr_niche
        DataFrame of shape (n_niches, n_LR_pairs).

    Returns
    -------
    DataFrame of shape (n_cells, n_LR_pairs).
    """
    result = eta @ lr_niche
    return result


def generate_matched_lr(
    lr_effect: pd.DataFrame,
    diff_lr: pd.DataFrame
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Match left/right partners in lr_effect to differential LR sets.

    Parameters
    ----------
    lr_effect
        DataFrame with columns named "L_R" for LR pairs.
    diff_lr
        DataFrame with differential expression per gene.

    Returns
    -------
    effect_mask
        Boolean array for valid LR pairs.
    l_df, r_df
        DataFrames of matched left/right expression values.
    """
    cols = np.array(lr_effect.columns)
    lefts = np.array([c.split('_')[0] for c in cols])
    rights = np.array([c.split('_')[1] for c in cols])

    valid = np.isin(lefts, diff_lr.columns) & np.isin(rights, diff_lr.columns)
    if not valid.any():
        empty = pd.DataFrame(index=lr_effect.index)
        return valid, empty, empty

    matched = cols[valid]
    l_vals = diff_lr[lefts[valid]].values
    r_vals = diff_lr[rights[valid]].values

    l_df = pd.DataFrame(l_vals, index=lr_effect.index, columns=matched)
    r_df = pd.DataFrame(r_vals, index=lr_effect.index, columns=matched)
    return valid, l_df, r_df


def mask_lr_effect(
    lr_effect: pd.DataFrame,
    receptor_diff: pd.DataFrame
) -> pd.DataFrame:
    """
    Mask ligand–receptor effects by receptor differential mask.
    """
    return lr_effect * receptor_diff


def SE_kernel(
    X: np.ndarray,
    length_scale: float
) -> np.ndarray:
    """
    Squared‐Exponential (RBF) kernel matrix.

    k(x_i, x_j) = exp(-||x_i - x_j||^2 / (2 * l^2))
    """
    X = np.asarray(X)
    sq = np.sum(X**2, axis=1)
    D2 = -2 * np.dot(X, X.T) + sq[:, None] + sq[None, :]
    D2 = np.clip(D2, 1e-12, None)
    return np.exp(-D2 / (2 * length_scale**2))


def spatial_auto_corr(
    tau: np.ndarray,
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Compute the weighted correlation between x and y
    using weight‐matrix tau, returning a value in [–1, 1].
    """
    # ensure arrays
    tau = np.asarray(tau, float)
    x   = np.asarray(x,   float)
    y   = np.asarray(y,   float)

    # subtract global means
    dx = x - x.mean()
    dy = y - y.mean()

    # numerator: sum_{i,j} tau_ij dx_i dy_j
    num = np.sum(tau * np.outer(dx, dy))

    # denominator: sqrt( sum tau_ij dx_i dx_j  *  sum tau_ij dy_i dy_j )
    var_x = np.sum(tau * np.outer(dx, dx))
    var_y = np.sum(tau * np.outer(dy, dy))
    denom = np.sqrt(var_x * var_y)

    # avoid divide‐by‐zero
    if denom <= 0:
        return 0.0

    return num / denom

def compute_spatial_corr_matrix(
    theta: pd.DataFrame,
    tau: np.ndarray,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Compute pairwise spatial autocorrelation between all cell‐type vectors in `theta`.

    Parameters
    ----------
    theta : pd.DataFrame
        shape = (n_spots, n_celltypes).  Each column is the expression/score vector for a cell type.
    tau : np.ndarray
        shape = (n_spots, n_spots).  Spatial weight matrix.
    autocorr_func : callable
        Function taking (tau, x, y) and returning a scalar correlation.
    show_progress : bool
        Whether to display a progress bar.

    Returns
    -------
    pd.DataFrame
        shape = (n_celltypes, n_celltypes).  Symmetric matrix of correlations.
    """
    celltypes = list(theta.columns)
    # initialize empty DataFrame
    spatial_corr = pd.DataFrame(
        np.nan,
        index=celltypes,
        columns=celltypes,
        dtype=float
    )

    iterator = tqdm(celltypes, desc="Computing spatial correlations") if show_progress else celltypes
    for i, ct1 in enumerate(iterator):
        x = theta[ct1].values
        # only compute upper triangle and mirror
        for ct2 in celltypes[i:]:
            y = theta[ct2].values
            val = spatial_auto_corr(tau=tau, x=x, y=y)
            spatial_corr.at[ct1, ct2] = val
            spatial_corr.at[ct2, ct1] = val

    return spatial_corr


def compute_interaction_dict(
    spatial_corr: pd.DataFrame,
    l_exp: pd.DataFrame,
    masked_r_exp: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Compute cell–cell interaction matrices for each ligand–receptor pair.

    For each ligand column `lig` in `l_exp` and corresponding receptor column `rec` 
    in `masked_r_exp`, this function computes

        Interaction[i,j] = spatial_corr[i,j] * l_exp.loc[i, lig] * masked_r_exp.loc[j, rec]

    and returns a dict mapping ligand names to the resulting DataFrame (n_celltypes × n_celltypes).

    Parameters
    ----------
    spatial_corr : pd.DataFrame
        Square DataFrame (n×n) of spatial correlations between cell types.
        Index and columns should be identical lists of cell‐type names.
    l_exp : pd.DataFrame
        DataFrame (n×m) of ligand expressions. Rows indexed by the same cell types as `spatial_corr`.
    masked_r_exp : pd.DataFrame
        DataFrame (n×m) of receptor expressions. Must have the same shape and row‐index as `l_exp`.
        Columns in one‐to‐one correspondence with `l_exp.columns`.

    Returns
    -------
    interaction_dict : dict
        Keys are ligand names (from `l_exp.columns`), values are DataFrames (n×n)
        of interaction scores between all cell‐type pairs.
    """
    # Ensure row‐alignment
    celltypes = spatial_corr.index
    l_exp_aligned = l_exp.reindex(index=celltypes, columns=l_exp.columns).fillna(0)
    r_exp_aligned = masked_r_exp.reindex(index=celltypes, columns=masked_r_exp.columns).fillna(0)

    interaction_dict = {}
    for lig, rec in zip(l_exp_aligned.columns, r_exp_aligned.columns):
        # extract series
        lig_s = l_exp_aligned[lig].values
        rec_s = r_exp_aligned[rec].values

        # outer product: ligand_i * receptor_j
        expr_outer = pd.DataFrame(
            np.outer(lig_s, rec_s),
            index=celltypes,
            columns=celltypes
        )

        # weight by spatial correlation
        interaction_matrix = spatial_corr * expr_outer

        # store by ligand name (or use f"{lig}_{rec}" if desired)
        interaction_dict[lig] = interaction_matrix

    return interaction_dict


##### For module module interaction
def compute_correlations(theta_logit, eta, cell_types, spatialnet,cutoff):

    results = {}
    for type1, type2 in itertools.product(cell_types, cell_types):
        # Filter relevant cell pairs
        subset = spatialnet[
            spatialnet['Cell1'].isin(theta_logit[type1].index) &
            spatialnet['Cell2'].isin(theta_logit[type2].index)
        ]
        
        if subset.empty:
            results[f"{type1} {type2}"] = None
            continue

        # Get topic distributions and compute correlations
        theta1 = theta_logit[type1].loc[subset['Cell1']].reset_index(drop=True)
        theta2 = theta_logit[type2].loc[subset['Cell2']].reset_index(drop=True)
        combined = pd.concat([theta1, theta2], axis=1)
        combined['cell'] = subset['Cell1'].values
        summed = combined.groupby('cell').sum()
        
        # Calculate correlation matrix
        cor_matrix = summed.corr(method='spearman')
        cor_matrix = cor_matrix.where(cor_matrix >= cutoff, 0)
        cor_matrix = cor_matrix.iloc[:theta1.shape[1], theta1.shape[1]:]
        
        # Apply eta weighting
        eta_vals = np.maximum(eta[type1].loc[type2].values, 0)
        cor_matrix *= eta_vals[:, np.newaxis]
        
        results[f"{type1} {type2}"] = cor_matrix

    return results

def construct_total_interactions(results, cell_types):

    # Collect valid matrices for each cell type
    big_mat_list = [
        pd.concat([results[f"{ct1} {ct2}"] for ct2 in cell_types 
                  if results[f"{ct1} {ct2}"] is not None], axis=1)
        for ct1 in cell_types if any(results[f"{ct1} {ct2}"] is not None 
                                    for ct2 in cell_types)
    ]
    
    # Combine matrices and clean names
    big_mat = pd.concat(big_mat_list, axis=0)
    big_mat.columns = [re.sub(r'\.\d+', '', col) for col in big_mat.columns]
    big_mat.index = [re.sub(r'\.\d+', '', idx) for idx in big_mat.index]
    
    return big_mat


##### For Gene Gene interaction
def build_sender_df(adata, theta_logit_dict, spatial_net_df, celltype1, celltype2, genes):

    # Filter spatial network for relevant cell pairs
    subset = spatial_net_df[
        spatial_net_df['Cell1'].isin(theta_logit_dict[celltype1].index) &
        spatial_net_df['Cell2'].isin(theta_logit_dict[celltype2].index)
    ].copy()
    # Extract gene expression data and aggregate by Cell1
    gene_data = adata[:, genes].to_df()
    result = gene_data.loc[subset['Cell2'], genes].set_axis(
        (f"{celltype2}_{col}" for col in genes), axis=1
    ).assign(Cell1=subset['Cell1'].values).groupby('Cell1').sum()

    return result

def multi_linear_regression(df, response_genes, predict_genes, p_val_cutoff=0.05):

    X = df[predict_genes]
    results = [
        np.insert(
            (lasso := Lasso(positive=True, alpha=0.1,fit_intercept=True)).fit(X, df[g]).coef_,
            0,
            lasso.intercept_
        )
        for g in response_genes
    ]
    
    return pd.DataFrame(results, columns=['const'] + predict_genes, index=response_genes)

def cal_g2g(celltype_sender, 
            states_sender, 
            celltype_receiver, 
            states_receiver,
            adata, 
            topic_words_dict, 
            theta_logit_dict, 
            spatial_net_df,
            ligands,
            frex_ = False,
            cutoff_r=0.5, 
            cutoff_s=0.8, 
            p_val_cutoff=0.05):

    # Get receiver target genes
    receiver_genes_df = topic_words_dict[celltype_receiver]
    if frex_:
        receiver_genes_df = frex(receiver_genes_df.T).T
        receiver_genes_df = pd.DataFrame(receiver_genes_df,
                                         columns = topic_words_dict[celltype_receiver].columns,
                                         index =topic_words_dict[celltype_receiver].index)
    receiver_genes = receiver_genes_df[receiver_genes_df[states_receiver] > cutoff_r].index
    receiver_data = adata[adata.obs.CellType == celltype_receiver, receiver_genes].to_df()
    receiver_data.columns = [f"{celltype_receiver}_{g}" for g in receiver_data.columns]
    
    # Initialize gene-to-state mapping
    r_state = f"{states_receiver}_receiver"
    gene2state = {r_state: list(receiver_data.columns)}

    # Process sender genes and combine data
    for sender, state in zip(celltype_sender, states_sender):
        sender_genes_df = topic_words_dict[sender]
        if frex_:
            sender_genes_df = frex(sender_genes_df.T).T
            sender_genes_df = pd.DataFrame(sender_genes_df,
                                           columns = topic_words_dict[sender].columns,
                                           index =topic_words_dict[sender].index)
              
        sender_genes = sender_genes_df[sender_genes_df[state] > cutoff_s].index
        sender_genes = list(set(sender_genes) & set(ligands))
        
        sender_data = build_sender_df(
            adata, theta_logit_dict, spatial_net_df, celltype_receiver, sender, sender_genes
        )
        receiver_data[sender_data.columns] = sender_data.reindex(receiver_data.index, fill_value=0)
        gene2state[state] = list(sender_data.columns)

    # Perform regression
    predict_genes = list(set(itertools.chain.from_iterable([genes for state, genes in gene2state.items() 
                                                            if state != r_state])))
    res_df = multi_linear_regression(receiver_data, gene2state[r_state], predict_genes, p_val_cutoff)

    return res_df, gene2state



def ecdf(arr):
    """Calculate the ECDF values for all elements in a 1D array."""
    return sp.stats.rankdata(arr, method="max") / arr.size

def frex(beta, w=0.5):
    """Calculate FREX (FRequency and EXclusivity) words
    A primarily internal function for calculating FREX words.
    Exclusivity is calculated by column-normalizing the beta matrix (thus representing the conditional probability of seeing
    the topic given the word).  Then the empirical CDF of the word is computed within the topic.  Thus words with
    high values are those where most of the mass for that word is assigned to the given topic.

    @param logbeta a K by V matrix containing the log probabilities of seeing word v conditional on topic k
    @param w a value between 0 and 1 indicating the proportion of the weight assigned to frequency

    """
    beta = np.log(beta)
    log_exclusivity = beta - sp.special.logsumexp(beta, axis=0)
    exclusivity_ecdf = np.apply_along_axis(ecdf, 1, log_exclusivity)
    freq_ecdf = np.apply_along_axis(ecdf, 1, beta)
    out = 1.0 / (w / exclusivity_ecdf + (1 - w) / freq_ecdf)
    out = pd.DataFrame(out,
                       columns =beta.columns,
                       index = beta.index)
    return out