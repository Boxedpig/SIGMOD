
"""
Spatial Variable Ligand Receptor Analysis (SVLR)

This module provides a collection of functions for analyzing spatial transcriptomics data,
focusing on ligand-receptor interactions, spatial neighborhood analysis, and niche creation.
"""

from typing import List, Dict, Tuple, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import minmax_scale
from tqdm.autonotebook import tqdm
import scanpy as sc


def get_LR_from_CellChaDB(
    lr_df: pd.DataFrame,
    complex_df: pd.DataFrame,
    gene_names_in_data: pd.Index
):
    """
    Filters and processes a ligand-receptor list against available gene names.
    
    Handles simple L-R pairs and complexes from CellChatDB data.

    Args:
        lr_df: DataFrame containing ligand-receptor pairs.
        complex_df: DataFrame defining protein complexes.
        gene_names_in_data: An index of gene names present in the expression data.

    Returns:
        A tuple containing two arrays: processed ligands and processed receptors.
    """
    lr_df = lr_df.sort_values('annotation')
    ligand = lr_df.ligand.values
    receptor = lr_df.receptor.values
    
    for i in range(len(ligand)):
        for n in [ligand, receptor]:
            l = n[i]
            if l in complex_df.index:
                bool_list = pd.Series(complex_df.loc[l].dropna().values).isin(var_names)
                # Check if all elements are True
                all_true = all(bool_list)
                # Set all elements based on the check
                bool_list = [all_true] * len(bool_list)
                n[i] = complex_df.loc[l].dropna().values[bool_list]
            else:
                n[i] = pd.Series(l).values[pd.Series(l).isin(var_names)]

    return ligand,receptor

def get_KNN_graph(
    obs_df: pd.DataFrame,
    radius: float,
    region_col: str = 'Region'
):
    """
    Constructs a spatial graph based on K-Nearest Neighbors within a given radius.

    Args:
        obs_df: DataFrame with spatial coordinates ('x', 'y') and optional region annotations.
        radius: The radius to define neighbors.
        region_col: The column name in obs_df containing region annotations.

    Returns:
        A DataFrame representing the spatial graph with columns for Cell1, Cell2, Distance,
        and optionally region information.
    """

    coords = obs_df[['x', 'y']]
    coords.index = obs_df.index

    # Find neighbors within the specified radius
    nbrs = NearestNeighbors(radius=radius).fit(coords)
    distances, indices = nbrs.radius_neighbors(coords, return_distance=True)

    # Format into a DataFrame
    knn_list = [
        pd.DataFrame({'Cell1': it, 'Cell2': indices[it][j], 'Distance': distances[it][j]})
        for it in range(len(indices)) for j in range(len(indices[it]))
    ]
    
    spatial_net = pd.concat(knn_list, ignore_index=True)
    spatial_net = spatial_net[spatial_net['Distance'] > 0] # Exclude self-loops

    # Map integer indices back to cell IDs
    id_to_cell = dict(enumerate(coords.index))
    spatial_net['Cell1'] = spatial_net['Cell1'].map(id_to_cell)
    spatial_net['Cell2'] = spatial_net['Cell2'].map(id_to_cell)

    # Add region information if available
    if region_col in obs_df.columns:
        cell_to_region = obs_df[region_col].to_dict()
        spatial_net['Region1'] = spatial_net['Cell1'].map(cell_to_region)
        spatial_net['Region2'] = spatial_net['Cell2'].map(cell_to_region)
        # Create a combined region identifier for edges between different regions
        spatial_net['combined_region'] = spatial_net.apply(
            lambda row: '_'.join(sorted([row['Region1'], row['Region2']])), axis=1
        )
        
    return spatial_net


def find_n_hop_spots(
    adjacent_df: pd.DataFrame,
    border_spots: Set[str], 
    region:str,
    n:int):
    """
    Finds all spots within n hops from a starting set of spots in a target region.

    Args:
        adjacent_df: DataFrame representing the spatial graph.
        border_spots: A set of starting spot IDs.
        region: The region to search within.
        n: The number of hops (e.g., 1-hop for direct neighbors).

    Returns:
        A set of spot IDs including the start spots and their n-hop neighbors.
    """
    current_hop_spots = set(border_spots)
    all_spots = set(border_spots)
    
    for _ in range(n):
        # Find adjacencies involving current hop spots
        next_hop_adjacencies = adjacent_df[
            (adjacent_df['Cell1'].isin(current_hop_spots) & (adjacent_df['Region2'] == region)) |
            (adjacent_df['Cell2'].isin(current_hop_spots) & (adjacent_df['Region1'] == region))
        ]
        
        # Extract next hop spots (belonging to the same region)
        # next_hop_spots = set(next_hop_adjacencies[next_hop_adjacencies['Region1'] == region]['Cell2']).union(
        #     set(next_hop_adjacencies[next_hop_adjacencies['Region2'] == region]['Cell1'])
        # )
        next_hop_spots = set(next_hop_adjacencies['Cell2']).union(
            set(next_hop_adjacencies['Cell1'])
        )
        
        # Update the sets for the next iteration
        new_spots = next_hop_spots - all_spots
        current_hop_spots = new_spots
        all_spots.update(new_spots)
    
    return all_spots

def create_subspaces_based_on_anno(
    adjacent_df: pd.DataFrame,
    obs: pd.DataFrame,
    n:int, 
    border: bool = False):
    """
    Creates spatial subspaces based on annotated regions and their borders.

    Args:
        adjacent_df: DataFrame representing the spatial graph.
        obs: DataFrame with spatial coordinates and region annotations.
        n: The number of hops to expand from border regions.
        include_borders: Whether to create additional subspaces for border regions.

    Returns:
        A dictionary where keys are region names (or border names) and values are
        DataFrames of the spots in that subspace.
    """   

    coor = obs.loc[:,['x', 'y']]
    coor.index = obs.index
    
    subspaces = {}

    # Subspaces for individual regions
    regions = obs['Region'].unique()
    for region in regions:
        
        spots = obs[(obs['Region'] == region)].index
        
        subspaces[region] = obs.loc[spots,:]
    if border:
        # Subspaces for pairs of regions (border spots and n-hop neighbors)
        for region_combinations in adjacent_df['combined_region'].unique():

            regions = region_combinations.split('_')

            if len(set(regions)) == 1:  # Single region, skip
                continue

            border_spots_region1_1 = adjacent_df[(adjacent_df['Region1'] == regions[0]) & (adjacent_df['Region2'] ==regions[1])]['Cell1'].unique()
            border_spots_region1_2 = adjacent_df[(adjacent_df['Region2'] == regions[0]) & (adjacent_df['Region1'] ==regions[1])]['Cell2'].unique()
            border_spots_region1 = np.append(border_spots_region1_1,border_spots_region1_2)
            border_spots_region1 = np.unique(border_spots_region1)

            border_spots_region2_1 = adjacent_df[(adjacent_df['Region1'] == regions[0]) & (adjacent_df['Region2'] ==regions[1])]['Cell2'].unique()
            border_spots_region2_2 = adjacent_df[(adjacent_df['Region2'] == regions[0]) & (adjacent_df['Region1'] ==regions[1])]['Cell1'].unique()
            border_spots_region2 = np.append(border_spots_region2_1,border_spots_region2_2)
            border_spots_region2 = np.unique(border_spots_region2)

            # Find n-hop neighbors for border spots in each region
            n_hop_neighbors_region1 = find_n_hop_spots(adjacent_df, border_spots_region1, regions[0], n)
            n_hop_neighbors_region2 = find_n_hop_spots(adjacent_df, border_spots_region2, regions[1], n)

            # Combine border spots and their n-hop neighbors
            combined_spots_region1 = set(border_spots_region1).union(n_hop_neighbors_region1)
            combined_spots_region2 = set(border_spots_region2).union(n_hop_neighbors_region2)

            # Create subspaces for border and n-hop neighbors
            subspaces[region_combinations] = pd.concat([obs.loc[list(combined_spots_region1),:],
                                                       obs.loc[list(combined_spots_region2),:]])

    return subspaces

def create_subspaces(
    obs: pd.DataFrame, 
    n:int,
    x_range:Tuple[float, float], 
    y_range:Tuple[float, float], 
    x_step: float, 
    y_step: float, 
    mx: float, 
    my: float):
    """
    Partitions a 2D space into a grid of overlapping rectangular subspaces.

    This function creates a grid of `n x n` subspaces and filters a DataFrame of points
    to find which points fall into each subspace. The subspaces can have margins,
    allowing them to overlap.

    Args:
        obs: DataFrame containing the points to be partitioned. Must have
             'x' and 'y' columns for coordinates.
        n: The number of divisions along each axis (creates an n x n grid).
        x_range: A tuple (min_x, max_x) defining the total range of the x-axis.
        y_range: A tuple (min_y, max_y) defining the total range of the y-axis.
        x_step: The width of each subspace before adding margins.
        y_step: The height of each subspace before adding margins.
        mx: The margin to add to each side of a subspace along the x-axis.
        my: The margin to add to each side of a subspace along the y-axis.

    Returns:
        A dictionary where keys are the names of the subspaces (e.g., "Subspace_1_1")
        and values are DataFrames containing the points that fall within that subspace.
    """
    subspaces = {}
    for i in range(n):
        for j in range(n):
            # Calculate the bounds of each subspace
            x_start = x_range[0] + i * (x_step + mx) - mx
            x_end = x_start + x_step + mx
            y_start = y_range[0] + j * (y_step + my) - my
            y_end = y_start + y_step + my

            # Filter the positions that fall within the current subspace
            subspace = obs[(obs['x'] >= x_start) & (obs['x'] <= x_end) &
                           (obs['y'] >= y_start) & (obs['y'] <= y_end)]

            if not subspace.empty:
                subspaces[f"Subspace_{i+1}_{j+1}"] = subspace

    return subspaces

# --- L-R Interaction Scoring Functions ---

def calculate_LR_mean(
    All_L_R: pd.DataFrame, 
    distMat: pd.DataFrame):
    """
    Calculates the interaction strength and its z-score for a ligand-receptor pair.

    Args:
        All_L_R: DataFrame with two columns (ligand, receptor expression).
        distMat: A matrix where non-zero entries indicate neighboring spots.

    Returns:
        A tuple containing the interaction strength and its z-score.
    """
    
    L_R = All_L_R.copy()
    L_R = np.array(L_R)
    
    df = np.matmul(L_R[:, 0].reshape(-1,1), L_R[:, 1].reshape((-1, 1)).T)
    df = np.multiply(df, distMat)
    
    observed_value = np.sum(df)
    
    mean_1 = np.mean(L_R[:, 0])
    mean_2 = np.mean(L_R[:, 1])
    sd_1 = np.std(L_R[:, 0], ddof=1)
    sd_2 = np.std(L_R[:, 1], ddof=1)
    n = np.sum(distMat)
    
    interaction_variance = 1 / (n * mean_1**2 * mean_2**2) * (sd_1**2 * sd_2**2 + sd_1**2 * mean_2**2 + sd_2**2 * mean_1**2)
    
    expected_value = mean_1 * mean_2 * n
    
    interaction_strength = observed_value / expected_value
    interaction_strength_z = (interaction_strength - 1) / np.sqrt(interaction_variance)
    
    return interaction_strength, interaction_strength_z


def calculate_graph_density(
    All_L_R: pd.DataFrame,
    distMat: pd.DataFrame,
    subpos: pd.DataFrame,):
    """
    Calculates the interaction density within a spatial subspace.

    Args:
        All_L_R: DataFrame with ligand and receptor expression.
        distMat: The full distance matrix.
        subpos: The obs DataFrame for the subspace, used to subset the matrices.

    Returns:
        The calculated graph density.
    """

    L_R = All_L_R.loc[subpos.index,:].copy()
    L_R = np.array(L_R)
    
    n = L_R.shape[0]

    dist_mat = distMat.copy()
    dist_mat = dist_mat[subpos.cell_order,:][:,subpos.cell_order]
    
    num_interactions = np.mean(np.sum(dist_mat,axis = 1))
    
    df = np.matmul(L_R[:, 0].reshape(-1,1), L_R[:, 1].reshape((-1, 1)).T)
    df = np.multiply(df, dist_mat)
    
    return np.sum(df > 0) / (n * num_interactions)


def run_svlr(
    ligand: np.ndarray,
    receptor: np.ndarray,
    distMat: pd.DataFrame,
    subspaces: Dict[str, pd.DataFrame],
    stdata: sc.AnnData):
    """
    Runs the main SVLR analysis pipeline for all ligand-receptor pairs.

    Args:
        ligand: Array of ligand gene names (or lists of names for complexes).
        receptor: Array of receptor gene names.
        distMat: Matrix indicating spatial proximity.
        subspaces: Dictionary of spatial subspaces.
        stdata: AnnData object containing expression data.

    Returns:
        A tuple containing:
        - List of statistical results for each L-R pair.
        - List of L-R pair names.
        - List of ligand names.
        - List of receptor names.
    """    
    lr_res = []
    lr_pair = []
    ligands = []
    receptors = []
    
    mean_spots_in_subspaces = np.array([subpos.shape[0] for subpos in subspaces.values()]).mean()
    print("mean spots in each subspaces is:  %d"%mean_spots_in_subspaces)
    
    for i in tqdm(range(len(ligand))):

        if (len(ligand[i]) > 0) * (len(receptor[i]) > 0):

            meanL = stdata[:, ligand[i]].X.mean(axis=1).reshape((1,-1))
            meanR = stdata[:, receptor[i]].X.mean(axis=1).reshape((1,-1))
            
            L_R_df = pd.DataFrame(np.array(np.concatenate((meanL, meanR), axis=0).T), index=stdata.obs_names, columns=["_".join(ligand[i]),"_".join(receptor[i])])
            L_R_df = L_R_df.loc[stdata.obs_names]
    
            L_R_mean = calculate_LR_mean(L_R_df, distMat)  # 计算得到的配受体互作强度/理论上的配受体互作强度

            l_r_divide = [calculate_graph_density(L_R_df, distMat,subpos) for subpos in subspaces.values()] # 计算划分的每一个区域的graph density
            density_c = np.array(l_r_divide)
            density_c = density_c * mean_spots_in_subspaces
            density_p = chisquare(density_c)[1]
    
            lr_res.append(np.concatenate([density_c, [density_p], L_R_mean]))
            lr_pair.append("_".join(ligand[i]) + "_" +"_".join(receptor[i]))
            ligands.append("_".join(ligand[i]))
            receptors.append("_".join(receptor[i]))

    return lr_res,lr_pair,ligands,receptors


def plotLR(lr_pair, adata, pos):
    
    genes = lr_pair.split('_')
    n_genes = len(genes)
    # Extract ligand and receptor data
    l_r_data = adata[:, genes].to_df()
    
    # Merge with spatial coordinates
    l_r_data = l_r_data.assign(x=pos.iloc[:, 0], y=pos.iloc[:, 1])

    width = 8 * n_genes
    height = 5
    # Plotting
    fig, axes = plt.subplots(1, n_genes, figsize=(width, height))
    
    for i in range(n_genes):
        
        scatter = axes[i].scatter(l_r_data['x'], l_r_data['y'], c=l_r_data[genes[i]], cmap='Reds')
        fig.colorbar(scatter, ax=axes[i], orientation='vertical')
        axes[i].set_title(f'{genes[i]} Expression')

    plt.tight_layout()
    
    
# --- Niche Creation and Processing ---
def SE_kernel(
    X: np.ndarray, 
    l: float):
    """
    Computes a Squared Exponential (SE) kernel matrix from coordinates.
    Kernel: exp(-||xi - xj||^2 / (2 * l^2))

    Args:
        X: An array of coordinates (n_spots, n_dims).
        l: The length scale parameter (l) of the kernel.

    Returns:
        The computed (n_spots, n_spots) kernel matrix.
    """
    X = np.array(X)
    Xsq = np.sum(np.square(X), 1)
    R2 = -2. * np.dot(X, X.T) + (Xsq[:, np.newaxis] + Xsq[np.newaxis, :])
    R2 = np.clip(R2, 1e-12, np.inf)
    return np.exp(-R2 / (2 * l ** 2))
    

def Create_Niche(
    stdata: sc.AnnData,
    pos: pd.DataFrame,
    l_secrete: float,
    l_contact: float,
    lr_res: pd.DataFrame
    ):
    """
    Creates niche interaction profiles based on L-R expression and spatial kernels.

    Args:
        stdata: AnnData object with expression data.
        pos: DataFrame with spot coordinates ('x', 'y').
        l_secrete: Kernel length scale for secreted signals.
        l_contact: Kernel length scale for contact-based signals.
        lr_res: DataFrame of L-R pairs with an 'annotation' column.

    Returns:
        A tuple containing:
        - A list of DataFrames [secreted_interactions, contact_interactions].
        - Index of secreted L-R pairs.
        - Index of contact L-R pairs.
    """
    
    results = []
    
    X_secrete = SE_kernel(pos, l_secrete)
    X_contact = SE_kernel(pos, l_contact)

    lr_res_secrete = lr_res[lr_res.annotation == "Secreted Signaling"]
    lr_res_contact = lr_res[lr_res.annotation != "Secreted Signaling"]
    
    ligand_exp_secrete = []
    receptor_exp_secrete = []

    for i in tqdm(range(lr_res_secrete.shape[0])):

        ligand = lr_res_secrete.ligands[i].split("_")
        receptor = lr_res_secrete.receptors[i].split("_")

        l_exp = np.squeeze(np.array(stdata[:, ligand].X.mean(axis=1)))
        r_exp = np.squeeze(np.array(stdata[:, receptor].X.mean(axis=1)))

        ligand_exp_secrete.append(l_exp)
        receptor_exp_secrete.append(r_exp)

    if (len(ligand_exp_secrete)>0) * (len(ligand_exp_secrete) >0):
        ligand_exp_secrete = pd.DataFrame(ligand_exp_secrete).T
        receptor_exp_secrete = pd.DataFrame(receptor_exp_secrete).T
        
        ligand_exp_secrete = np.matmul(X_secrete,ligand_exp_secrete)
        LR_secrete = np.multiply(ligand_exp_secrete,receptor_exp_secrete)
        results.append(LR_secrete)


    ligand_exp_nonsecrete = []
    receptor_exp_nonsecrete = []    

    for i in tqdm(range(lr_res_contact.shape[0])):

        ligand = lr_res_contact.ligands[i].split("_")
        receptor = lr_res_contact.receptors[i].split("_")

        l_exp = np.squeeze(np.array(stdata[:, ligand].X.mean(axis=1)))
        r_exp = np.squeeze(np.array(stdata[:, receptor].X.mean(axis=1)))

        ligand_exp_nonsecrete.append(l_exp)
        receptor_exp_nonsecrete.append(r_exp)

    if (len(ligand_exp_nonsecrete)>0) * (len(receptor_exp_nonsecrete) >0):
        ligand_exp_nonsecrete = pd.DataFrame(ligand_exp_nonsecrete).T
        receptor_exp_nonsecrete = pd.DataFrame(receptor_exp_nonsecrete).T
        
        ligand_exp_nonsecrete = np.matmul(X_contact,ligand_exp_nonsecrete)
        LR_nonsecrete = np.multiply(ligand_exp_nonsecrete,receptor_exp_nonsecrete)
        results.append(LR_nonsecrete)
    
    return results,lr_res_secrete.index,lr_res_contact.index


# --- Post-processing Functions ---

def process_dataframe_dpgmm(
    df: pd.DataFrame,
    max_components: int = 3,
    random_state: int = 0) -> pd.DataFrame:
    """
    Normalize each column of df to [0,1] and keep only the highest-mean DPGMM component.

    Parameters
    ----------
    df
        Input data.
    max_components
        Maximum mixture components.
    random_state
        Seed for reproducibility.

    Returns
    -------
    DataFrame
        Same shape as df, but with all values outside the main component set to 0.
    """
    # normalize once
    df_norm = pd.DataFrame(
        minmax_scale(df, feature_range=(0, 1)),
        index=df.index, columns=df.columns
    )

    def _filter(col: pd.Series) -> pd.Series:
        X = col.values.reshape(-1, 1)
        try:
            dp = BayesianGaussianMixture(
                n_components=max_components,
                covariance_type='full',
                weight_concentration_prior_type='dirichlet_process',
                weight_concentration_prior=1e-2,
                random_state=random_state
            )
            labels = dp.fit_predict(X)
            means = dp.means_.ravel()
            if means.size == 0:
                return pd.Series(0, index=col.index)
            keep = means.argmax()
            # zero out everything not in the highest-mean component
            return col.where(labels == keep, 0)
        except Exception:
            return pd.Series(0, index=col.index)

    # apply to each column
    return df_norm.apply(_filter)


def process_niche_res_list(
    niche_res_list: list[pd.DataFrame],
    max_components: int = 3,
    random_state: int = 0) -> list[pd.DataFrame]:
    """
    Process a list of DataFrames with process_dataframe_dpgmm, showing a progress bar.
    """
    return [
        process_dataframe_dpgmm(df, max_components, random_state)
        for df in tqdm(niche_res_list, desc="Processing DataFrames")
    ]