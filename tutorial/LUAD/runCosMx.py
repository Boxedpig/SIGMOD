import pandas as pd
import numpy as np
import scanpy as sc
import anndata as AD
import os
import warnings
warnings.filterwarnings('ignore')

import torch

import sklearn.neighbors
from sklearn import preprocessing

from sklearn.preprocessing import minmax_scale

from entmax import entmax_bisect

import pickle

import importlib

import seaborn as sns
import matplotlib.pyplot as plt

from SIGMOD import SIGMOD_main as NG
from SIGMOD import SIGMOD_utils as ut

import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7,8,9'

device = "cuda:0"
entmax_prior = 1.5

cosmx = sc.read_h5ad("./Data/cosmx_human_nsclc_clustered.h5ad")
adata = cosmx[(cosmx.obs['sample']=='LUAD-9 R1')]
adata.obs['CellType'] = adata.obs['cell_type'].str.replace(r'tumor \d+', 'tumors', regex=True)

spatial_net = pd.read_csv("./Data/spatial_net.csv")
# spatial_net_df_all = spatial_net.pivot_table(index='Cell1', columns='cluster',fill_value=0, aggfunc='size')

celltypes = adata.obs['CellType'].unique()

#['tumors','epithelial','neutrophil','fibroblast','endothelial','T CD8 memory','B-cell','T CD4 memory','T CD4 naive','NK','monocyte','macrophage','plasmablast','Treg','T CD8 naive','mDC','pDC','mast']
# celltypes = adata.obs['CellType'].astype(str).unique()
# celltypes = ['tumors']

truncted_max = 10
truncted_min = 0

topic_num = {
 'tumors':3,
 'epithelial':2,
 'neutrophil':2,
 'fibroblast':3,
 'endothelial':2,
 'T CD8 memory':2,
 'B-cell':2,
 'T CD4 memory':2,
 'T CD4 naive':2,
 'NK':2,
 'monocyte':2,
 'macrophage':2,
 'plasmablast':2,
 'Treg':2,
 'T CD8 naive':2,
 'mDC':2,
 'pDC':2,
 'mast':2
}
prior = {
 'tumors':1,
 'epithelial':1,
 'neutrophil':1,
 'fibroblast':1,
 'endothelial':1,
 'T CD8 memory':1,
 'B-cell':1,
 'T CD4 memory':1,
 'T CD4 naive':1,
 'NK':1,
 'monocyte':1,
 'macrophage':1,
 'plasmablast':1,
 'Treg':1,
 'T CD8 naive':1,
 'mDC':1,
 'pDC':1,
 'mast':1
}

start_time = time.time()
for celltype in celltypes:

    if not os.path.exists('./Results/'+celltype):
        os.mkdir('./Results/'+celltype) 
    
    cells = list(adata.obs[adata.obs.CellType==celltype].index)

    ### Get cell type specific anndata and spatial net dataframe
    spatial_net_df = spatial_net.pivot_table(index='Cell1', columns='cluster',fill_value=0, aggfunc='size')
    spatial_net_df.index = spatial_net_df.index.astype(str)
    # ## min-max normalzation
    spatial_net_df_ = minmax_scale(spatial_net_df,feature_range=(0,1), axis=1)
    spatial_net_df = pd.DataFrame(spatial_net_df_,index =spatial_net_df.index,columns= spatial_net_df.columns)
    # spatial_net_df = spatial_net_df.div(spatial_net_df.sum(axis=1)+1e-6, axis=0)
    
    adata_subset = adata.copy()
    adata_subset = adata_subset[cells,:]
    intersect_cells = spatial_net_df.index.intersection(adata_subset.obs.index)
    spatial_net_df = spatial_net_df.loc[intersect_cells,:]
    spatial_net_df.insert(len(spatial_net_df.columns), 'Self', 1)
    adata_subset =adata_subset[intersect_cells,:]
    
    G = torch.tensor(spatial_net_df.values)

    pos = pd.DataFrame(adata_subset.obsm['spatial'],columns = ['x','y'])
    pos['sample'] = "LUAD-9"
    L = ut.compute_sparse_laplacian_for_multiple_samples(pos, sample_column='sample', k=10, sigma=100)
    L_file = 'Data/LUAD_' + celltype + '_L.pickle'
    with open(L_file, 'wb') as f:
         pickle.dump(L,f)

    # L_file = 'Data/LUAD_' + celltype + '_L.pickle'
    # with open(L_file, 'rb') as f:
    #      L = pickle.load(f)

    pos_tensor = torch.from_numpy(pos[["x","y"]].values)
    print("Position dimension \t", pos_tensor.shape)
    
    n_genes = adata_subset.shape[1]
    n_obs = adata_subset.obs.shape[0]
    n_niches = G.shape[1]
    print(n_obs,'\t',n_niches)

    X= adata_subset.layers['counts'].toarray() ### Extract Raw counts
    Apt_dense = torch.tensor(X)
    
    ind = torch.range(0, n_obs-1)
    b_index = adata_subset.obs.CellType 
    le_obs = preprocessing.LabelEncoder()
    b_index_ = le_obs.fit_transform(b_index)
    b_index_ = torch.as_tensor(b_index_)
    b_index_ = b_index_.reshape([n_obs,1])

    ind=ind.to(device)
    b_index_=b_index_.to(device)
    Apt_dense = Apt_dense.to(device)
    neighborCell = G.to(device)
    neighborCell = neighborCell.type(torch.float)
    pos_tensor = pos_tensor.to(device)
    L = L.to(device)

    NGmod = NG.NicheGuidedDeconv(torch.tensor(n_obs),
                              torch.tensor(n_genes),
                              torch.tensor(1), # batch num
                              torch.tensor(topic_num[celltype]), # topics num
                              torch.tensor(n_niches), # niches num
                              entmax_prior = entmax_prior,
                              cudadevice = device)

    NGmod.train(
            x_data = Apt_dense,
            niche_mat = neighborCell, 
            pos=pos_tensor,
            batch_index=b_index_,
            L=L,
            prior_strength = torch.tensor(1.0) *  prior[celltype],
            truncted_max = torch.tensor(1.0) * truncted_max,
            truncted_min = torch.tensor(1.0) * truncted_min,
            use_niche = True,
            spatial_regularization = 200,
            lr = 1e-1,
            n_iter=500)
    
    posterior_samples = NGmod.get_posterior_samples(num_samples=100)

    eta = posterior_samples['eta'].mean(dim=0)
    caux = posterior_samples['caux'].mean(dim=0)
    delta = posterior_samples['delta'].mean(dim=0)
    tau = posterior_samples['tau'].mean(dim=0)
    lambda_ = posterior_samples['lambda_'].mean(dim=0)

    lambda_tilde = torch.sqrt(
            (caux**2 * tau**2 * delta**2 * lambda_**2)
            / (caux**2 + tau**2 * delta**2 * lambda_**2)
            )
    scaled_eta = eta * lambda_tilde# 

    theta = posterior_samples['theta'].mean(dim=0)
    theta_logit = entmax_bisect(theta,alpha=entmax_prior,dim=1)
    topic_words = posterior_samples['per_topic_mu_fg'].mean(dim=0)
    topic_words_softmax = topic_words.softmax(dim=-1)

    detection_y_s = posterior_samples['detection_y_s'].mean(dim=0)

    s_g_gene_add = posterior_samples['s_g_gene_add'].mean(dim=0)

    alpha_g_inverse = posterior_samples['alpha_g_inverse'].mean(dim=0)
    alpha = torch.ones((1,1),device = device) / alpha_g_inverse.pow(2)

    prefix = './Results/'+celltype+'/SIGMOD_'

    eta_df = pd.DataFrame(eta.cpu().detach().numpy())
    eta_df.index = spatial_net_df.columns
    eta_df.to_csv(prefix+'eta_raw_denovo.csv')
    eta_df = pd.DataFrame(scaled_eta.cpu().detach().numpy())
    eta_df.index = spatial_net_df.columns
    eta_df.to_csv(prefix+'eta_denovo.csv')
    theta_df = pd.DataFrame(theta.cpu().detach().numpy())
    theta_df.index = adata_subset.obs.index
    theta_df.to_csv(prefix+'theta_denovo.csv')
    theta_df = pd.DataFrame(theta_logit.cpu().detach().numpy())
    theta_df.index = adata_subset.obs.index
    theta_df.to_csv(prefix+'theta_logit_denovo.csv')
    topic_words_df = pd.DataFrame(topic_words.cpu().detach().numpy())
    topic_words_df.columns = adata_subset.var_names
    topic_words_df.to_csv(prefix+'topicWords_denovo.csv')
    topic_words_df = pd.DataFrame(topic_words_softmax.cpu().detach().numpy())
    topic_words_df.columns = adata_subset.var_names
    topic_words_df.to_csv(prefix+'topicWords_softmax_denovo.csv')


end_time = time.time()
print(f"Running Time: {end_time - start_time:.4f} s")