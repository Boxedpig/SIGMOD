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

from entmax import entmax_bisect

import pickle

import importlib

import seaborn as sns
import matplotlib.pyplot as plt

import sys
from SIGMOD import SIGMOD_main as NG
from SIGMOD import SIGMOD_utils as ut

import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7,8,9'

adata = sc.read_10x_h5(
    filename="./Data/cell_feature_matrix.h5"
)

df = pd.read_csv(
    "./Data/obs.csv",index_col=0)
df.index = df.index.astype(str)

adata = adata[df.index,]
adata.obs = df.copy()
adata.obs["Sample"] = 'BRCA'

spatial_net_df = pd.read_csv("./Data/LRpair_matrix_nolabel.csv",index_col=0)
spatial_net_df = spatial_net_df.apply(lambda row: 
                         (row - row.min()) / (row.max() - row.min()) 
                         if row.max() != row.min() else row, axis=1)
spatial_net_df.index = spatial_net_df.index.astype(str)

adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].copy().to_numpy()

topic_num = 20
prior_strength = 1
entmax_prior = 1.5
truncted_min = 0
truncted_max = 10

device = "cuda:0"

# os.mkdir('./Results/') 

cells = list(adata.obs.index)

### Get cell type specific anndata and spatial net dataframe

adata = adata[cells,:]
intersect_cells = spatial_net_df.index.intersection(adata.obs.index)

pos = pd.DataFrame(adata.obsm['spatial'],columns = ['x','y'])
pos['sample'] = "BRCA"
L = ut.compute_sparse_laplacian_for_multiple_samples(pos, sample_column='sample', k=5, sigma=3)

L_file = './Data/BRCA' + '_L.pickle'
with open(L_file, 'wb') as f:
     pickle.dump(L,f)

spatial_net_df = spatial_net_df.loc[intersect_cells,:]
adata =adata[intersect_cells,:]

G = torch.tensor(spatial_net_df.values)

pos_tensor = torch.from_numpy(pos[["x","y"]].values)
print("Position dimension \t", pos_tensor.shape)

n_genes = adata.shape[1]
n_obs = adata.obs.shape[0]
n_niches = G.shape[1]
print(n_obs,'\t',n_niches)

# L_file = 'BRCA'  + '_L.pickle'
# with open(L_file, 'rb') as f:
#      L = pickle.load(f)

X= adata.X.toarray()
Apt_dense = torch.tensor(X)

ind = torch.range(0, n_obs-1)
b_index = adata.obs.Sample 
le_obs = preprocessing.LabelEncoder()
b_index_ = le_obs.fit_transform(b_index)
b_index_ = torch.as_tensor(b_index_)
b_index_ = b_index_.reshape([n_obs,1])

ind=ind.to(device)
b_index_=b_index_.to(device)
Apt_dense = Apt_dense.to(device)
neighborCell = G.to(device)
neighborCell = neighborCell.type(torch.float)
L = L.to(device)


# ### Init gene expression
sc.pp.normalize_total(adata)
adj = adata.X
adj = adj.astype("float32").tocoo()
values = adj.data
indices = np.vstack((adj.row, adj.col))
i = torch.LongTensor(indices)
v = torch.FloatTensor(values)
shape = adj.shape
adj = torch.sparse_coo_tensor(i, v, torch.Size(shape))
col_sums = torch.sparse.sum(adj, dim=0).to_dense()
means = col_sums / shape[0]
init_bg_mean = np.log(means + 1e-15)
init_bg_mean = init_bg_mean.to(device)

os.makedirs(f"Results/Others/SIGMOD", exist_ok = True)

start_time = time.time()

for j in range(5):
    NGmod = NG.NicheGuidedDeconv(torch.tensor(n_obs),
                                torch.tensor(n_genes),
                                torch.tensor(1), # batch num
                                torch.tensor(topic_num), # topics num
                                torch.tensor(n_niches), # niches num
                                entmax_prior = entmax_prior,
                                cudadevice = device)

    NGmod.train(
            x_data = Apt_dense,
            niche_mat = neighborCell, 
            pos=pos_tensor,
            batch_index=b_index_,
            L=L,
            init_bg_mean = init_bg_mean,
            prior_strength = torch.tensor(1.0) *  prior_strength,
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
    topic_words = posterior_samples['beta'].mean(dim=0)
    bg = posterior_samples['bg'].mean(dim=0)
    topic_words_softmax = topic_words+(bg + init_bg_mean)

    prefix = 'Results/Others/SIGMOD/'

    eta_df = pd.DataFrame(eta.cpu().detach().numpy())
    eta_df.index = spatial_net_df.columns
    eta_df.to_csv(prefix+f"eta_raw{j}.csv")
    eta_df = pd.DataFrame(scaled_eta.cpu().detach().numpy())
    eta_df.index = spatial_net_df.columns
    eta_df.to_csv(prefix+f"eta{j}.csv")
    
    theta_df = pd.DataFrame(theta.cpu().detach().numpy())
    theta_df.index = adata.obs.index
    theta_df.to_csv(prefix+f"theta{j}.csv")
    theta_df = pd.DataFrame(theta_logit.cpu().detach().numpy())
    theta_df.index = adata.obs.index
    theta_df.to_csv(prefix+f"topic_proportions{j}.csv")

    topic_words_df = pd.DataFrame(topic_words.cpu().detach().numpy())
    topic_words_df.columns = adata.var_names
    topic_words_df.to_csv(prefix+f"topic_assignments{j}.csv")
    topic_words_df = pd.DataFrame(topic_words_softmax.cpu().detach().numpy())
    topic_words_df.columns = adata.var_names
    topic_words_df.to_csv(prefix+f"topicWords_softmax{j}.csv")

end_time = time.time()
print(f"Running Time: {end_time - start_time:.4f} s")