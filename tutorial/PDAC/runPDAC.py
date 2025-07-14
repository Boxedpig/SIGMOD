import scanpy as sc
import pandas as pd
import numpy as np
from sklearn import preprocessing
import scipy.sparse as sp

import pickle

import os

import torch
import scipy as sp

from entmax import entmax_bisect

import sys
sys.path.append("..")

from SIGMOD import SIGMOD_main as NG
from SIGMOD import SIGMOD_utils as ut

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7,8,9'

devices = "cuda:0"
entmax_prior = 1.25


with open('/home/liuchang/3.0_Result/4.0_NGDeonv/0315/PDAC/Data/sc_mean_normal.pickle', 'rb') as f:
    inf_aver = pickle.load(f)

niche_df = pd.read_csv("/home/liuchang/3.0_Result/4.0_NGDeonv/0315/PDAC/Results/SVLR/2_niche12_res.csv")
ligandsExp = np.array(niche_df.T)
ligandsExp = ligandsExp.astype(np.float32)
ligandsExp_t = torch.from_numpy(ligandsExp.T)
print("niche dimension \t", ligandsExp_t.shape)

stdata = sc.read_h5ad("/home/liuchang/3.0_Result/4.0_NGDeonv/0315/PDAC/Data/st.h5ad")
stdata.obs["sample"] = "PDAC"
stdata.var.index = stdata.var.features
stdata.var_names_make_unique()

pos = stdata.obs.loc[:,["x","y","sample"]]
L = ut.compute_sparse_laplacian_for_multiple_samples(pos, sample_column='sample', k=4, sigma=1)

L_file = './Results/SIGMOD/PDAC_L.pickle'
with open(L_file, 'wb') as f:
     pickle.dump(L,f)

pos_tensor = torch.from_numpy(pos[["x","y"]].values)
print("niche dimension \t", pos_tensor.shape)

intersec_genes = list(set(stdata.var_names) & set(inf_aver.columns))
print("Intersect genes length \t", len(intersec_genes))

n_obs = stdata.shape[0]
print("Number of Oberservation \t", n_obs)
n_var = len(intersec_genes)
print("Number of Var \t", n_var)
n_niche = niche_df.shape[1]
print("Number of Niche \t", n_niche)

stdata_raw = stdata
stdata_raw = stdata_raw[stdata.obs_names, intersec_genes]
X = stdata_raw.X
Acoo = X.tocoo()
Apt = torch.sparse.LongTensor(torch.LongTensor([Acoo.row.tolist(), Acoo.col.tolist()]),
                              torch.LongTensor(Acoo.data.astype(float)))
Apt_dense = Apt.to_dense()

inf_aver = inf_aver.loc[:, intersec_genes]
mean_per_cluster_mu_fg = torch.from_numpy(inf_aver.values).float()
mean_per_cluster_mu_fg.shape

n_topic = mean_per_cluster_mu_fg.shape[0]
print("Number of Topic \t", n_topic)

b_index = stdata.obs['sample']
le_obs = preprocessing.LabelEncoder()
b_index_ = le_obs.fit_transform(b_index)
b_index_ = torch.as_tensor(b_index_)
b_index_ = b_index_.reshape([n_obs,1])
b_index_=b_index_.to(devices)

n_batch = len(stdata.obs['sample'].unique())
print("Number of Batch \t", n_batch)

mean_per_cluster_mu_fg = mean_per_cluster_mu_fg.to(devices)
Apt_dense = Apt_dense.to(devices)
ligandsExp_t = ligandsExp_t.to(devices)
pos_tensor = pos_tensor.to(devices)
L = L.to(devices)

NGmod = NG.NicheGuidedDeconv(n_obs=torch.tensor(n_obs), 
                             n_vars=torch.tensor(n_var), 
                             n_batch=torch.tensor(n_batch), 
                             n_topics=torch.tensor(n_topic), 
                             n_niches=torch.tensor(n_niche), 
                             entmax_prior = entmax_prior,
                             cudadevice=devices, 
                             used_cov=False)
NGmod.train(ref_signatures = mean_per_cluster_mu_fg, 
            x_data = Apt_dense,
            niche_mat = ligandsExp_t, 
            pos=pos_tensor,
            batch_index=b_index_,
            L = L,
            use_niche = True,
            spatial_regularization = 200,
            lr = 0.1,
            n_iter=500,
            guide = 'AutoNormal')

posterior_samples = NGmod.get_posterior_samples(num_samples=300)

eta = posterior_samples['eta'].mean(dim=0)
caux = posterior_samples['caux'].mean(dim=0)
delta = posterior_samples['delta'].mean(dim=0)
tau = posterior_samples['tau'].mean(dim=0)
lambda_ = posterior_samples['lambda_'].mean(dim=0)
lambda_tilde = torch.sqrt(
            (caux**2 * tau**2 * delta**2 * lambda_**2)
            / (caux**2 + tau**2 * delta**2 * lambda_**2)
        )
scaled_eta = eta * lambda_tilde

theta = posterior_samples['theta'].mean(dim=0) 
theta_logit = entmax_bisect(theta,alpha=entmax_prior,dim=1)

prefix = "SIGMOD0515"

eta_df = pd.DataFrame(eta.cpu().detach().numpy())
eta_df.columns = inf_aver.index
eta_df.to_csv(f'./Results/{prefix}/eta.csv')

scaled_eta_df = pd.DataFrame(scaled_eta.cpu().detach().numpy())
scaled_eta_df.columns = inf_aver.index
scaled_eta_df.to_csv(f'./Results/{prefix}/eta_scaled.csv')

theta_logit_df = pd.DataFrame(theta_logit.cpu().detach().numpy())
theta_logit_df.columns = inf_aver.index
theta_logit_df.to_csv(f'./Results/{prefix}/theta_logit.csv')

theta_df = pd.DataFrame(theta.cpu().detach().numpy())
theta_df.columns = inf_aver.index
theta_df.to_csv(f'./Results/{prefix}/theta.csv')