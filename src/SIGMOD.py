from typing import Callable, Dict, Any
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.distributions import Gamma, Rejector
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.infer.autoguide import AutoGuideList, AutoNormal, AutoDelta
from pyro.nn import PyroModule

from entmax import entmax_bisect

def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    """Creates a one-hot encoded tensor."""
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)

class TruncatedGamma(TorchDistribution):
    """
    A Gamma distribution truncated to a specified interval [low, high].
    Uses rejection sampling for the `sample` method.
    """
    arg_constraints = {
        'concentration': constraints.positive,
        'rate': constraints.positive,
        'low': constraints.dependent,
        'high': constraints.dependent
    }
    support = constraints.interval(0.0, float('inf'))
    has_rsample = False

    def __init__(self, concentration, rate, low, high, validate_args=None):
        self.concentration = concentration
        self.rate = rate
        self.low = low
        self.high = high

        if torch.any(self.low >= self.high):
            raise ValueError("Parameter 'low' must be less than 'high'.")
        if torch.any(self.low < 0):
            raise ValueError("Parameter 'low' must be non-negative.")

        self.base_dist = Gamma(concentration, rate)

        self.Z = self.base_dist.cdf(high) - self.base_dist.cdf(low)

        self.support = constraints.interval(low, high)

        log_prob_low = self.base_dist.log_prob(self.low)
        log_prob_high = self.base_dist.log_prob(self.high)
        self._log_scale = -(torch.min(log_prob_low, log_prob_high)) + 1e-6

        super().__init__(self.base_dist.batch_shape, self.base_dist.event_shape, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        """Generate samples using rejection sampling."""
        proposal = self.base_dist

        def log_prob_accept(x):
            """Accept samples within the truncated range."""
            return torch.where(
                (x >= self.low) & (x <= self.high),
                torch.zeros_like(x),
                torch.tensor(float('-inf'), device=x.device)
            )

        rejector = Rejector(proposal, log_prob_accept, self._log_scale, event_shape=self.base_dist.event_shape)
        samples = rejector.sample(sample_shape)
        return samples

    def log_prob(self, value):
        """Calculates the log probability of a value in the truncated distribution."""
        if self._validate_args:
            self._validate_sample(value)
        log_prob = self.base_dist.log_prob(value) - torch.log(self.Z)
        log_prob = torch.where(
            (value >= self.low) & (value <= self.high),
            log_prob,
            torch.tensor(float('-inf'), device=value.device)
        )
        return log_prob


class NicheGuidedDeconv(PyroModule):
    """
    A Bayesian model for niche-guided deconvolution of spatial transcriptomics data.
    """
    def __init__(
        self,
        n_obs: int,
        n_vars: int,
        n_batch: int,
        n_topics: int,
        n_niches: int,
        entmax_prior: float,
        hyp_prior: Dict[str, torch.Tensor],
        cudadevice: str = "cuda",
        used_spatial: bool = False,
        used_cov: bool = False,     
    ):
        super().__init__()
        pyro.clear_param_store()

        self.n_obs = n_obs
        self.n_vars = n_vars
        self.n_batch = n_batch
        self.n_topics = n_topics
        self.n_niches = n_niches
        self.entmax_prior = entmax_prior
        self.used_spatial = used_spatial
        self.used_cov = used_cov
        self.hypers_to_block = list(hyp_prior.keys())

        # Set device and move hyperparameters
        self.device = torch.device(cudadevice if torch.cuda.is_available() else "cpu")
        for key, value in hyp_prior.items():
            setattr(self, key, value.to(self.device))
        
        self.ones_n_batch_1 = torch.ones((self.n_batch, 1), device=self.device)
        self.ones = torch.ones((1, 1), device=self.device)

        # Initialize data-related attributes to None
        self.Ref_Signatures = None
        self.X_data = None
        self.batch_index = None
        self.niche_mat = None
        self.pos = None
        self.L = None # Adjacency matrix
        
    def train(
        self,
        x_data: torch.Tensor,
        niche_mat: torch.Tensor,
        pos: torch.Tensor,
        batch_index: torch.Tensor,
        L: torch.Tensor,
        ref_signatures: torch.Tensor,
        prior_strength: torch.Tensor,
        truncted_min: torch.Tensor,
        truncted_max: torch.Tensor,
        init_bg_mean: torch.Tensor = None,
        use_niche: bool = True,
        use_bg_mean: bool = False,
        spatial_regularization: float = 200.0,
        n_iter: int = 1000,
        lr: float = 0.1,
        guide: str = "AutoNormal",
    ):
        """
        Trains the model using stochastic variational inference.

        Args:
            x_data: Gene expression data (observations x genes).
            niche_mat: Niche information matrix (observations x niches).
            pos: Spatial coordinates of observations.
            batch_index: Batch index for each observation.
            L: Adjacency matrix for spatial regularization.
            ref_signatures: Reference cell type signatures (topics x genes).
            prior_strength: Strength for the word-topic distribution prior.
            truncted_min: Minimum value for the truncated gamma prior.
            truncted_max: Maximum value for the truncated gamma prior.
            init_bg_mean: Initial background mean (optional).
            use_niche: Flag to use niche information.
            use_bg_mean: Flag to use background mean.
            spatial_regularization: Weight for the spatial regularization term.
            n_iter: Number of optimization iterations.
            lr: Learning rate for the Adam optimizer.
            guide: The type of Pyro guide to use for inference.
        """

        pyro.clear_param_store()

        self.X_data = x_data.to(self.device)
        self.niche_mat = niche_mat.to(self.device)
        self.pos = pos.to(self.device)
        self.batch_index = batch_index.to(self.device)
        self.L = L.to(self.device)
        self.Ref_Signatures = ref_signatures.to(self.device) if ref_signatures is not None else None
        self.prior_strength = prior_strength.to(self.device)
        self.truncted_min = truncted_min.to(self.device)
        self.truncted_max = truncted_max.to(self.device)
        self.init_bg_mean = init_bg_mean.to(self.device) if init_bg_mean is not None else None

        self.model = pyro.poutine.block(self._model, hide=self.hypers_to_block)
        self.guide = self._get_guide(guide, self.model)

        loss_fn = pyro.infer.TraceEnum_ELBO().differentiable_loss

        with pyro.poutine.trace(param_only=True) as param_capture:
            loss_fn(
                self.model, self.guide, self.Ref_Signatures, self.init_bg_mean,
                self.prior_strength, self.truncted_max, self.truncted_min, self.X_data,
                self.niche_mat, self.batch_index, use_niche, use_bg_mean
            )

        # Now only works for AutoDelta For other distribution, should be matched with its shape
        params = {site["value"].unconstrained() for site in param_capture.trace.nodes.values()}
        optimizer = torch.optim.Adam(params, lr=lr, betas=(0.90, 0.999))

        self.losses = []
        for i in range(n_iter):
            elbo_loss = loss_fn(
                self.model, self.guide, self.Ref_Signatures, self.init_bg_mean,
                self.prior_strength, self.truncted_max, self.truncted_min, self.X_data,
                self.niche_mat, self.batch_index, use_niche, use_bg_mean
            )
            reg_loss = self.regularized_loss(params) * spatial_regularization
            loss = elbo_loss + reg_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            self.losses.append(loss.item())
            if i % 20 == 0:
                print(f"[iteration {i+1:04d}] loss: {elbo_loss:.4f} (ELBO) + {reg_loss:.4f} (Reg) = {loss.item():.4f}")

    def _model(
        self,
        ref_signatures, init_bg_mean, prior_strength, truncted_max, truncted_min,
        x_data, niche_mat, batch_index, use_niche, use_bg_mean,
    ):
        """The generative model definition."""
        obs2sample = one_hot(batch_index, self.n_batch)
        niche_mu = torch.zeros(self.n_obs, self.n_topics, device=self.device)

        if use_niche:
            with pyro.plate('niche_influence_plate', self.n_niches):
                eta = pyro.sample(
                    "eta",
                    dist.MultivariateNormal(
                        torch.zeros(self.n_topics, device=self.device),
                        torch.eye(self.n_topics, device=self.device)
                    )
                ) # (n_niches, n_topics)
            
            # Hierarchical shrinkage prior (Horseshoe) for eta
            caux = pyro.sample("caux", dist.InverseGamma(0.5 * self.ones, 0.5 * self.ones))
            with pyro.plate('topic_plate', self.n_topics):
                delta = pyro.sample("delta", dist.HalfCauchy(self.ones))
            with pyro.plate('niche_plate', self.n_niches):
                tau = pyro.sample("tau", dist.HalfCauchy(self.ones))
                with pyro.plate('topic_plate_inner', self.n_topics):
                    lambda_ = pyro.sample("lambda_", dist.HalfCauchy(self.ones))
            
            # Calculate scaled eta
            lambda_tilde_sq = (caux**2 * tau**2 * delta**2 * lambda_**2) / \
                              (caux**2 + tau**2 * delta**2 * lambda_**2)
            scaled_eta = eta * torch.sqrt(lambda_tilde_sq)
            niche_mu = niche_mat @ scaled_eta  # (n_obs, n_topics)
        
        # Cell abundances (theta)
        if self.used_cov:
            scale = pyro.sample('scale', dist.Gamma(torch.ones(self.n_topics, device=self.device), 1.0).to_event(1))
            cholesky_corr = pyro.sample('cholesky_corr', dist.LKJCholesky(self.n_topics, 1.0))
            L_cov = cholesky_corr * torch.sqrt(scale[:, None])
            with pyro.plate("cell_abundances", self.n_obs):
                theta = pyro.sample("theta", dist.MultivariateNormal(niche_mu, scale_tril=L_cov))
        else:
            with pyro.plate("cell_abundances", self.n_obs):
                theta = pyro.sample(
                    "theta",
                    dist.MultivariateNormal(niche_mu, torch.eye(self.n_topics, device=self.device))
                )

        theta_logit = entmax_bisect(theta, alpha=self.entmax_prior, dim=1) # (n_obs, n_topics)

        # Gene expression level scaling (m_g)
        m_g_mean = pyro.sample("m_g_mean", dist.Gamma(self.m_g_mu_mean_var_ratio_hyp * self.m_g_mu_hyp, self.m_g_mu_mean_var_ratio_hyp))
        m_g_alpha_e_inv = pyro.sample("m_g_alpha_e_inv", dist.Exponential(self.m_g_alpha_hyp_mean))
        m_g_alpha_e = self.ones / m_g_alpha_e_inv.pow(2)
        m_g = pyro.sample("m_g", dist.Gamma(m_g_alpha_e, m_g_alpha_e / m_g_mean).expand([1, self.n_vars]).to_event(2))

        # Location-specific detection efficiency (detection_y_s)
        detection_mean_y_e = pyro.sample("detection_mean_y_e", dist.Gamma(self.detection_mean_hyp_prior_alpha, self.detection_mean_hyp_prior_beta).expand([self.n_batch, 1]).to_event(2))
        detection_hyp_prior = self.ones_n_batch_1 * self.detection_hyp_prior_alpha
        beta_detect = (obs2sample @ detection_hyp_prior) / (obs2sample @ detection_mean_y_e)
        with pyro.plate('obs_plate', self.n_obs, dim=-2):
            detection_y_s = pyro.sample("detection_y_s", dist.Gamma(obs2sample @ detection_hyp_prior, beta_detect))

        # Gene-specific additive component (s_g_gene_add)
        s_g_gene_add_alpha_hyp = pyro.sample("s_g_gene_add_alpha_hyp", dist.Gamma(self.gene_add_alpha_hyp_prior_alpha, self.gene_add_alpha_hyp_prior_beta))
        s_g_gene_add_mean = pyro.sample("s_g_gene_add_mean", dist.Gamma(self.gene_add_mean_hyp_prior_alpha, self.gene_add_mean_hyp_prior_beta).expand([self.n_batch, 1]).to_event(2))
        s_g_gene_add_alpha_e_inv = pyro.sample("s_g_gene_add_alpha_e_inv", dist.Exponential(s_g_gene_add_alpha_hyp).expand([self.n_batch, 1]).to_event(2))
        s_g_gene_add_alpha_e = self.ones / s_g_gene_add_alpha_e_inv.pow(2)
        s_g_gene_add = pyro.sample("s_g_gene_add", dist.Gamma(s_g_gene_add_alpha_e, s_g_gene_add_alpha_e / s_g_gene_add_mean).expand([self.n_batch, self.n_vars]).to_event(2))

        # Gene-specific overdispersion (alpha)
        alpha_g_phi_hyp = pyro.sample("alpha_g_phi_hyp", dist.Gamma(self.alpha_g_phi_hyp_prior_alpha, self.alpha_g_phi_hyp_prior_beta))
        alpha_g_inverse = pyro.sample("alpha_g_inverse", dist.Exponential(alpha_g_phi_hyp).expand([1, self.n_vars]).to_event(2))
        alpha = self.ones / alpha_g_inverse.pow(2)

        # Expected expression (mu)
        if ref_signatures is not None:
            # ref_signatures = F.softmax(ref_signatures,dim=-1)
            base_expr = (theta_logit @ ref_signatures) * m_g
            # Add batch-specific additive term if more than one batch exists
            if self.n_batch > 1:
                mu = (base_expr + (obs2sample @ s_g_gene_add)) * detection_y_s
            else:
                mu = base_expr * detection_y_s
        else:
            if use_bg_mean:
                with pyro.plate("genes_plate_bg", self.n_vars):
                    bg = pyro.sample("bg", dist.Normal(torch.zeros_like(init_bg_mean), torch.ones_like(init_bg_mean))) + init_bg_mean
                with pyro.plate("topic_gene_plate", self.n_topics, dim=-2):
                    with pyro.plate("genes_plate_beta", self.n_vars):
                        beta = pyro.sample("beta", dist.Normal(0., 1.)) + bg
            else:
                beta = pyro.sample(
                    "per_topic_mu_fg",
                    TruncatedGamma(prior_strength, prior_strength, truncted_min, truncted_max).expand([self.n_topics, self.n_vars]).to_event(2)
                )
            
            per_topic_mu_fg = F.softmax(beta, dim=-1)
            mu = (theta_logit @ per_topic_mu_fg) * detection_y_s

        # Likelihood       
        # Batch Size
        # with pyro.plate('observe_data', size=self.n_obs,subsample_size=100) as ind:
        #     pyro.sample(
        #         "data_target",
        #         dist.GammaPoisson(concentration = alpha, rate = alpha / mu[ind]).to_event(1),
        #         infer={"enumerate": "parallel"},
        #         obs = x_data.index_select(0, ind),
        #     )
        with pyro.plate('observe_data', self.n_obs, dim=-1):
            pyro.sample(
                "data_target",
                dist.GammaPoisson(concentration=alpha, rate=alpha / mu).to_event(1),
                infer={"enumerate": "parallel"},
                obs=x_data,
            )
    
    
    def _get_guide(self, guide_name: str, model: Callable) -> AutoGuideList:
        """Constructs and returns a Pyro guide."""
        if guide_name == "AutoNormal":
            guide = AutoGuideList(model)
            guide.append(AutoNormal(poutine.block(model, hide=["theta", "d", "cholesky_corr", "scale"])))
            guide.append(AutoDelta(poutine.block(model, expose=["theta"])))
            return guide
        raise ValueError(f"Guide '{guide_name}' not recognized.")
    
    def regularized_loss(self, params: set) -> torch.Tensor:
        """
        Calculates a spatial regularization loss to encourage similar cell abundances
        in adjacent spots. Loss = trace(Θ^T * L * Θ).
        """
        # Find the theta parameter for regularization
        theta_param = next((p for p in params if p.shape == (self.n_obs, self.n_topics)), None)
        
        if theta_param is None:
            return torch.tensor(0.0, device=self.device)

        # Apply entmax transformation to get cell abundances
        theta = theta_param.reshape(self.n_obs, self.n_topics)
        theta_abundances = entmax_bisect(theta, alpha=self.entmax_prior, dim=1)
        
        # Calculate regularization term
        output = torch.matmul(torch.matmul(theta_abundances.T, self.L), theta_abundances)
        return torch.trace(output)
        
    def get_posterior_samples(self, num_samples: int = 100) -> Dict[str, torch.Tensor]:
        """
        Generates samples from the posterior distribution of the latent variables.
        """
        posterior_samples = defaultdict(list)
        # Note: This requires the model arguments to be set as attributes correctly
        for _ in range(num_samples):
            guide_trace = poutine.trace(self.guide).get_trace(
                self.Ref_Signatures, self.init_bg_mean, self.prior_strength, 
                self.truncted_max, self.truncted_min, self.X_data, self.niche_mat, 
                self.batch_index, True, False # Assuming default use_niche/use_bg_mean
            )
            for name, node in guide_trace.nodes.items():
                if node["type"] == "sample":
                    posterior_samples[name].append(node["value"])

        return {k: torch.stack(v) for k, v in posterior_samples.items()}
