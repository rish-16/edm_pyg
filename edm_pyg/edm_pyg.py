import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_geometric.nn as tgnn

import numpy as np
from edm_pyg import edm_utils

class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_init_offset: int = -2):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze() * 1000
        assert len(x.shape) == 1
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class DefaultNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """
    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = edm_utils.cosine_beta_schedule(timesteps)
        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = edm_utils.polynomial_schedule(timesteps, s=precision, power=power)
        else:
            # custom noise scheduler
            # raise ValueError(noise_schedule)
            alphas2 = noise_schedule(timesteps)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        self.gamma = torch.nn.Parameter(torch.from_numpy(-log_alphas2_to_sigmas2).float(), requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]

class GammaNetwork(torch.nn.Module):
    """The gamma network models a monotonic increasing function. Construction as in the VDM paper."""
    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.]))
        self.show_schedule()

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print('Gamma schedule:')
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
                gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma

class EDMWrapper(nn.Module):
    def __init__(self, 
                dynamics, 
                num_timesteps, 
                indim,
                scheduler, 
                objective, 
                batch_size, 
                condition_time=True, 
                grad_clip=False, 
                ema=False, 
                ema_decay=None, 
                device=None
            ):
        super().__init__()

        self.dynamics = dynamics
        self.indim = indim
        self.num_timesteps = num_timesteps
        self.scheduler = scheduler

        if self.scheduler == "learned":
            self.gamma = GammaNetwork()
        else:
            self.gamma = DefaultNoiseSchedule(scheduler, timesteps=num_timesteps, precision=1e-4)

        self.objective = objective
        self.batch_size = batch_size
        self.condition_time = condition_time
        self.grad_clip = grad_clip
        self.ema = ema
        self.ema_decay = ema_decay
        self.device = device

    def sigma(self, gamma):
        return torch.sqrt(torch.sigmoid(gamma))

    def alpha(self, gamma):
        return torch.sqrt(torch.sigmoid(-gamma))

    def SNR(self, gamma):
        return torch.exp(-gamma)

    def normalize(self, x, ptr):
        pass

    def unnormalize(self, x, h_cat, h_int, ptr):
        pass

    def sigma_and_alpha_t_given_s(self, gamma_t, gamma_s):
        pass

    def kl_prior(self, z, ptr):
        pass

    def compute_x_pred(self, pred, zt, gamma_t):
        pass

    def objective(self, pred, gamma_t, noise):
        pass

    def log_constants_p_x_given_z0(self, x, ptr):
        dof_x = 

        zeros = torch.zeros((x.size(0), 1))
        gamma_0 = self.gamma(zeros)

        log_sigma_x = 0.5 * gamma_0

        return dof_x * (-log_sigma_x - 0.5 * np.log(2 * np.pi))

    def sample_p_xh_given_z0(self, z0, ptr, fix_noise):
        zeros = torch.zeros(size=(z0.size(0), 1)).to(z0.device)
        gamma_0 = self.gamma(zeros)
        sigma_x = self.SNR(-0.5 * gamma_0)
        net_out = self.phi(z0, zeros, ptr)

        mu_x = self.compute_x_pred(net_out, z0, gamma_0)
        xh = self.sample_normal(mu_x, sigma_x, ptr, fix_noise)

        x = xh[:, :3]

        h_int = z0[:,-1:] if self.include_charges else torch.zeros(0).to(z0.device)
        x, h_cat, h_int = self.unnormalize(x, z0[:, 3:-1], h_int, ptr)

        h_cat = F.one_hot(torch.argmax(h_cat, dim=-1), self.num_classes)
        h_int = torch.round(h_int).long()
        h = {'integer': h_int, 'categorical': h_cat}
        return x, h

    def sample_normal(self, mu, sigma, ptr, fix_noise):
        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_noise(ptr)
        return my + sigma * eps

    def log_p_xh_z0_without_constants(self, x, h, zt, gamma_0, noise, pred, ptr, epsilon=1e-10):
        pass

    def compute_loss(self, x, h, ptr, include_t0):
        pass

    def forward(self, batch):
        x, h, score = self.normalize(batch)

        if self.training and self.loss_type == "l2":
            score = torch.zeros_like(score)

        if self.training:
            loss, loss_dict = self.compute_loss(batch.x, batch.h, batch.batch, include_t0=False)
        else:
            loss, loss_dict = self.compute_loss(batch.x, batch.h, batch.batch, include_t0=True)

        nex_log_p_xh = loss

        assert neg_log_pxh.size() == delta_log_px.size()
        neg_log_pxh = neg_log_pxh - delta_log_px

        return neg_log_pxh

    def phi(self, z, t, ptr):
        pass

    def sample_p_zs_given_zt(self, s, t, zt, ptr, fix_noise):
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s)

        sigma_s = self.sigma(gamma_s)
        sigma_t = self.sigma(gamma_t)

        eps_t = self.phi(zt, t, ptr)

        edm_utils.assert_mean_zero(zt[:, :3], ptr)
        edm_utils.assert_mean_zero(eps_t[:, :3], ptr)

        mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t
        sigma = sigma2_t_given_s * sigma_s / sigma_t

        zs = self.sample_normal(mu, sigma, ptr, fix_noise)
        zs = torch.cat([
                edm_utils.remove_mean(zs[:, :3], ptr),
                zs[:, 3:]
            ], dim=1)

        return zs

    def sample_noise(self, ptr):
        # sample_combined_position_feature_noise
        z_x = edm_utils.sample_cog_zero_gaussian(ptr)
        z_h = edm_utils.sample_regular_gaussian(ptr)
        z = torch.cat([z_x, z_h], dim=1)

        return z

    @torch.no_grad()
    def sample(self, n_samples, ptr, fix_noise):
        z = self.sample_noise(ptr) # merge single sample and multi sample noise sampling
        edm_utils.assert_mean_zero(z, ptr) # TODO: write assertion

        for s in reversed(range(0, self.num_timesteps)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.num_timesteps
            t_array = t_array / self.num_timesteps

            z = self.sample_p_zs_given_zt(s_array, t_array, z, ptr, fix_noise)

        x, h = self.sample_p_xh_given_z0(z, ptr, fix_noise)

        edm_utils.assert_mean_zero(x, ptr)
        edm_utils.remove_mean_if_necessary(x, ptr)

        return x, h