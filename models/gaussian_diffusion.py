import enum
import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class ModelMeanType(enum.Enum):
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon

class GaussianDiffusion(nn.Module):
    def __init__(self, mean_type, noise_schedule, noise_scale, noise_min, noise_max,
            steps, device, history_num_per_term=10, beta_fixed=True):

        self.mean_type = mean_type
        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.device = device

        self.history_num_per_term = history_num_per_term
        self.Lt_history = th.zeros(steps, history_num_per_term, dtype=th.float64).to(device)
        self.Lt_count = th.zeros(steps, dtype=int).to(device)


        self.betas = th.tensor(self.get_betas(), dtype=th.float64).to(self.device)
        if beta_fixed:
            self.betas[0] = 0.00001 
        assert len(self.betas.shape) == 1, "betas must be 1-D"
        assert len(self.betas) == self.steps, "num of betas must equal to diffusion steps"
        assert (self.betas > 0).all() and (self.betas <= 1).all(), "betas out of range"

        self.calculate_for_diffusion()

        super(GaussianDiffusion, self).__init__()

    def get_betas(self):
        """
        Given the schedule name, create the betas for the diffusion process.
        """
        if self.noise_schedule == "linear" or self.noise_schedule == "linear-var":
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max

            if self.noise_schedule == "linear":
                betas = np.linspace(start, end, self.steps, dtype=np.float64)
            else:
                betas = betas_from_linear_variance(
                    self.steps,
                    np.linspace(start, end, self.steps, dtype=np.float64)
                )
        elif self.noise_schedule == "cosine":
            betas = betas_for_alpha_bar(
                self.steps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            )
        elif self.noise_schedule == "binomial":
            ts = np.arange(self.steps)
            betas = [1 / (self.steps - t + 1) for t in ts]
        else:
            raise NotImplementedError(f"unknown beta schedule: {self.noise_schedule}!")

        betas = np.asarray(betas, dtype=np.float64)
        betas = np.clip(betas, 1e-8, 0.999)
        return betas

    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = th.cumprod(alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = th.cat([th.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(self.device)  # alpha_{t-1}
        self.alphas_cumprod_next = th.cat([self.alphas_cumprod[1:], th.tensor([0.0]).to(self.device)]).to(self.device)  # alpha_{t+1}
        assert self.alphas_cumprod_prev.shape == (self.steps,)

        self.sqrt_alphas_cumprod = th.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = th.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = th.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = th.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = th.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_log_variance_clipped = th.log(
            th.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * th.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * th.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
    def sample_timesteps(self, batch_size, device, method='uniform', uniform_prob=0.001):
        if method == 'importance':  # importance sampling
            if not (self.Lt_count == self.history_num_per_term).all():
                return self.sample_timesteps(batch_size, device, method='uniform')
            
            Lt_sqrt = th.sqrt(th.mean(self.Lt_history ** 2, axis=-1))
            pt_all = Lt_sqrt / th.sum(Lt_sqrt)
            pt_all *= 1- uniform_prob
            pt_all += uniform_prob / len(pt_all)

            assert pt_all.sum(-1) - 1. < 1e-5

            t = th.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)

            return t, pt
        
        elif method == 'uniform':  # uniform sampling
            t = th.randint(0, self.steps, (batch_size,), device=device).long()
            pt = th.ones_like(t).float()

            return t, pt
            
        else:
            raise ValueError
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, model, x, t):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """
        B, C = x.shape[:2]
        assert t.shape == (B, )
        model_output = model(x, t)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
        
        if self.mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        elif self.mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x, t, eps=model_output)
        else:
            raise NotImplementedError(self.mean_type)
        
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def SNR(self, t):
        t = t.long().clamp(min=0, max=self.steps - 1)
        ac = self.alphas_cumprod.to(t.device)[t]
        return ac / (1 - ac + 1e-12)
    
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        # res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
        
    @th.no_grad()
    def p_sample(self, model, x_start, steps, sampling_noise=False):
        assert steps <= self.steps, "Too much steps in inference."
        if steps == 0:
            x_t = x_start
        else:
            t = th.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start, t)

        indices = list(range(steps))[::-1]

        if self.noise_scale == 0.:
            for i in indices:
                t = th.tensor([i] * x_t.shape[0]).to(x_start.device)
                x_t = model(x_t, t)
            return x_t

        for i in indices:
            t = th.tensor([i] * x_t.shape[0]).to(x_start.device)
            out = self.p_mean_variance(model, x_t, t)
            if sampling_noise:
                noise = th.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )  # no noise when t == 0
                x_t = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
            else:
                x_t = out["mean"]
        return x_t

    def training_losses(self, model, x_start, reweight=False):
        batch_size, device = x_start.size(0), x_start.device
        ts, pt = self.sample_timesteps(batch_size, device, 'importance')
        noise = th.randn_like(x_start)
        if self.noise_scale != 0.:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        terms = {}
        model_output = model(x_t, ts)
        target = {
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.mean_type]

        assert model_output.shape == target.shape == x_start.shape

        mse = mean_flat((target - model_output) ** 2)
        loss = mse
        weight = th.ones_like(mse)
        if reweight == True:
            if self.mean_type == ModelMeanType.START_X:
                weight = self.SNR(ts - 1) - self.SNR(ts)
                weight = th.where((ts == 0), 1.0, weight)
                loss = mse
            elif self.mean_type == ModelMeanType.EPSILON:
                weight = (1 - self.alphas_cumprod[ts]) / ((1-self.alphas_cumprod_prev[ts])**2 * (1-self.betas[ts]))
                weight = th.where((ts == 0), 1.0, weight)
                likelihood = mean_flat((x_start - self._predict_xstart_from_eps(x_t, ts, model_output))**2 / 2.0)
                loss = th.where((ts == 0), likelihood, mse)
        else:
            weight = th.tensor([1.0] * len(target)).to(device)

        terms["loss"] = weight * loss
        
        # update Lt_history & Lt_count
        for t, loss in zip(ts, terms["loss"]):
            if self.Lt_count[t] == self.history_num_per_term:
                Lt_history_old = self.Lt_history.clone()
                self.Lt_history[t, :-1] = Lt_history_old[t, 1:]
                self.Lt_history[t, -1] = loss.detach()
            else:
                try:
                    self.Lt_history[t, self.Lt_count[t]] = loss.detach()
                    self.Lt_count[t] += 1
                except:
                    print(t)
                    print(self.Lt_count[t])
                    print(loss)
                    raise ValueError

        terms["loss"] /= pt
        return terms

    def training_losses_cfg(self, model, x_start, cond_drop_prob=0.1, reweight=False):
        """
        CFG training with Condition Dropout

        """
        batch_size, device = x_start.size(0), x_start.device
        ts, pt = self.sample_timesteps(batch_size, device, 'importance')
        noise = th.randn_like(x_start)

        if self.noise_scale != 0.:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        cond = x_start.clone()

        if cond_drop_prob > 0:
            drop_mask = th.rand(batch_size, 1, device=device) < cond_drop_prob
            cond = cond * (~drop_mask).float()


        model_output = model(x_t, ts, cond)

        target = {
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.mean_type]

        assert model_output.shape == target.shape == x_start.shape

        mse = mean_flat((target - model_output) ** 2)
        loss = mse

        weight = th.ones_like(mse)
        if reweight == True:
            if self.mean_type == ModelMeanType.START_X:
                weight = self.SNR(ts - 1) - self.SNR(ts)
                weight = th.where((ts == 0), 1.0, weight)
                loss = mse
            elif self.mean_type == ModelMeanType.EPSILON:
                weight = (1 - self.alphas_cumprod[ts]) / (
                        (1 - self.alphas_cumprod_prev[ts]) ** 2 * (1 - self.betas[ts])
                )
                weight = th.where((ts == 0), 1.0, weight)
                likelihood = mean_flat(
                    (x_start - self._predict_xstart_from_eps(x_t, ts, model_output)) ** 2 / 2.0
                )
                loss = th.where((ts == 0), likelihood, mse)
        else:
            weight = th.tensor([1.0] * len(target)).to(device)

        terms = {}
        terms["loss"] = weight * loss

        for t, loss_val in zip(ts, terms["loss"]):
            if self.Lt_count[t] == self.history_num_per_term:
                Lt_history_old = self.Lt_history.clone()
                self.Lt_history[t, :-1] = Lt_history_old[t, 1:]
                self.Lt_history[t, -1] = loss_val.detach()
            else:
                try:
                    self.Lt_history[t, self.Lt_count[t]] = loss_val.detach()
                    self.Lt_count[t] += 1
                except:
                    print(t)
                    print(self.Lt_count[t])
                    print(loss_val)
                    raise ValueError

        terms["loss"] /= pt
        return terms

    @th.no_grad()
    def p_sample_cfg(self, model, x_start, steps, cfg_scale=1.0, sampling_noise=False):
        """
         CFG sampling
        """
        assert steps <= self.steps, "Too much steps in inference."
        if steps == 0:
            x_t = x_start
        else:
            t = th.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start, t)
            

        indices = list(range(steps))[::-1]

        for i in indices:
            t = th.tensor([i] * x_t.shape[0]).to(x_start.device)

            # CFG
            # Conditioned prediction
            with th.no_grad():
                output_cond = model(x_t, t, x_start)

            # Unconditioned prediction
            with th.no_grad():
                output_uncond = model(x_t, t, None)

            model_output = output_uncond + cfg_scale * (output_cond - output_uncond)

            # x_0
            if self.mean_type == ModelMeanType.START_X:
                pred_xstart = model_output
            elif self.mean_type == ModelMeanType.EPSILON:
                pred_xstart = self._predict_xstart_from_eps(x_t, t, eps=model_output)
            else:
                raise NotImplementedError(self.mean_type)

            # Calculate p_mean_variance
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x_t, t=t
            )

            # Sample x_{t-1}
            if sampling_noise:
                noise = th.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )
                model_variance = self._extract_into_tensor(
                    self.posterior_variance, t, x_t.shape
                )
                x_t = model_mean + nonzero_mask * th.sqrt(model_variance) * noise
            else:
                x_t = model_mean

        return x_t

    @th.no_grad() 
    def p_sample_ag(self, model_main, model_weak, x_start, 
        steps, cond, w=2.0, sampling_noise=False): 
        """
        Autoguidance (AG) sampling: zw = w * z1 + (1 - w) * z0 (w >= 1)
        """
        assert steps <= self.steps, "Too much steps in inference."
        assert w >= 1.0, "auto guidance weight w must be >= 1.0"
        if steps == 0:
            x_t = x_start
        else:
            t = th.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start, t)

        indices = list(range(steps))[::-1]

        for i in indices:
            t = th.tensor([i] * x_t.shape[0]).to(x_start.device)
            
            output_main = model_main(x_t, t, cond)
            output_weak = model_weak(x_t, t, cond)

            model_output = w * output_main + (1.0 - w) * output_weak

            # x_0 prediction
            if self.mean_type == ModelMeanType.START_X:
                pred_xstart = model_output
            elif self.mean_type == ModelMeanType.EPSILON:
                pred_xstart = self._predict_xstart_from_eps(x_t, t, eps=model_output)
            else:
                raise NotImplementedError(self.mean_type)

            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x_t, t=t
            )

            # Sample x_{t-1}
            if sampling_noise:
                noise = th.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )
                model_variance = self._extract_into_tensor(
                    self.posterior_variance, t, x_t.shape
                )
                x_t = model_mean + nonzero_mask * th.sqrt(model_variance) * noise
            else:
                x_t = model_mean

        return x_t  

    def training_losses_a2g(
        self, main_model, weak_model, guide_net, x_start, cond,
        pop_global_Q_head: float = 0.35,
        pop_global_Q_tail: float = 0.45,
        lambda_ag: float = 0.2,
        lambda_pop: float = 0.2,
        reweight: bool = False,
        pop_K_candidate: int = 50,
        item_groups=None,
        popularity_bins=None,
    ):
        """
        Adaptive Autoguidance (A2G).
        """
        assert pop_global_Q_head + pop_global_Q_tail <= 1.0, "pop_global_Q_head + pop_global_Q_tail must be less than or equal to 1.0"
        B, device = x_start.size(0), x_start.device
        ts, pt = self.sample_timesteps(B, device, 'importance')
        noise = th.randn_like(x_start)

        if self.noise_scale != 0.:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        #  Forward main & weak
        out_main = main_model(x_t, ts, cond)
        with th.no_grad():
            out_weak = weak_model(x_t, ts, cond)

        target = {
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.mean_type]

        if self.mean_type == ModelMeanType.EPSILON:
            x0_main = self._predict_xstart_from_eps(x_t, ts, out_main)
            x0_weak = self._predict_xstart_from_eps(x_t, ts, out_weak)
            z1_for_ag, z0_for_ag = x0_main, x0_weak
            base_pred = out_main
        else:
            z1_for_ag, z0_for_ag = out_main, out_weak
            base_pred = out_main

        # Base diffusion loss
        mse = mean_flat((target - base_pred) ** 2)
        loss_use = mse
        weight = th.ones_like(mse)

        if reweight:
            if self.mean_type == ModelMeanType.START_X:
                weight = self.SNR(ts - 1) - self.SNR(ts)
                weight = th.where((ts == 0), 1.0, weight)
            else:  # EPSILON mode
                weight = (1 - self.alphas_cumprod[ts]) / (
                    (1 - self.alphas_cumprod_prev[ts]) ** 2 * (1 - self.betas[ts])
                )
                weight = th.where((ts == 0), 1.0, weight)
                likelihood = mean_flat(
                    (x_start - self._predict_xstart_from_eps(x_t, ts, base_pred)) ** 2 / 2.0
                )
                loss_use = th.where((ts == 0), likelihood, mse)

        w = guide_net(z1_for_ag.detach(), z0_for_ag.detach(), item_groups=item_groups)  # [B]

        wv = w.view(-1, 1).expand_as(z1_for_ag)
        z_ag = wv * z1_for_ag + (1.0 - wv) * z0_for_ag

        ag_mse = mean_flat((z_ag - x_start) ** 2)

        loss_main_ag = weight * (loss_use + lambda_ag * ag_mse)
        loss_total = loss_main_ag / pt

        # 4. Unified popularity regularization
        loss_pop = th.tensor(0.0, device=device)
        if popularity_bins is not None and lambda_pop > 0:

            if isinstance(popularity_bins, th.Tensor):
                pop_bins_t = popularity_bins.to(device=device)
            else:
                pop_bins_t = th.as_tensor(popularity_bins, device=device)

            head_mask = (pop_bins_t == 1)
            mid_mask  = (pop_bins_t == 2)
            tail_mask = (pop_bins_t == 3)

            head_mask_f = head_mask.float().unsqueeze(0)
            mid_mask_f  = mid_mask.float().unsqueeze(0)
            tail_mask_f = tail_mask.float().unsqueeze(0)

            eps = 1e-8

            hist_head = (x_start * head_mask_f).sum(dim=1)   # [B]
            hist_mid  = (x_start * mid_mask_f).sum(dim=1)    # [B]
            hist_tail = (x_start * tail_mask_f).sum(dim=1)   # [B]

            hist_counts = th.stack([hist_head, hist_mid, hist_tail], dim=1)  # [B, 3]
            hist_counts = hist_counts + eps
            H_batch = hist_counts / hist_counts.sum(dim=1, keepdim=True)     # [B, 3]
            H_mean = H_batch.mean(dim=0)                                     # [3]

            pop_global_Q_mid = 1.0 - pop_global_Q_head - pop_global_Q_tail
            global_Q = th.tensor([pop_global_Q_head, pop_global_Q_mid, pop_global_Q_tail], device=device, dtype=x_start.dtype)

            head_ratio_hist = H_mean[0]
            
            gamma_m = 1.0 - head_ratio_hist
            gamma_m = th.clamp(gamma_m, 0.0, 1.0)

            T = gamma_m * H_mean + (1.0 - gamma_m) * global_Q   # [3]
            T = T / T.sum()

            # Top-K aware popularity distribution from z_ag
            tau_pop = 1.0
            probs = F.softmax(z_ag / tau_pop, dim=1)   # [B, I]

        
            K_candidate = pop_K_candidate
            K_candidate = min(K_candidate, probs.size(1))

            probs_topk, idx_topk = probs.topk(K_candidate, dim=1)   # [B, K], [B, K]

            bins_topk = pop_bins_t[idx_topk]   # [B, K]

            topk_head = (bins_topk == 1).sum(dim=1)   # [B]
            topk_mid  = (bins_topk == 2).sum(dim=1)   # [B]
            topk_tail = (bins_topk == 3).sum(dim=1)   # [B]

            rec_dist = th.stack(
                [topk_head, topk_mid, topk_tail], dim=1
            ).float()                      # [B, 3]
            rec_dist = rec_dist + eps
            rec_dist = rec_dist / rec_dist.sum(dim=1, keepdim=True)

            # (1) Head over-exposure
            head_ratio_rec = rec_dist[:, 0]   # [B]
            head_target    = T[0]             # scalar
            over_head      = F.relu(head_ratio_rec - head_target)   # [B]
            L_head = over_head.mean()

            # (2) Tail under-exposure
            tail_ratio_rec = rec_dist[:, 2]   # [B]
            tail_target    = T[2]             # scalar
            under_tail     = F.relu(tail_target - tail_ratio_rec)   # [B]
            L_tail = under_tail.mean()

            # (3) Overall imbalance via entropy
            rec_mean = rec_dist.mean(dim=0)                         # [3]
            rec_mean = rec_mean / (rec_mean.sum() + eps)

            H_rec = -(rec_mean * (rec_mean + eps).log()).sum()
            H_T   = -(T * (T + eps).log()).sum()

            L_entropy = F.relu(H_T - H_rec)

            alpha_h = getattr(self, "pop_alpha_head", 1.0)
            beta_t  = getattr(self, "pop_beta_tail", 1.0)
            gamma_e = getattr(self, "pop_gamma_entropy", 1.0)

            pop_loss = alpha_h * L_head + beta_t * L_tail + gamma_e * L_entropy
            loss_pop = pop_loss
            loss_total = loss_total + lambda_pop * loss_pop


        terms = {
            "loss": loss_total,
            "loss_base": (weight * loss_use) / pt,
            "loss_ag": (weight * (lambda_ag * ag_mse)) / pt,
            "loss_pop": loss_pop,
            "w_mean": w.mean().detach(),
            "w_std": w.std().detach(),
            "w_min": w.min().detach(),
            "w_max": w.max().detach(),
        }


        for t, l in zip(ts, terms["loss"]):
            if self.Lt_count[t] == self.history_num_per_term:
                old = self.Lt_history.clone()
                self.Lt_history[t, :-1] = old[t, 1:]
                self.Lt_history[t, -1] = l.detach()
            else:
                self.Lt_history[t, self.Lt_count[t]] = l.detach()
                self.Lt_count[t] += 1

        return terms

    @th.no_grad()
    def p_sample_a2g(
        self,
        model_main,
        model_weak,
        guide_net,
        x_start,
        steps,
        cond,
        sampling_noise=False,
        item_groups=None,
    ):
        assert steps <= self.steps, "Too many steps."
        device = x_start.device

        if steps == 0:
            x_t = x_start
        else:
            t = th.tensor([steps - 1] * x_start.shape[0], device=device)
            x_t = self.q_sample(x_start, t)

        indices = list(range(steps))[::-1]

        for i in indices:
            t = th.tensor([i] * x_t.shape[0], device=device)

            z1 = model_main(x_t, t, cond)
            z0 = model_weak(x_t, t, cond)

            if self.mean_type == ModelMeanType.EPSILON:
                x0_main = self._predict_xstart_from_eps(x_t, t, z1)
                x0_weak = self._predict_xstart_from_eps(x_t, t, z0)
                
                w = guide_net(x0_main, x0_weak, item_groups=item_groups)  # [B]
                
                wv = w.view(-1, 1).expand_as(x0_main)
                z_ag = wv * x0_main + (1.0 - wv) * x0_weak
                pred_xstart = z_ag
                
            else:  # START_X mode
                # Get adaptive weights directly
                w = guide_net(z1, z0, item_groups=item_groups)  # [B]
                
                wv = w.view(-1, 1).expand_as(z1)
                z_ag = wv * z1 + (1.0 - wv) * z0
                pred_xstart = z_ag

            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x_t, t=t
            )

            if sampling_noise:
                noise = th.randn_like(x_t)
                nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                model_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
                x_t = model_mean + nonzero_mask * th.sqrt(model_variance) * noise
            else:
                x_t = model_mean

        return x_t

    def training_losses_self_guidance(self, main_model, guidance_net, x_start,
                                      guidance_weight=0.1, reweight=False):
        batch_size, device = x_start.size(0), x_start.device
        ts, pt = self.sample_timesteps(batch_size, device, 'importance')
        noise = th.randn_like(x_start)

        if self.noise_scale != 0.:
            x_t = self.q_sample(x_start, ts, noise) # adding noise
        else:
            x_t = x_start

        # main model
        main_output = main_model(x_t, ts)

        # self-guidance: y = x_start
        guidance_correction = guidance_net(x_t, x_start) # the direction to correct main_output

        # x_θ(x_t, t) + w·ε_φ(x_t, x_0)
        model_output = main_output + guidance_weight * guidance_correction

        target = {
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.mean_type]

        assert model_output.shape == target.shape == x_start.shape

        mse = mean_flat((target - model_output) ** 2)
        loss = mse

        weight = th.ones_like(mse)
        if reweight == True:
            if self.mean_type == ModelMeanType.START_X:
                weight = self.SNR(ts - 1) - self.SNR(ts)
                weight = th.where((ts == 0), 1.0, weight)
                loss = mse
            elif self.mean_type == ModelMeanType.EPSILON:
                weight = (1 - self.alphas_cumprod[ts]) / (
                        (1 - self.alphas_cumprod_prev[ts]) ** 2 * (1 - self.betas[ts])
                )
                weight = th.where((ts == 0), 1.0, weight)
                likelihood = mean_flat(
                    (x_start - self._predict_xstart_from_eps(x_t, ts, model_output)) ** 2 / 2.0
                )
                loss = th.where((ts == 0), likelihood, mse)
        else:
            weight = th.tensor([1.0] * len(target)).to(device)

        terms = {}
        terms["loss"] = weight * loss

        # update Lt_history & Lt_count
        for t, loss_val in zip(ts, terms["loss"]):
            if self.Lt_count[t] == self.history_num_per_term:
                Lt_history_old = self.Lt_history.clone()
                self.Lt_history[t, :-1] = Lt_history_old[t, 1:]
                self.Lt_history[t, -1] = loss_val.detach()
            else:
                try:
                    self.Lt_history[t, self.Lt_count[t]] = loss_val.detach()
                    self.Lt_count[t] += 1
                except:
                    print(t)
                    print(self.Lt_count[t])
                    print(loss_val)
                    raise ValueError

        terms["loss"] /= pt
        return terms

    def p_sample_self_guidance(self, main_model, guidance_net, x_start, steps,
                               guidance_weight=0.1, sampling_noise=False):
        """
        Self-guidance sampling
        """
        assert steps <= self.steps, "Too much steps in inference."

        if steps == 0:
            x_t = x_start
        else:
            t = th.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start, t)

        indices = list(range(steps))[::-1]

        if self.noise_scale == 0.:
            for i in indices:
                t = th.tensor([i] * x_t.shape[0]).to(x_start.device)
                pred_x0 = main_model(x_t, t)
                correction = guidance_net(x_t, x_start)
                x_t = pred_x0 + guidance_weight * correction
            return x_t

        for i in indices:
            t = th.tensor([i] * x_t.shape[0]).to(x_start.device)

            # Main model predict x_0
            with th.no_grad():
                pred_x0 = main_model(x_t, t)

            # Calculate p_mean_variance
            if self.mean_type == ModelMeanType.START_X:
                pred_xstart = pred_x0
            elif self.mean_type == ModelMeanType.EPSILON:
                pred_xstart = self._predict_xstart_from_eps(x_t, t, eps=pred_x0)
            else:
                raise NotImplementedError(self.mean_type)

            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x_t, t=t
            )

            # Self-guidance: y = x_start
            with th.no_grad():
                correction = guidance_net(x_t, x_start)

            # Mean = μ(x_t, t) + w·ε_φ(x_t, y)
            guided_mean = model_mean + guidance_weight * correction

            # Sample x_{t-1}
            if sampling_noise:
                noise = th.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )
                model_variance = self._extract_into_tensor(
                    self.posterior_variance, t, x_t.shape
                )
                x_t = guided_mean + nonzero_mask * th.sqrt(model_variance) * noise
            else:
                x_t = guided_mean

        return x_t
        
def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))