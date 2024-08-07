import os

import numpy as np
import torch
import torchvision.utils as tvu

from tqdm import tqdm, trange

from functions.process_data import download_process_data
from models.diffusion import Model


def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    (bs,) = t.shape
    assert x_shape[0] == bs
    out = torch.gather(a, 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out


def denoising_step_flexible_mask(x, t, *, model, logvar, betas):
    """
    Sample from p(x_{t-1} | x_t)
    """
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)

    model_output = model(x, t)
    weighted_score = betas / torch.sqrt(1 - alphas_cumprod)
    mean = extract(1 / torch.sqrt(alphas), t, x.shape) * (x - extract(weighted_score, t, x.shape) * model_output)

    logvar = extract(logvar, t, x.shape)
    noise = torch.randn_like(x)
    mask = 1 - (t == 0).float()
    mask = mask.reshape((x.shape[0],) + (1,) * (len(x.shape) - 1))
    sample = mean + mask * torch.exp(0.5 * logvar) * noise
    sample = sample.float()
    return sample


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = torch.tensor(np.log(np.append(posterior_variance[1], betas[1:])), device=self.device)

        elif self.model_var_type == "fixedsmall":
            self.logvar = torch.tensor(np.log(np.maximum(posterior_variance, 1e-20)), device=self.device)

    @torch.no_grad
    def image_editing_sample(self):
        # print("Loading model")
        ## URLs are broken
        # if self.config.data.dataset == "LSUN":
        #     if self.config.data.category == "bedroom":
        #         url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
        #     elif self.config.data.category == "church_outdoor":
        #         url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        # elif self.config.data.dataset == "CelebA_HQ":
        #     url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        # else:
        #     raise ValueError
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                model_ckpt_path = "checkpoints/diffusion_lsun_bedroom_model/model-2388000.ckpt"
            elif self.config.data.category == "church_outdoor":
                model_ckpt_path = "checkpoints/diffusion_lsun_church_model/model-4432000.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":
            model_ckpt_path = "checkpoints/celeba_hq.ckpt"
        else:
            raise ValueError

        model = Model(self.config)
        ckpt = torch.load(model_ckpt_path, map_location=self.device, weights_only=True)
        model.load_state_dict(ckpt)
        model.to(self.device)
        # model = torch.nn.DataParallel(model)  ## Slow! Do not use
        model.eval()
        # print("Model loaded")

        # download_process_data(path="colab_demo")
        n = self.config.sampling.batch_size
        # print("Start sampling")

        [mask, img] = torch.load(self.args.mask_image_file, weights_only=True)

        mask = mask.to(self.config.device)
        img = img.to(self.config.device).unsqueeze(dim=0).repeat(n, 1, 1, 1)
        x0 = img

        tvu.save_image(x0, os.path.join(self.args.image_folder, f"original_input.png"))
        x0 = (x0 - 0.5) * 2.0

        for sample_itr in trange(self.args.sample_step, desc="Sample", leave=True):
            tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f"x0_{sample_itr}.png"))

            e = torch.randn_like(x0)
            total_noise_levels = self.args.t
            a = (1 - self.betas).cumprod(dim=0)
            x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder, f"init_{sample_itr}.png"))

            for i in tqdm(reversed(range(total_noise_levels)), total=total_noise_levels, desc=f"Iteration {sample_itr}"):
                t = (torch.ones(n) * i).to(self.device)
                x_ = denoising_step_flexible_mask(x, t=t, model=model, logvar=self.logvar, betas=self.betas)
                x = x0 * a[i].sqrt() + e * (1.0 - a[i]).sqrt()
                x[:, (mask != 1.0)] = x_[:, (mask != 1.0)]
                # added intermediate step vis
                if (i - 99) % 100 == 0:
                    img_path = os.path.join(self.args.image_folder, f"noise_t_{i:03d}_{sample_itr}.png")
                    tvu.save_image((x + 1) * 0.5, img_path)

            x0[:, (mask != 1.0)] = x[:, (mask != 1.0)]
            # torch.save(x, os.path.join(self.args.image_folder, f"samples_{sample_itr}.pth"))
            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder, f"samples_{sample_itr}.png"))
