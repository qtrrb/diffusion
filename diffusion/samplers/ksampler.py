import k_diffusion as K
import torch.nn as nn
import torch


# based on Katherine Crowson's Text to Image (CC12M Diffusion)
class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, cond, uncond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale


class KSampler:
    def __init__(self, model, sampler):
        self.model = model
        self.sampler = sampler

    @torch.no_grad()
    def sample(
        self,
        S,
        conditioning,
        batch_size,
        shape,
        unconditional_guidance_scale,
        unconditional_conditioning,
        **kwargs
    ):
        model_wrap = K.external.CompVisDenoiser(self.model)
        sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()
        sigmas = K.sampling.get_sigmas_karras(S, sigma_min, sigma_max, device="cuda")
        model_wrap = CFGDenoiser(model_wrap)
        x = torch.randn([batch_size, *shape], device="cuda") * sigmas[0]
        extra_args = {
            "cond": conditioning,
            "uncond": unconditional_conditioning,
            "cond_scale": unconditional_guidance_scale,
        }
        sampler = getattr(K.sampling, self.sampler)
        samples = sampler(model_wrap, x, sigmas, extra_args=extra_args)
        return samples, ""
