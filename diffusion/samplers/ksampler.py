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
        if isinstance(cond, dict):
            assert isinstance(uncond, dict)

            if uncond["c_crossattn"][0].size(1) < cond["c_crossattn"][0].size(1):
                padding_length = cond["c_crossattn"][0].size(1) - uncond["c_crossattn"][
                    0
                ].size(1)
                padding = torch.zeros(
                    (
                        uncond["c_crossattn"][0].size(0),
                        padding_length,
                        uncond["c_crossattn"][0].size(2),
                    )
                ).to("cuda")
                uncond["c_crossattn"][0] = torch.cat(
                    [uncond["c_crossattn"][0], padding], dim=1
                )
            elif uncond["c_crossattn"][0].size(1) > cond["c_crossattn"][0].size(1):
                padding_length = uncond["c_crossattn"][0].size(1) - cond["c_crossattn"][
                    0
                ].size(1)
                padding = torch.zeros(
                    (
                        cond["c_crossattn"][0].size(0),
                        padding_length,
                        cond["c_crossattn"][0].size(2),
                    )
                ).to("cuda")
                cond["c_crossattn"][0] = torch.cat(
                    [cond["c_crossattn"][0], padding], dim=1
                )

            cond_in = dict()
            for k in cond:
                if isinstance(cond[k], list):
                    cond_in[k] = [
                        torch.cat([uncond[k][i], cond[k][i]])
                        for i in range(len(cond[k]))
                    ]
                else:
                    cond_in[k] = torch.cat([uncond[k], cond[k]])
        else:
            if uncond.size(1) < cond.size(1):
                padding_length = cond.size(1) - uncond.size(1)
                padding = torch.zeros(
                    (uncond.size(0), padding_length, uncond.size(2))
                ).to("cuda")
                uncond = torch.cat([uncond, padding], dim=1)
            elif uncond.size(1) > cond.size(1):
                padding_length = uncond.size(1) - cond.size(1)
                padding = torch.zeros((cond.size(0), padding_length, cond.size(2))).to(
                    "cuda"
                )
                cond = torch.cat([cond, padding], dim=1)
            cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale


class KSampler:
    def __init__(self, model, sampler):
        self.model = model
        self.sampler = sampler
        self.model_wrap = K.external.CompVisDenoiser(self.model)
        self.sigma_min, self.sigma_max = (
            self.model_wrap.sigmas[0].item(),
            self.model_wrap.sigmas[-1].item(),
        )
        self.timesteps = None

    @torch.no_grad()
    def make_schedule(self, ddim_num_steps, **kwargs):
        self.timesteps = ddim_num_steps

    @torch.no_grad()
    def stochastic_encode(self, x0, t):
        sigmas = K.sampling.get_sigmas_karras(
            self.timesteps, self.sigma_min, self.sigma_max, device="cuda"
        )
        z = x0 + torch.randn_like(x0) * sigmas[self.timesteps - t]
        return z

    @torch.no_grad()
    def decode(
        self,
        z_enc,
        cond,
        t_enc,
        unconditional_guidance_scale,
        unconditional_conditioning,
    ):
        sigmas = K.sampling.get_sigmas_karras(
            self.timesteps, self.sigma_min, self.sigma_max, device="cuda"
        )
        sigmas = sigmas[self.timesteps - t_enc :]
        model_wrap = CFGDenoiser(self.model_wrap)
        extra_args = {
            "cond": cond,
            "uncond": unconditional_conditioning,
            "cond_scale": unconditional_guidance_scale,
        }
        sampler = getattr(K.sampling, self.sampler)
        samples = sampler(model_wrap, z_enc, sigmas, extra_args=extra_args)
        return samples

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
        sigma_min, sigma_max = (
            self.model_wrap.sigmas[0].item(),
            self.model_wrap.sigmas[-1].item(),
        )
        sigmas = K.sampling.get_sigmas_karras(S, sigma_min, sigma_max, device="cuda")
        model_wrap = CFGDenoiser(self.model_wrap)
        x = torch.randn([batch_size, *shape], device="cuda") * sigmas[0]
        extra_args = {
            "cond": conditioning,
            "uncond": unconditional_conditioning,
            "cond_scale": unconditional_guidance_scale,
        }
        sampler = getattr(K.sampling, self.sampler)
        samples = sampler(model_wrap, x, sigmas, extra_args=extra_args)
        return samples, ""
