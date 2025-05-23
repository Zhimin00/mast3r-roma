import os
import math
from re import U
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import warnings
from warnings import warn
from PIL import Image
import pdb
from romatch.utils import get_tuple_transform_ops
from romatch.utils.local_correlation import local_correlation
from romatch.utils.utils import cls_to_flow_refine, get_autocast_params, cls_to_flow
from romatch.utils.kde import kde

class ConvRefiner(nn.Module):
    def __init__(
        self,
        in_dim=6,
        hidden_dim=16,
        out_dim=2,
        dw=False,
        kernel_size=5,
        hidden_blocks=3,
        displacement_emb = None,
        displacement_emb_dim = None,
        local_corr_radius = None,
        corr_in_other = None,
        no_im_B_fm = False,
        amp = False,
        concat_logits = False,
        use_bias_block_1 = True,
        use_cosine_corr = False,
        disable_local_corr_grad = False,
        is_classifier = False,
        sample_mode = "bilinear",
        norm_type = nn.BatchNorm2d,
        bn_momentum = 0.1,
        amp_dtype = torch.float16,
    ):
        super().__init__()
        self.bn_momentum = bn_momentum
        self.block1 = self.create_block(
            in_dim, hidden_dim, dw=dw, kernel_size=kernel_size, bias = use_bias_block_1,
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    kernel_size=kernel_size,
                    norm_type=norm_type,
                )
                for hb in range(hidden_blocks)
            ]
        )
        self.hidden_blocks = self.hidden_blocks
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        if displacement_emb:
            self.has_displacement_emb = True
            self.disp_emb = nn.Conv2d(2,displacement_emb_dim,1,1,0)
        else:
            self.has_displacement_emb = False
        self.local_corr_radius = local_corr_radius
        self.corr_in_other = corr_in_other
        self.no_im_B_fm = no_im_B_fm
        self.amp = amp
        self.concat_logits = concat_logits
        self.use_cosine_corr = use_cosine_corr
        self.disable_local_corr_grad = disable_local_corr_grad
        self.is_classifier = is_classifier
        self.sample_mode = sample_mode
        self.amp_dtype = amp_dtype
        
    def create_block(
        self,
        in_dim,
        out_dim,
        dw=False,
        kernel_size=5,
        bias = True,
        norm_type = nn.BatchNorm2d,
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            assert (
                out_dim % in_dim == 0
            ), "outdim must be divisible by indim for depthwise"
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
            bias=bias,
        )
        norm = norm_type(out_dim, momentum = self.bn_momentum) if norm_type is nn.BatchNorm2d else norm_type(num_channels = out_dim)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)
        
    def forward(self, x, y, flow, scale_factor = 1, logits = None):
        b,c,hs,ws = x.shape
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, enabled=self.amp, dtype=self.amp_dtype)
        with torch.autocast(autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):            
            x_hat = F.grid_sample(y, flow.permute(0, 2, 3, 1), align_corners=False, mode = self.sample_mode)
            if self.has_displacement_emb:
                im_A_coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=x.device),
                    torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=x.device),
                ), indexing='ij'
                )
                im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
                im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
                in_displacement = flow-im_A_coords
                emb_in_displacement = self.disp_emb(40/32 * scale_factor * in_displacement)
                if self.local_corr_radius:
                    if self.corr_in_other:
                        # Corr in other means take a kxk grid around the predicted coordinate in other image
                        local_corr = local_correlation(x,y,local_radius=self.local_corr_radius,flow = flow, 
                                                       sample_mode = self.sample_mode)
                    else:
                        raise NotImplementedError("Local corr in own frame should not be used.")
                    if self.no_im_B_fm:
                        x_hat = torch.zeros_like(x)
                    d = torch.cat((x, x_hat, emb_in_displacement, local_corr), dim=1)
                else:    
                    d = torch.cat((x, x_hat, emb_in_displacement), dim=1)
            else:
                if self.no_im_B_fm:
                    x_hat = torch.zeros_like(x)
                d = torch.cat((x, x_hat), dim=1)
            if self.concat_logits:
                d = torch.cat((d, logits), dim=1)
            d = self.block1(d)
            d = self.hidden_blocks(d)
        d = self.out_conv(d.float())
        displacement, certainty = d[:, :-1], d[:, -1:]
        return displacement, certainty

class CosKernel(nn.Module):  # similar to softmax kernel
    def __init__(self, T, learn_temperature=False):
        super().__init__()
        self.learn_temperature = learn_temperature
        if self.learn_temperature:
            self.T = nn.Parameter(torch.tensor(T))
        else:
            self.T = T

    def __call__(self, x, y, eps=1e-6):
        c = torch.einsum("bnd,bmd->bnm", x, y) / (
            x.norm(dim=-1)[..., None] * y.norm(dim=-1)[:, None] + eps
        )
        if self.learn_temperature:
            T = self.T.abs() + 0.01
        else:
            T = torch.tensor(self.T, device=c.device)
        K = ((c - 1.0) / T).exp()
        return K

class GP(nn.Module):
    def __init__(
        self,
        kernel,
        T=1,
        learn_temperature=False,
        only_attention=False,
        gp_dim=64,
        basis="fourier",
        covar_size=5,
        only_nearest_neighbour=False,
        sigma_noise=0.1,
        no_cov=False,
        predict_features = False,
    ):
        super().__init__()
        self.K = kernel(T=T, learn_temperature=learn_temperature)
        self.sigma_noise = sigma_noise
        self.covar_size = covar_size
        self.pos_conv = torch.nn.Conv2d(2, gp_dim, 1, 1)
        self.only_attention = only_attention
        self.only_nearest_neighbour = only_nearest_neighbour
        self.basis = basis
        self.no_cov = no_cov
        self.dim = gp_dim
        self.predict_features = predict_features

    def get_local_cov(self, cov):
        K = self.covar_size
        b, h, w, h, w = cov.shape
        hw = h * w
        cov = F.pad(cov, 4 * (K // 2,))  # pad v_q
        delta = torch.stack(
            torch.meshgrid(
                torch.arange(-(K // 2), K // 2 + 1), torch.arange(-(K // 2), K // 2 + 1),
                indexing = 'ij'),
            dim=-1,
        )
        positions = torch.stack(
            torch.meshgrid(
                torch.arange(K // 2, h + K // 2), torch.arange(K // 2, w + K // 2),
                indexing = 'ij'),
            dim=-1,
        )
        neighbours = positions[:, :, None, None, :] + delta[None, :, :]
        points = torch.arange(hw)[:, None].expand(hw, K**2)
        local_cov = cov.reshape(b, hw, h + K - 1, w + K - 1)[
            :,
            points.flatten(),
            neighbours[..., 0].flatten(),
            neighbours[..., 1].flatten(),
        ].reshape(b, h, w, K**2)
        return local_cov

    def reshape(self, x):
        return rearrange(x, "b d h w -> b (h w) d")

    def project_to_basis(self, x):
        if self.basis == "fourier":
            return torch.cos(8 * math.pi * self.pos_conv(x))
        elif self.basis == "linear":
            return self.pos_conv(x)
        else:
            raise ValueError(
                "No other bases other than fourier and linear currently im_Bed in public release"
            )

    def get_pos_enc(self, y):
        b, c, h, w = y.shape
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=y.device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=y.device),
            ),
            indexing = 'ij'
        )

        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        coarse_embedded_coords = self.project_to_basis(coarse_coords)
        return coarse_embedded_coords

    def forward(self, x, y, **kwargs):
        b, c, h1, w1 = x.shape
        b, c, h2, w2 = y.shape
        f = self.get_pos_enc(y)
        b, d, h2, w2 = f.shape
        x, y, f = self.reshape(x.float()), self.reshape(y.float()), self.reshape(f)
        K_xx = self.K(x, x)
        K_yy = self.K(y, y)
        K_xy = self.K(x, y)
        K_yx = K_xy.permute(0, 2, 1)
        sigma_noise = self.sigma_noise * torch.eye(h2 * w2, device=x.device)[None, :, :]
        with warnings.catch_warnings():
            K_yy_inv = torch.linalg.inv(K_yy + sigma_noise)

        mu_x = K_xy.matmul(K_yy_inv.matmul(f))
        mu_x = rearrange(mu_x, "b (h w) d -> b d h w", h=h1, w=w1)
        if not self.no_cov:
            cov_x = K_xx - K_xy.matmul(K_yy_inv.matmul(K_yx))
            cov_x = rearrange(cov_x, "b (h w) (r c) -> b h w r c", h=h1, w=w1, r=h1, c=w1)
            local_cov_x = self.get_local_cov(cov_x)
            local_cov_x = rearrange(local_cov_x, "b h w K -> b K h w")
            gp_feats = torch.cat((mu_x, local_cov_x), dim=1)
        else:
            gp_feats = mu_x
        return gp_feats

class Decoder(nn.Module):
    def __init__(
        self, embedding_decoder, gps, proj, conv_refiner, detach=False, scales="all", pos_embeddings = None,
        num_refinement_steps_per_scale = 1, warp_noise_std = 0.0, displacement_dropout_p = 0.0, gm_warp_dropout_p = 0.0,
        flow_upsample_mode = "bilinear", amp_dtype = torch.float16,
    ):
        super().__init__()
        self.embedding_decoder = embedding_decoder
        self.num_refinement_steps_per_scale = num_refinement_steps_per_scale
        self.gps = gps
        self.proj = proj
        self.conv_refiner = conv_refiner
        self.detach = detach
        if pos_embeddings is None:
            self.pos_embeddings = {}
        else:
            self.pos_embeddings = pos_embeddings
        if scales == "all":
            self.scales = ["32", "16", "8", "4", "2", "1"]
        else:
            self.scales = scales
        self.warp_noise_std = warp_noise_std
        self.refine_init = 4
        self.displacement_dropout_p = displacement_dropout_p
        self.gm_warp_dropout_p = gm_warp_dropout_p
        self.flow_upsample_mode = flow_upsample_mode
        self.amp_dtype = amp_dtype
        
    def get_placeholder_flow(self, b, h, w, device):
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            ),
            indexing = 'ij'
        )
        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        return coarse_coords
    
    def get_positional_embedding(self, b, h ,w, device):
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            ),
            indexing = 'ij'
        )

        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        coarse_embedded_coords = self.pos_embedding(coarse_coords)
        return coarse_embedded_coords

    def forward(self, f1, f2, gt_warp = None, gt_prob = None, upsample = False, flow = None, certainty = None, scale_factor = 1):
        coarse_scales = self.embedding_decoder.scales()
        all_scales = self.scales if not upsample else ["8", "4", "2", "1"] 
        sizes = {scale: f1[scale].shape[-2:] for scale in f1}
        h, w = sizes[1]
        b = f1[1].shape[0]
        device = f1[1].device
        coarsest_scale = int(all_scales[0])
        old_stuff = torch.zeros(
            b, self.embedding_decoder.hidden_dim, *sizes[coarsest_scale], device=f1[coarsest_scale].device
        )
        corresps = {}
        if not upsample:
            flow = self.get_placeholder_flow(b, *sizes[coarsest_scale], device)
            certainty = 0.0
        else:
            flow = F.interpolate(
                    flow,
                    size=sizes[coarsest_scale],
                    align_corners=False,
                    mode="bilinear",
                )
            certainty = F.interpolate(
                    certainty,
                    size=sizes[coarsest_scale],
                    align_corners=False,
                    mode="bilinear",
                )
        displacement = 0.0
        for new_scale in all_scales:
            ins = int(new_scale)
            corresps[ins] = {}
            f1_s, f2_s = f1[ins], f2[ins]
            if new_scale in self.proj:
                autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(f1_s.device, str(f1_s)=='cuda', self.amp_dtype)
                with torch.autocast(autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
                    if not autocast_enabled:
                        f1_s, f2_s = f1_s.to(torch.float32), f2_s.to(torch.float32)
                    f1_s, f2_s = self.proj[new_scale](f1_s), self.proj[new_scale](f2_s)

            if ins in coarse_scales:
                old_stuff = F.interpolate(
                    old_stuff, size=sizes[ins], mode="bilinear", align_corners=False
                )
                gp_posterior = self.gps[new_scale](f1_s, f2_s)
                gm_warp_or_cls, certainty, old_stuff = self.embedding_decoder(
                    gp_posterior, f1_s, old_stuff, new_scale
                )
                if self.embedding_decoder.is_classifier:
                    flow = cls_to_flow_refine(
                        gm_warp_or_cls,
                    ).permute(0,3,1,2)#gm_warp_or_cls: [B,C,H,W]
                    corresps[ins].update({"gm_cls": gm_warp_or_cls,"gm_certainty": certainty,}) if self.training else None
                else:
                    corresps[ins].update({"gm_flow": gm_warp_or_cls,"gm_certainty": certainty,}) if self.training else None
                    flow = gm_warp_or_cls.detach()
            if new_scale in self.conv_refiner:
                corresps[ins].update({"flow_pre_delta": flow}) if self.training else None
                delta_flow, delta_certainty = self.conv_refiner[new_scale](
                    f1_s, f2_s, flow, scale_factor = scale_factor, logits = certainty,
                )                    
                corresps[ins].update({"delta_flow": delta_flow,}) if self.training else None
                displacement = ins*torch.stack((delta_flow[:, 0].float() / (self.refine_init * w),
                                                delta_flow[:, 1].float() / (self.refine_init * h),),dim=1,)
                flow = flow + displacement
                certainty = (
                    certainty + delta_certainty
                )  # predict both certainty and displacement
            corresps[ins].update({
                "certainty": certainty,
                "flow": flow,             
            })
            if new_scale != "1":
                flow = F.interpolate(
                    flow,
                    size=sizes[ins // 2],
                    mode=self.flow_upsample_mode,
                )
                certainty = F.interpolate(
                    certainty,
                    size=sizes[ins // 2],
                    mode=self.flow_upsample_mode,
                )
                if self.detach:
                    flow = flow.detach()
                    certainty = certainty.detach()
            #torch.cuda.empty_cache()        
        return corresps


class Warp_Decoder(nn.Module):
    def __init__(
        self, embedding_decoder, proj, conv_refiner, detach=False, scales="all", pos_embeddings = None,
        num_refinement_steps_per_scale = 1, warp_noise_std = 0.0, displacement_dropout_p = 0.0, gm_warp_dropout_p = 0.0,
        flow_upsample_mode = "bilinear", amp_dtype = torch.float16,
    ):
        super().__init__()
        self.embedding_decoder = embedding_decoder
        self.num_refinement_steps_per_scale = num_refinement_steps_per_scale
        self.proj = proj
        self.conv_refiner = conv_refiner
        self.detach = detach
        if pos_embeddings is None:
            self.pos_embeddings = {}
        else:
            self.pos_embeddings = pos_embeddings
        if scales == "all":
            self.scales = ["32", "16", "8", "4", "2", "1"]
        else:
            self.scales = scales
        self.warp_noise_std = warp_noise_std
        self.refine_init = 4
        self.displacement_dropout_p = displacement_dropout_p
        self.gm_warp_dropout_p = gm_warp_dropout_p
        self.flow_upsample_mode = flow_upsample_mode
        self.amp_dtype = amp_dtype
        
    def get_placeholder_flow(self, b, h, w, device):
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            ),
            indexing = 'ij'
        )
        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        return coarse_coords
    
    def get_positional_embedding(self, b, h ,w, device):
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            ),
            indexing = 'ij'
        )

        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        coarse_embedded_coords = self.pos_embedding(coarse_coords)
        return coarse_embedded_coords

    def forward(self, f1, f2, gt_warp = None, gt_prob = None, upsample = False, flow = None, certainty = None, scale_factor = 1):
        coarse_scales = self.embedding_decoder.scales()
        all_scales = self.scales if not upsample else ["8", "4", "2", "1"] 
        sizes = {scale: f1[scale].shape[-2:] for scale in f1}
        h, w = sizes[1]
        b = f1[1].shape[0]
        device = f1[1].device
        coarsest_scale = int(all_scales[0])
        old_stuff = torch.zeros(
            b, self.embedding_decoder.hidden_dim, *sizes[coarsest_scale], device=f1[coarsest_scale].device
        )
        corresps = {}
        if not upsample:
            flow = self.get_placeholder_flow(b, *sizes[coarsest_scale], device)
            certainty = 0.0
        else:
            flow = F.interpolate(
                    flow,
                    size=sizes[coarsest_scale],
                    align_corners=False,
                    mode="bilinear",
                )
            certainty = F.interpolate(
                    certainty,
                    size=sizes[coarsest_scale],
                    align_corners=False,
                    mode="bilinear",
                )
        displacement = 0.0
        for new_scale in all_scales:
            ins = int(new_scale)
            corresps[ins] = {}
            f1_s, f2_s = f1[ins], f2[ins]
            if new_scale in self.proj:
                autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(f1_s.device, str(f1_s)=='cuda', self.amp_dtype)
                with torch.autocast(autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
                    if not autocast_enabled:
                        f1_s, f2_s = f1_s.to(torch.float32), f2_s.to(torch.float32)
                    f1_s, f2_s = self.proj[new_scale](f1_s), self.proj[new_scale](f2_s)

            if ins in coarse_scales:
                old_stuff = F.interpolate(
                    old_stuff, size=sizes[ins], mode="bilinear", align_corners=False
                )
                gm_warp_or_cls, certainty, old_stuff = self.embedding_decoder(
                    f1_s, f2_s, old_stuff, new_scale
                )
                if self.embedding_decoder.is_classifier:
                    flow = cls_to_flow_refine(
                        gm_warp_or_cls,
                    ).permute(0,3,1,2)#gm_warp_or_cls: [B,C,H,W]
                    corresps[ins].update({"gm_cls": gm_warp_or_cls,"gm_certainty": certainty,}) if self.training else None
                else:
                    corresps[ins].update({"gm_flow": gm_warp_or_cls,"gm_certainty": certainty,}) if self.training else None
                    flow = gm_warp_or_cls.detach()
            if new_scale in self.conv_refiner:
                corresps[ins].update({"flow_pre_delta": flow}) if self.training else None
                delta_flow, delta_certainty = self.conv_refiner[new_scale](
                    f1_s, f2_s, flow, scale_factor = scale_factor, logits = certainty,
                )                    
                corresps[ins].update({"delta_flow": delta_flow,}) if self.training else None
                displacement = ins*torch.stack((delta_flow[:, 0].float() / (self.refine_init * w),
                                                delta_flow[:, 1].float() / (self.refine_init * h),),dim=1,)
                flow = flow + displacement
                certainty = (
                    certainty + delta_certainty
                )  # predict both certainty and displacement
            corresps[ins].update({
                "certainty": certainty,
                "flow": flow,             
            })
            if new_scale != "1":
                flow = F.interpolate(
                    flow,
                    size=sizes[ins // 2],
                    mode=self.flow_upsample_mode,
                )
                certainty = F.interpolate(
                    certainty,
                    size=sizes[ins // 2],
                    mode=self.flow_upsample_mode,
                )
                if self.detach:
                    flow = flow.detach()
                    certainty = certainty.detach()
            #torch.cuda.empty_cache()        
        return corresps

class WarpHead(nn.Module):
    def __init__(
            self,
            decoder,
            sample_mode = "threshold_balanced",
    ):
        super().__init__()
        self.decoder = decoder
        self.sample_mode = sample_mode

    def forward(self, f_q_pyramid, f_s_pyramid, upsample = False, scale_factor = 1):
        corresps = self.decoder(f_q_pyramid, 
                                f_s_pyramid, 
                                upsample = upsample, 
                                scale_factor=scale_factor)
        return corresps
    
    @torch.inference_mode()
    def match(
        self,
        f_A_pyramid,
        f_B_pyramid,
        *args,
        batched=False,
        device=None,
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train(False)
        with torch.no_grad():
            hs, ws = self.h_resized, self.w_resized
            finest_scale = 1
            # Run matcher
            corresps = self.forward(f_A_pyramid, f_B_pyramid)
            
            im_A_to_im_B = corresps[finest_scale]["flow"]
            certainty = corresps[finest_scale]["certainty"]
            b, h, w = certainty.shape[:2]
            im_A_to_im_B = im_A_to_im_B.permute(
                0, 2, 3, 1
            )
            # Create im_A meshgrid
            im_A_coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                    torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
                ),
                indexing='ij'
            )
            im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
            im_A_coords = im_A_coords[None].expand(b, 2, h, w)
            certainty = certainty.sigmoid()  # logits -> probs
            im_A_coords = im_A_coords.permute(0, 2, 3, 1)
            if (im_A_to_im_B.abs() > 1).any() and True:
                wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
                certainty[wrong[:, None]] = 0
            im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
            warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
            return (
                warp,
                certainty[:, 0]
            )

    def sample(
        self,
        matches,
        certainty,
        num=10000,
    ):
        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            certainty = certainty.clone()
            certainty[certainty > upper_thresh] = 1
        matches, certainty = (
            matches.reshape(-1, 4),
            certainty.reshape(-1),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        good_samples = torch.multinomial(certainty, 
                          num_samples = min(expansion_factor*num, len(certainty)), 
                          replacement=False)
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        if "balanced" not in self.sample_mode:
            return good_matches, good_certainty
        density = kde(good_matches, std=0.1)
        p = 1 / (density+1)
        p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        balanced_samples = torch.multinomial(p, 
                          num_samples = min(num,len(good_certainty)), 
                          replacement=False)
        return good_matches[balanced_samples], good_certainty[balanced_samples]
    
    def to_pixel_coordinates(self, coords, H_A, W_A, H_B = None, W_B = None):
        if coords.shape[-1] == 2:
            return self._to_pixel_coordinates(coords, H_A, W_A) 
        
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[...,:2], coords[...,2:]
        return self._to_pixel_coordinates(kpts_A, H_A, W_A), self._to_pixel_coordinates(kpts_B, H_B, W_B)

    def _to_pixel_coordinates(self, coords, H, W):
        kpts = torch.stack((W/2 * (coords[...,0]+1), H/2 * (coords[...,1]+1)),axis=-1)
        return kpts
 
    def to_normalized_coordinates(self, coords, H_A, W_A, H_B, W_B):
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[...,:2], coords[...,2:]
        kpts_A = torch.stack((2/W_A * kpts_A[...,0] - 1, 2/H_A * kpts_A[...,1] - 1),axis=-1)
        kpts_B = torch.stack((2/W_B * kpts_B[...,0] - 1, 2/H_B * kpts_B[...,1] - 1),axis=-1)
        return kpts_A, kpts_B

    def match_keypoints(self, x_A, x_B, warp, certainty, return_tuple = True, return_inds = False):
        x_A_to_B = F.grid_sample(warp[...,-2:].permute(2,0,1)[None], x_A[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        cert_A_to_B = F.grid_sample(certainty[None,None,...], x_A[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        D = torch.cdist(x_A_to_B, x_B)
        inds_A, inds_B = torch.nonzero((D == D.min(dim=-1, keepdim = True).values) * (D == D.min(dim=-2, keepdim = True).values) * (cert_A_to_B[:,None] > self.sample_thresh), as_tuple = True)
        
        if return_tuple:
            if return_inds:
                return inds_A, inds_B
            else:
                return x_A[inds_A], x_B[inds_B]
        else:
            if return_inds:
                return torch.cat((inds_A, inds_B),dim=-1)
            else:
                return torch.cat((x_A[inds_A], x_B[inds_B]),dim=-1)

    def match_keypoints2(self, x_A, x_B, warp, certainty): 
        assert len(warp.shape) == 3 and int(warp.shape[2]) == 4, str(warp.shape)
        H,W2,_ = warp.shape
        print(H, W2)
        W = W2//2
        
        x_A_from_B = F.grid_sample(warp[:, W:, :2].permute(2,0,1)[None], x_B[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        cert_A_from_B = F.grid_sample(certainty[None, None, :, W:], x_B[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        # match in the coordinate system of A
        D = torch.cdist(x_A, x_A_from_B)
        inds_A, inds_B = torch.nonzero((D == D.min(dim=-1, keepdim = True).values) * (D == D.min(dim=-2, keepdim = True).values) * (cert_A_from_B[None,:] > 0.01), as_tuple = True)
        return inds_A, inds_B, cert_A_from_B[inds_B]

class DPT(nn.Module):
    def __init__(
        self,
        in_dim = 512,
        hidden_dim = 256,
        out_dim = 3,
        dw = False,
        kernel_size = 5,
        use_bias_block_1 = True,
        hidden_blocks=8,
        norm_type = nn.BatchNorm2d,
        amp = False,
        amp_dtype = torch.float16,
        bn_momentum = 0.1,
    ):
        super().__init__()
        self.bn_momentum = bn_momentum
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.block1 = self.create_block(
            in_dim, hidden_dim, dw=dw, kernel_size=kernel_size, bias = use_bias_block_1,
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    kernel_size=kernel_size,
                    norm_type=norm_type,
                )
                for hb in range(hidden_blocks)
            ]
        )
        self.hidden_blocks = self.hidden_blocks
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        self.amp = amp
        self.amp_dtype = amp_dtype

    def create_block(
        self,
        in_dim,
        out_dim,
        dw=False,
        kernel_size=5,
        bias = True,
        norm_type = nn.BatchNorm2d,
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            assert (
                out_dim % in_dim == 0
            ), "outdim must be divisible by indim for depthwise"
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
            bias=bias,
        )
        norm = norm_type(out_dim, momentum = self.bn_momentum) if norm_type is nn.BatchNorm2d else norm_type(num_channels = out_dim)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)

    def forward(self, f):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(f.device, enabled=self.amp, dtype=self.amp_dtype)
        with torch.autocast(autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):  
            out = self.block1(f)
            out = self.hidden_blocks(out)
        out = self.out_conv(out.float())
        pts, certainty = out[:, :-1], out[:, -1:]
        return pts, certainty

class DPTRefiner(nn.Module):
    def __init__(
        self,
        in_dim=6,
        hidden_dim=16,
        out_dim=4,
        dw=False,
        kernel_size=5,
        hidden_blocks=8,
        displacement_emb_dim = 128,
        amp = False,
        use_bias_block_1 = True,
        sample_mode = "bilinear",
        norm_type = nn.BatchNorm2d,
        bn_momentum = 0.1,
        amp_dtype = torch.float16,
        concat_logits = False,
    ):
        super().__init__()
        self.bn_momentum = bn_momentum
        self.block1 = self.create_block(
            in_dim, hidden_dim, dw=dw, kernel_size=kernel_size, bias = use_bias_block_1,
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    kernel_size=kernel_size,
                    norm_type=norm_type,
                )
                for hb in range(hidden_blocks)
            ]
        )
        self.hidden_blocks = self.hidden_blocks
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        self.disp_emb = nn.Conv2d(3, displacement_emb_dim, 1, 1, 0)
        self.amp = amp
        self.sample_mode = sample_mode
        self.amp_dtype = amp_dtype
        self.concat_logits = concat_logits
    
    def create_block(
        self,
        in_dim,
        out_dim,
        dw=False,
        kernel_size=5,
        bias = True,
        norm_type = nn.BatchNorm2d,
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            assert (
                out_dim % in_dim == 0
            ), "outdim must be divisible by indim for depthwise"
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
            bias=bias,
        )
        norm = norm_type(out_dim, momentum = self.bn_momentum) if norm_type is nn.BatchNorm2d else norm_type(num_channels = out_dim)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)
        
    def forward(self, x, y, pts, logits = None):
        emb_pts = self.disp_emb(pts)
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, enabled=self.amp, dtype=self.amp_dtype)
        with torch.autocast(autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):            
            d = torch.cat((x, y, emb_pts), dim=1)
            if self.concat_logits:
                d = torch.cat((d, logits), dim=1)
            d = self.block1(d)
            d = self.hidden_blocks(d)
        d = self.out_conv(d.float())
        displacement, certainty = d[:, :-1], d[:, -1:]
        return displacement, certainty
 
class Decoder_dpt(nn.Module):
    def __init__(
        self, embedding_decoder, dpt1, dpt2, dpt_refiner1, dpt_refiner2, gps, proj, conv_refiner, detach=False, scales="all", pos_embeddings = None,
        num_refinement_steps_per_scale = 1, warp_noise_std = 0.0, displacement_dropout_p = 0.0, gm_warp_dropout_p = 0.0,
        flow_upsample_mode = "bilinear", amp_dtype = torch.float16,
    ):
        super().__init__()
        self.embedding_decoder = embedding_decoder
        self.num_refinement_steps_per_scale = num_refinement_steps_per_scale
        self.gps = gps
        self.proj = proj
        self.conv_refiner = conv_refiner
        self.detach = detach
        if pos_embeddings is None:
            self.pos_embeddings = {}
        else:
            self.pos_embeddings = pos_embeddings
        if scales == "all":
            self.scales = ["32", "16", "8", "4", "2", "1"]
        else:
            self.scales = scales
        self.warp_noise_std = warp_noise_std
        self.refine_init = 4
        self.displacement_dropout_p = displacement_dropout_p
        self.gm_warp_dropout_p = gm_warp_dropout_p
        self.flow_upsample_mode = flow_upsample_mode
        self.amp_dtype = amp_dtype
        self.dpt1 = dpt1
        self.dpt_refiner1 = dpt_refiner1
        self.dpt2 = dpt2
        self.dpt_refiner2 = dpt_refiner2
        
    def get_placeholder_flow(self, b, h, w, device):
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            ),
            indexing = 'ij'
        )
        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        return coarse_coords
    
    def get_positional_embedding(self, b, h ,w, device):
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            ),
            indexing = 'ij'
        )

        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        coarse_embedded_coords = self.pos_embedding(coarse_coords)
        return coarse_embedded_coords

    def forward(self, f1, f2, gt_warp = None, gt_prob = None, upsample = False, flow = None, certainty = None, pts1 = None, pts_conf1 = None, pts2 = None, pts_conf2 = None, scale_factor = 1):
        coarse_scales = self.embedding_decoder.scales()
        all_scales = self.scales if not upsample else ["8", "4", "2", "1"] 
        sizes = {scale: f1[scale].shape[-2:] for scale in f1}
        h, w = sizes[1]
        b = f1[1].shape[0]
        device = f1[1].device
        coarsest_scale = int(all_scales[0])
        old_stuff = torch.zeros(
            b, self.embedding_decoder.hidden_dim, *sizes[coarsest_scale], device=f1[coarsest_scale].device
        )
        old_pts1 = torch.zeros(
            b, 3, *sizes[coarsest_scale], device=f1[coarsest_scale].device
        )
        old_pts2 = torch.zeros(
            b, 3, *sizes[coarsest_scale], device=f1[coarsest_scale].device
        )
        corresps = {}
        if not upsample:
            flow = self.get_placeholder_flow(b, *sizes[coarsest_scale], device)
            certainty = 0.0
            pts1 = old_pts1
            pts_conf1 = 0.0
            pts2 = old_pts2
            pts_conf2 = 0.0
        else:
            flow = F.interpolate(
                    flow,
                    size=sizes[coarsest_scale],
                    align_corners=False,
                    mode="bilinear",
                )
            certainty = F.interpolate(
                    certainty,
                    size=sizes[coarsest_scale],
                    align_corners=False,
                    mode="bilinear",
                )
            pts1 = F.interpolate(
                pts1,
                size=sizes[coarsest_scale],
                align_corners=False,
                mode="bilinear",
            )
            pts_conf1 = F.interpolate(
                pts_conf1,
                size=sizes[coarsest_scale],
                align_corners=False,
                mode="bilinear",
            )
            pts2 = F.interpolate(
                pts2,
                size=sizes[coarsest_scale],
                align_corners=False,
                mode="bilinear",
            )
            pts_conf2 = F.interpolate(
                pts_conf2,
                size=sizes[coarsest_scale],
                align_corners=False,
                mode="bilinear",
            )
        displacement = 0.0
        for new_scale in all_scales:
            ins = int(new_scale)
            corresps[ins] = {}
            f1_s, f2_s = f1[ins], f2[ins]
            if new_scale in self.proj:
                autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(f1_s.device, str(f1_s)=='cuda', self.amp_dtype)
                with torch.autocast(autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
                    if not autocast_enabled:
                        f1_s, f2_s = f1_s.to(torch.float32), f2_s.to(torch.float32)
                    f1_s, f2_s = self.proj[new_scale](f1_s), self.proj[new_scale](f2_s)

            if ins in coarse_scales:
                old_stuff = F.interpolate(
                    old_stuff, size=sizes[ins], mode="bilinear", align_corners=False
                )
                gp_posterior = self.gps[new_scale](f1_s, f2_s)
                gm_warp_or_cls, certainty, old_stuff = self.embedding_decoder(
                    gp_posterior, f1_s, old_stuff, new_scale
                )
                
                if self.embedding_decoder.is_classifier:
                    flow = cls_to_flow_refine(
                        gm_warp_or_cls,
                    ).permute(0,3,1,2)#gm_warp_or_cls: [B,C,H,W]
                    corresps[ins].update({"gm_cls": gm_warp_or_cls,"gm_certainty": certainty,}) if self.training else None
                else:
                    corresps[ins].update({"gm_flow": gm_warp_or_cls,"gm_certainty": certainty,}) if self.training else None
                    flow = gm_warp_or_cls.detach()
                pts1, pts_conf1 = self.dpt1(f1_s)
                pts2, pts_conf2 = self.dpt2(f2_s)
                corresps[ins].update({"pts1": pts1, "pts_conf1": pts_conf1, "pts2": pts2, "pts_conf2": pts_conf2}) if self.training else None
                    
            if new_scale in self.conv_refiner:
                corresps[ins].update({"flow_pre_delta": flow}) if self.training else None
                delta_flow, delta_certainty = self.conv_refiner[new_scale](
                    f1_s, f2_s, flow, scale_factor = scale_factor, logits = certainty,
                )                    
                corresps[ins].update({"delta_flow": delta_flow,}) if self.training else None
                displacement = ins*torch.stack((delta_flow[:, 0].float() / (self.refine_init * w),
                                                delta_flow[:, 1].float() / (self.refine_init * h),),dim=1,)
                flow = flow + displacement
                certainty = (
                    certainty + delta_certainty
                )  # predict both certainty and displacement
            if new_scale in self.dpt_refiner1:
                corresps[ins].update({"pts1_pre_delta": pts1, "pts2_pre_delta": pts2}) if self.training else None
                delta_pts1, delta_conf1 = self.dpt_refiner1[new_scale](
                    f1_s, f2_s, pts1, logits = pts_conf1
                )
                delta_pts2, delta_conf2 = self.dpt_refiner2[new_scale](
                    f2_s, f1_s, pts2, logits = pts_conf2
                )
                pts1 = pts1 + delta_pts1
                pts2 = pts2 + delta_pts2
                pts_conf1 = (pts_conf1 + delta_conf1)
                pts_conf2 = (pts_conf2 + delta_conf2)

            corresps[ins].update({
                "certainty": certainty,
                "flow": flow,    
                "pts1": pts1,
                "pts2": pts2,
                "pts_conf1": pts_conf1,
                "pts_conf2": pts_conf2,         
            })
            if new_scale != "1":
                flow = F.interpolate(
                    flow,
                    size=sizes[ins // 2],
                    mode=self.flow_upsample_mode,
                )
                certainty = F.interpolate(
                    certainty,
                    size=sizes[ins // 2],
                    mode=self.flow_upsample_mode,
                )
                pts1 = F.interpolate(
                    pts1,
                    size=sizes[ins // 2],
                    mode=self.flow_upsample_mode,
                )
                pts_conf1 = F.interpolate(
                    pts_conf1,
                    size=sizes[ins // 2],
                    mode=self.flow_upsample_mode,
                )
                pts2 = F.interpolate(
                    pts2,
                    size=sizes[ins // 2],
                    mode=self.flow_upsample_mode,
                )
                pts_conf2 = F.interpolate(
                    pts_conf2,
                    size=sizes[ins // 2],
                    mode=self.flow_upsample_mode,
                )
                if self.detach:
                    flow = flow.detach()
                    certainty = certainty.detach()
                    pts1 = pts1.detach()
                    pts_conf1 = pts_conf1.detach()
                    pts2 = pts2.detach()
                    pts_conf2 = pts_conf2.detach()
            #torch.cuda.empty_cache()        
        return corresps

class Decoder_only_dpt(nn.Module):
    def __init__(
        self, dpt1, dpt2, dpt_refiner1, dpt_refiner2, gps, proj, detach=False, scales="all", pos_embeddings = None,
        num_refinement_steps_per_scale = 1, warp_noise_std = 0.0, displacement_dropout_p = 0.0, gm_warp_dropout_p = 0.0,
        flow_upsample_mode = "bilinear", amp_dtype = torch.float16,
    ):
        super().__init__()
        self.num_refinement_steps_per_scale = num_refinement_steps_per_scale
        self.gps = gps
        self.proj = proj
        self.detach = detach
        if pos_embeddings is None:
            self.pos_embeddings = {}
        else:
            self.pos_embeddings = pos_embeddings
        if scales == "all":
            self.scales = ["32", "16", "8", "4", "2", "1"]
        else:
            self.scales = scales
        self.warp_noise_std = warp_noise_std
        self.refine_init = 4
        self.displacement_dropout_p = displacement_dropout_p
        self.gm_warp_dropout_p = gm_warp_dropout_p
        self.flow_upsample_mode = flow_upsample_mode
        self.amp_dtype = amp_dtype
        self.dpt1 = dpt1
        self.dpt_refiner1 = dpt_refiner1
        self.dpt2 = dpt2
        self.dpt_refiner2 = dpt_refiner2
        
    def get_placeholder_flow(self, b, h, w, device):
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            ),
            indexing = 'ij'
        )
        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        return coarse_coords
    
    def get_positional_embedding(self, b, h ,w, device):
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            ),
            indexing = 'ij'
        )

        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        coarse_embedded_coords = self.pos_embedding(coarse_coords)
        return coarse_embedded_coords

    def forward(self, f1, f2, gt_warp = None, gt_prob = None, upsample = False, flow = None, certainty = None, pts1 = None, pts_conf1 = None, pts2 = None, pts_conf2 = None, scale_factor = 1):
        coarse_scales = 16
        all_scales = self.scales if not upsample else ["8", "4", "2", "1"] 
        sizes = {scale: f1[scale].shape[-2:] for scale in f1}
        h, w = sizes[1]
        b = f1[1].shape[0]
        device = f1[1].device
        coarsest_scale = int(all_scales[0])
        
        old_pts1 = torch.zeros(
            b, 3, *sizes[coarsest_scale], device=f1[coarsest_scale].device
        )
        old_pts2 = torch.zeros(
            b, 3, *sizes[coarsest_scale], device=f1[coarsest_scale].device
        )
        corresps = {}
        if not upsample:
            pts1 = old_pts1
            pts_conf1 = 0.0
            pts2 = old_pts2
            pts_conf2 = 0.0
        else:
            pts1 = F.interpolate(
                pts1,
                size=sizes[coarsest_scale],
                align_corners=False,
                mode="bilinear",
            )
            pts_conf1 = F.interpolate(
                pts_conf1,
                size=sizes[coarsest_scale],
                align_corners=False,
                mode="bilinear",
            )
            pts2 = F.interpolate(
                pts2,
                size=sizes[coarsest_scale],
                align_corners=False,
                mode="bilinear",
            )
            pts_conf2 = F.interpolate(
                pts_conf2,
                size=sizes[coarsest_scale],
                align_corners=False,
                mode="bilinear",
            )
        for new_scale in all_scales:
            ins = int(new_scale)
            corresps[ins] = {}
            f1_s, f2_s = f1[ins], f2[ins]
            if new_scale in self.proj:
                autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(f1_s.device, str(f1_s)=='cuda', self.amp_dtype)
                with torch.autocast(autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
                    if not autocast_enabled:
                        f1_s, f2_s = f1_s.to(torch.float32), f2_s.to(torch.float32)
                    f1_s, f2_s = self.proj[new_scale](f1_s), self.proj[new_scale](f2_s)

            if ins in coarse_scales:
                pts1, pts_conf1 = self.dpt1(f1_s)
                pts2, pts_conf2 = self.dpt2(f2_s)
                corresps[ins].update({"pts1": pts1, "pts_conf1": pts_conf1, "pts2": pts2, "pts_conf2": pts_conf2}) if self.training else None
                    
            if new_scale in self.dpt_refiner1:
                corresps[ins].update({"pts1_pre_delta": pts1, "pts2_pre_delta": pts2}) if self.training else None
                delta_pts1, delta_conf1 = self.dpt_refiner1[new_scale](
                    f1_s, f2_s, pts1, logits = pts_conf1
                )
                delta_pts2, delta_conf2 = self.dpt_refiner2[new_scale](
                    f2_s, f1_s, pts2, logits = pts_conf2
                )
                pts1 = pts1 + delta_pts1
                pts2 = pts2 + delta_pts2
                pts_conf1 = (pts_conf1 + delta_conf1)
                pts_conf2 = (pts_conf2 + delta_conf2)

            corresps[ins].update({   
                "pts1": pts1,
                "pts2": pts2,
                "pts_conf1": pts_conf1,
                "pts_conf2": pts_conf2,         
            })
            if new_scale != "1":
                certainty = F.interpolate(
                    certainty,
                    size=sizes[ins // 2],
                    mode=self.flow_upsample_mode,
                )
                pts1 = F.interpolate(
                    pts1,
                    size=sizes[ins // 2],
                    mode=self.flow_upsample_mode,
                )
                pts_conf1 = F.interpolate(
                    pts_conf1,
                    size=sizes[ins // 2],
                    mode=self.flow_upsample_mode,
                )
                pts2 = F.interpolate(
                    pts2,
                    size=sizes[ins // 2],
                    mode=self.flow_upsample_mode,
                )
                pts_conf2 = F.interpolate(
                    pts_conf2,
                    size=sizes[ins // 2],
                    mode=self.flow_upsample_mode,
                )
                if self.detach:
                    pts1 = pts1.detach()
                    pts_conf1 = pts_conf1.detach()
                    pts2 = pts2.detach()
                    pts_conf2 = pts_conf2.detach()
            #torch.cuda.empty_cache()        
        return corresps

class RegressionMatcher(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        h=448,
        w=448,
        sample_mode = "threshold_balanced",
        upsample_preds = False,
        symmetric = False,
        name = None,
        attenuate_cert = None,
        shift_range_lat = 20,
        shift_range_lon = 20
    ):
        super().__init__()
        self.attenuate_cert = attenuate_cert
        self.encoder = encoder
        self.decoder = decoder
        self.name = name
        self.w_resized = w
        self.h_resized = h
        self.og_transforms = get_tuple_transform_ops(resize=None, normalize=True)
        self.sample_mode = sample_mode
        self.upsample_preds = upsample_preds
        self.upsample_res = (14*16*6, 14*16*6)
        self.symmetric = symmetric
        self.sample_thresh = 0.05
            
    def get_output_resolution(self):
        if not self.upsample_preds:
            return self.h_resized, self.w_resized
        else:
            return self.upsample_res
    
    def extract_backbone_features(self, batch, batched = True, upsample = False):
        x_q = batch["im_A"]
        x_s = batch["im_B"]
        if batched:
            X = torch.cat((x_q, x_s), dim = 0)
            feature_pyramid = self.encoder(X, upsample = upsample)
        else:
            feature_pyramid = self.encoder(x_q, upsample = upsample), self.encoder(x_s, upsample = upsample)
        return feature_pyramid

    def sample(
        self,
        matches,
        certainty,
        num=10000,
    ):
        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            certainty = certainty.clone()
            certainty[certainty > upper_thresh] = 1
        matches, certainty = (
            matches.reshape(-1, 4),
            certainty.reshape(-1),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        good_samples = torch.multinomial(certainty, 
                          num_samples = min(expansion_factor*num, len(certainty)), 
                          replacement=False)
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        if "balanced" not in self.sample_mode:
            return good_matches, good_certainty
        density = kde(good_matches, std=0.1)
        p = 1 / (density+1)
        p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        balanced_samples = torch.multinomial(p, 
                          num_samples = min(num,len(good_certainty)), 
                          replacement=False)
        return good_matches[balanced_samples], good_certainty[balanced_samples]

    def forward(self, batch, batched = True, upsample = False, scale_factor = 1):
        feature_pyramid = self.extract_backbone_features(batch, batched=batched, upsample = upsample)
        if batched:
            f_q_pyramid = {
                scale: f_scale.chunk(2)[0] for scale, f_scale in feature_pyramid.items()
            }
            f_s_pyramid = {
                scale: f_scale.chunk(2)[1] for scale, f_scale in feature_pyramid.items()
            }
        else:
            f_q_pyramid, f_s_pyramid = feature_pyramid
        corresps = self.decoder(f_q_pyramid, 
                                f_s_pyramid, 
                                upsample = upsample, 
                                **(batch["corresps"] if "corresps" in batch else {}),
                                scale_factor=scale_factor)
        
        return corresps
    
    def forward_symmetric(self, batch, batched = True, upsample = False, scale_factor = 1):
        feature_pyramid = self.extract_backbone_features(batch, batched = batched, upsample = upsample)
        f_q_pyramid = feature_pyramid
        f_s_pyramid = {
            scale: torch.cat((f_scale.chunk(2)[1], f_scale.chunk(2)[0]), dim = 0)
            for scale, f_scale in feature_pyramid.items()
        }
        corresps = self.decoder(f_q_pyramid, 
                                f_s_pyramid, 
                                upsample = upsample, 
                                **(batch["corresps"] if "corresps" in batch else {}),
                                scale_factor=scale_factor)
        return corresps
    
    def conf_from_fb_consistency(self, flow_forward, flow_backward, th = 2):
        # assumes that flow forward is of shape (..., H, W, 2)
        has_batch = False
        if len(flow_forward.shape) == 3:
            flow_forward, flow_backward = flow_forward[None], flow_backward[None]
        else:
            has_batch = True
        H,W = flow_forward.shape[-3:-1]
        th_n = 2 * th / max(H,W)
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1 + 1 / W, 1 - 1 / W, W), 
            torch.linspace(-1 + 1 / H, 1 - 1 / H, H), indexing = "xy"),
                             dim = -1).to(flow_forward.device)
        coords_fb = F.grid_sample(
            flow_backward.permute(0, 3, 1, 2), 
            flow_forward, 
            align_corners=False, mode="bilinear").permute(0, 2, 3, 1)
        diff = (coords - coords_fb).norm(dim=-1)
        in_th = (diff < th_n).float()
        if not has_batch:
            in_th = in_th[0]
        return in_th
         
    def to_pixel_coordinates(self, coords, H_A, W_A, H_B = None, W_B = None):
        if coords.shape[-1] == 2:
            return self._to_pixel_coordinates(coords, H_A, W_A) 
        
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[...,:2], coords[...,2:]
        return self._to_pixel_coordinates(kpts_A, H_A, W_A), self._to_pixel_coordinates(kpts_B, H_B, W_B)

    def _to_pixel_coordinates(self, coords, H, W):
        kpts = torch.stack((W/2 * (coords[...,0]+1), H/2 * (coords[...,1]+1)),axis=-1)
        return kpts
 
    def to_normalized_coordinates(self, coords, H_A, W_A, H_B, W_B):
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[...,:2], coords[...,2:]
        kpts_A = torch.stack((2/W_A * kpts_A[...,0] - 1, 2/H_A * kpts_A[...,1] - 1),axis=-1)
        kpts_B = torch.stack((2/W_B * kpts_B[...,0] - 1, 2/H_B * kpts_B[...,1] - 1),axis=-1)
        return kpts_A, kpts_B

    def match_keypoints(self, x_A, x_B, warp, certainty, return_tuple = True, return_inds = False):
        x_A_to_B = F.grid_sample(warp[...,-2:].permute(2,0,1)[None], x_A[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        cert_A_to_B = F.grid_sample(certainty[None,None,...], x_A[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        D = torch.cdist(x_A_to_B, x_B)
        inds_A, inds_B = torch.nonzero((D == D.min(dim=-1, keepdim = True).values) * (D == D.min(dim=-2, keepdim = True).values) * (cert_A_to_B[:,None] > self.sample_thresh), as_tuple = True)
        
        if return_tuple:
            if return_inds:
                return inds_A, inds_B
            else:
                return x_A[inds_A], x_B[inds_B]
        else:
            if return_inds:
                return torch.cat((inds_A, inds_B),dim=-1)
            else:
                return torch.cat((x_A[inds_A], x_B[inds_B]),dim=-1)

    def match_keypoints2(self, x_A, x_B, warp, certainty):
        
        assert len(warp.shape) == 3 and int(warp.shape[2]) == 4, str(warp.shape)
        H,W2,_ = warp.shape
        print(H, W2)
        W = W2//2
        
        x_A_from_B = F.grid_sample(warp[:, W:, :2].permute(2,0,1)[None], x_B[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        cert_A_from_B = F.grid_sample(certainty[None, None, :, W:], x_B[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        # match in the coordinate system of A
        D = torch.cdist(x_A, x_A_from_B)
        inds_A, inds_B = torch.nonzero((D == D.min(dim=-1, keepdim = True).values) * (D == D.min(dim=-2, keepdim = True).values) * (cert_A_from_B[None,:] > 0.01), as_tuple = True)
        return inds_A, inds_B, cert_A_from_B[inds_B]
        
    @torch.inference_mode()
    def match(
        self,
        im_A_input,
        im_B_input,
        *args,
        batched=False,
        device=None,
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Check if inputs are file paths or already loaded images
        if isinstance(im_A_input, (str, os.PathLike)):
            im_A = Image.open(im_A_input).convert("RGB")
        else:
            im_A = im_A_input

        if isinstance(im_B_input, (str, os.PathLike)):
            im_B = Image.open(im_B_input).convert("RGB")
        else:
            im_B = im_B_input

        symmetric = self.symmetric
        self.train(False)
        with torch.no_grad():
            if not batched:
                b = 1
                w, h = im_A.size
                w2, h2 = im_B.size
                # Get images in good format
                ws = self.w_resized
                hs = self.h_resized

                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True, clahe=False
                )
                im_A, im_B = test_transform((im_A, im_B))
                batch = {"im_A": im_A[None].to(device), "im_B": im_B[None].to(device)}
            else:
                b, c, h, w = im_A.shape
                b, c, h2, w2 = im_B.shape
                assert w == w2 and h == h2, "For batched images we assume same size"
                batch = {"im_A": im_A.to(device), "im_B": im_B.to(device)}
                if h != self.h_resized or self.w_resized != w:
                    warn("Model resolution and batch resolution differ, may produce unexpected results")
                hs, ws = h, w
            finest_scale = 1
            # Run matcher
            if symmetric:
                corresps = self.forward_symmetric(batch)
            else:
                corresps = self.forward(batch, batched=True)

            if self.upsample_preds:
                hs, ws = self.upsample_res

            if self.attenuate_cert:
                low_res_certainty = F.interpolate(
                    corresps[16]["certainty"], size=(hs, ws), align_corners=False, mode="bilinear"
                )
                cert_clamp = 0
                factor = 0.5
                low_res_certainty = factor * low_res_certainty * (low_res_certainty < cert_clamp)

            if self.upsample_preds:
                finest_corresps = corresps[finest_scale]
                torch.cuda.empty_cache()
                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True
                )
                if isinstance(im_A_input, (str, os.PathLike)):
                    im_A, im_B = test_transform(
                        (Image.open(im_A_input).convert('RGB'), Image.open(im_B_input).convert('RGB')))
                else:
                    im_A, im_B = test_transform((im_A_input, im_B_input))

                im_A, im_B = im_A[None].to(device), im_B[None].to(device)
                scale_factor = math.sqrt(self.upsample_res[0] * self.upsample_res[1] / (self.w_resized * self.h_resized))
                batch = {"im_A": im_A, "im_B": im_B, "corresps": finest_corresps}
                if symmetric:
                    corresps = self.forward_symmetric(batch, upsample=True, batched=True, scale_factor=scale_factor)
                else:
                    corresps = self.forward(batch, batched=True, upsample=True, scale_factor=scale_factor)

            im_A_to_im_B = corresps[finest_scale]["flow"]
            certainty = corresps[finest_scale]["certainty"] - (low_res_certainty if self.attenuate_cert else 0)
            if finest_scale != 1:
                im_A_to_im_B = F.interpolate(
                    im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
                )
                certainty = F.interpolate(
                    certainty, size=(hs, ws), align_corners=False, mode="bilinear"
                )
            im_A_to_im_B = im_A_to_im_B.permute(
                0, 2, 3, 1
            )
            # Create im_A meshgrid
            im_A_coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                    torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
                ),
                indexing='ij'
            )
            im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
            im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
            certainty = certainty.sigmoid()  # logits -> probs
            im_A_coords = im_A_coords.permute(0, 2, 3, 1)
            if (im_A_to_im_B.abs() > 1).any() and True:
                wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
                certainty[wrong[:, None]] = 0
            im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
            if symmetric:
                A_to_B, B_to_A = im_A_to_im_B.chunk(2)
                q_warp = torch.cat((im_A_coords, A_to_B), dim=-1)
                im_B_coords = im_A_coords
                s_warp = torch.cat((B_to_A, im_B_coords), dim=-1)
                warp = torch.cat((q_warp, s_warp), dim=2)
                certainty = torch.cat(certainty.chunk(2), dim=3)
            else:
                warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
            if batched:
                return (
                    warp,
                    certainty[:, 0]
                )
            else:
                return (
                    warp[0],
                    certainty[0, 0],
                )

    def visualize_warp(self, warp, certainty, im_A = None, im_B = None, 
                       im_A_path = None, im_B_path = None, device = "cuda", symmetric = True, save_path = None, unnormalize = False):
        #assert symmetric == True, "Currently assuming bidirectional warp, might update this if someone complains ;)"
        H,W2,_ = warp.shape
        W = W2//2 if symmetric else W2
        if im_A is None:
            from PIL import Image
            im_A, im_B = Image.open(im_A_path).convert("RGB"), Image.open(im_B_path).convert("RGB")
        if not isinstance(im_A, torch.Tensor):
            im_A = im_A.resize((W,H))
            im_B = im_B.resize((W,H))    
            x_B = (torch.tensor(np.array(im_B)) / 255).to(device).permute(2, 0, 1)
            if symmetric:
                x_A = (torch.tensor(np.array(im_A)) / 255).to(device).permute(2, 0, 1)
        else:
            if symmetric:
                x_A = im_A
            x_B = im_B
        im_A_transfer_rgb = F.grid_sample(
        x_B[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
        )[0]
        if symmetric:
            im_B_transfer_rgb = F.grid_sample(
            x_A[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
            )[0]
            warp_im = torch.cat((im_A_transfer_rgb,im_B_transfer_rgb),dim=2)
            white_im = torch.ones((H,2*W),device=device)
        else:
            warp_im = im_A_transfer_rgb
            white_im = torch.ones((H, W), device = device)
        vis_im = certainty * warp_im + (1 - certainty) * white_im
        if save_path is not None:
            from romatch.utils import tensor_to_pil
            tensor_to_pil(vis_im, unnormalize=unnormalize).save(save_path)
        return vis_im

class RegressionMatcher_adapter(nn.Module):
    def __init__(
        self,
        encoder,
        adapter,
        decoder,
        h=448,
        w=448,
        sample_mode = "threshold_balanced",
        upsample_preds = False,
        symmetric = False,
        name = None,
        attenuate_cert = None,
    ):
        super().__init__()
        self.attenuate_cert = attenuate_cert
        self.encoder = encoder
        self.adapter = adapter
        self.decoder = decoder
        self.name = name
        self.w_resized = w
        self.h_resized = h
        self.og_transforms = get_tuple_transform_ops(resize=None, normalize=True)
        self.sample_mode = sample_mode
        self.upsample_preds = upsample_preds
        self.upsample_res = (14*16*6, 14*16*6)
        self.symmetric = symmetric
        self.sample_thresh = 0.05
            
    def get_output_resolution(self):
        if not self.upsample_preds:
            return self.h_resized, self.w_resized
        else:
            return self.upsample_res
    
    def extract_backbone_features(self, batch, batched = True, upsample = False):
        x_q = batch["im_A"]
        x_s = batch["im_B"]
        if batched:
            X = torch.cat((x_q, x_s), dim = 0)
            feature_pyramid = self.encoder(X, upsample = upsample)
        else:
            feature_pyramid = self.encoder(x_q, upsample = upsample), self.encoder(x_s, upsample = upsample)
        return feature_pyramid


    def sample(
        self,
        matches,
        certainty,
        num=10000,
    ):
        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            certainty = certainty.clone()
            certainty[certainty > upper_thresh] = 1
        matches, certainty = (
            matches.reshape(-1, 4),
            certainty.reshape(-1),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        good_samples = torch.multinomial(certainty, 
                          num_samples = min(expansion_factor*num, len(certainty)), 
                          replacement=False)
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        if "balanced" not in self.sample_mode:
            return good_matches, good_certainty
        density = kde(good_matches, std=0.1)
        p = 1 / (density+1)
        p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        balanced_samples = torch.multinomial(p, 
                          num_samples = min(num,len(good_certainty)), 
                          replacement=False)
        return good_matches[balanced_samples], good_certainty[balanced_samples]

    def forward(self, batch, end2end=False, batched = True, upsample = False, scale_factor = 1):
        feature_pyramid = self.extract_backbone_features(batch, batched=batched, upsample = upsample)
        if batched:
            f_q_pyramid = {
                scale: f_scale.chunk(2)[0] for scale, f_scale in feature_pyramid.items()
            }
            f_s_pyramid = {
                scale: f_scale.chunk(2)[1] for scale, f_scale in feature_pyramid.items()
            }
        else:
            f_q_pyramid, f_s_pyramid = feature_pyramid
        f_q_pyramid = self.adapter(f_q_pyramid, batch['domainid'])
        f_s_pyramid = self.adapter(f_s_pyramid, batch['domainid'])
        corresps = self.decoder(f_q_pyramid, 
                                f_s_pyramid, 
                                upsample = upsample, 
                                **(batch["corresps"] if "corresps" in batch else {}),
                                scale_factor=scale_factor)
        
        return corresps

    def forward_symmetric(self, batch, batched = True, upsample = False, scale_factor = 1):
        feature_pyramid = self.extract_backbone_features(batch, batched = batched, upsample = upsample)
        f_q_pyramid = feature_pyramid
        f_s_pyramid = {
            scale: torch.cat((f_scale.chunk(2)[1], f_scale.chunk(2)[0]), dim = 0)
            for scale, f_scale in feature_pyramid.items()
        }
        f_q_pyramid = self.adapter(f_q_pyramid, batch['domainid'])
        f_s_pyramid = self.adapter(f_s_pyramid, batch['domainid'])
        corresps = self.decoder(f_q_pyramid, 
                                f_s_pyramid, 
                                upsample = upsample, 
                                **(batch["corresps"] if "corresps" in batch else {}),
                                scale_factor=scale_factor)
        return corresps
    
    def conf_from_fb_consistency(self, flow_forward, flow_backward, th = 2):
        # assumes that flow forward is of shape (..., H, W, 2)
        has_batch = False
        if len(flow_forward.shape) == 3:
            flow_forward, flow_backward = flow_forward[None], flow_backward[None]
        else:
            has_batch = True
        H,W = flow_forward.shape[-3:-1]
        th_n = 2 * th / max(H,W)
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1 + 1 / W, 1 - 1 / W, W), 
            torch.linspace(-1 + 1 / H, 1 - 1 / H, H), indexing = "xy"),
                             dim = -1).to(flow_forward.device)
        coords_fb = F.grid_sample(
            flow_backward.permute(0, 3, 1, 2), 
            flow_forward, 
            align_corners=False, mode="bilinear").permute(0, 2, 3, 1)
        diff = (coords - coords_fb).norm(dim=-1)
        in_th = (diff < th_n).float()
        if not has_batch:
            in_th = in_th[0]
        return in_th
         
    def to_pixel_coordinates(self, coords, H_A, W_A, H_B = None, W_B = None):
        if coords.shape[-1] == 2:
            return self._to_pixel_coordinates(coords, H_A, W_A) 
        
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[...,:2], coords[...,2:]
        return self._to_pixel_coordinates(kpts_A, H_A, W_A), self._to_pixel_coordinates(kpts_B, H_B, W_B)

    def _to_pixel_coordinates(self, coords, H, W):
        kpts = torch.stack((W/2 * (coords[...,0]+1), H/2 * (coords[...,1]+1)),axis=-1)
        return kpts
 
    def to_normalized_coordinates(self, coords, H_A, W_A, H_B, W_B):
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[...,:2], coords[...,2:]
        kpts_A = torch.stack((2/W_A * kpts_A[...,0] - 1, 2/H_A * kpts_A[...,1] - 1),axis=-1)
        kpts_B = torch.stack((2/W_B * kpts_B[...,0] - 1, 2/H_B * kpts_B[...,1] - 1),axis=-1)
        return kpts_A, kpts_B

    def match_keypoints(self, x_A, x_B, warp, certainty, return_tuple = True, return_inds = False):
        x_A_to_B = F.grid_sample(warp[...,-2:].permute(2,0,1)[None], x_A[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        cert_A_to_B = F.grid_sample(certainty[None,None,...], x_A[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        D = torch.cdist(x_A_to_B, x_B)
        inds_A, inds_B = torch.nonzero((D == D.min(dim=-1, keepdim = True).values) * (D == D.min(dim=-2, keepdim = True).values) * (cert_A_to_B[:,None] > self.sample_thresh), as_tuple = True)
        
        if return_tuple:
            if return_inds:
                return inds_A, inds_B
            else:
                return x_A[inds_A], x_B[inds_B]
        else:
            if return_inds:
                return torch.cat((inds_A, inds_B),dim=-1)
            else:
                return torch.cat((x_A[inds_A], x_B[inds_B]),dim=-1)
    def match_keypoints2(self, x_A, x_B, warp, certainty):
        assert len(warp.shape) == 3 and int(warp.shape[2]) == 4, str(warp.shape)
        H,W2,_ = warp.shape
        print(H, W2)
        W = W2//2
        # warp B keypoints into image A
        #x_A_from_B = F.grid_sample(warp[:, :, :2].permute(2,0,1)[None], x_B[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        #cert_A_from_B = F.grid_sample(certainty[None, None, :, :], x_B[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        
        x_A_from_B = F.grid_sample(warp[:, W:, :2].permute(2,0,1)[None], x_B[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        cert_A_from_B = F.grid_sample(certainty[None, None, :, W:], x_B[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        # match in the coordinate system of A
        D = torch.cdist(x_A, x_A_from_B)
        inds_A, inds_B = torch.nonzero((D == D.min(dim=-1, keepdim = True).values) * (D == D.min(dim=-2, keepdim = True).values) * (cert_A_from_B[None,:] > 0.01), as_tuple = True)
        return inds_A, inds_B, cert_A_from_B[inds_B]
        
    @torch.inference_mode()
    def match(
        self,
        im_A_input,
        im_B_input,
        domainid,
        *args,
        batched=False,
        device=None,
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Check if inputs are file paths or already loaded images
        if isinstance(im_A_input, (str, os.PathLike)):
            im_A = Image.open(im_A_input).convert("RGB")
        else:
            im_A = im_A_input

        if isinstance(im_B_input, (str, os.PathLike)):
            im_B = Image.open(im_B_input).convert("RGB")
        else:
            im_B = im_B_input

        symmetric = self.symmetric
        self.train(False)
        with torch.no_grad():
            if not batched:
                b = 1
                w, h = im_A.size
                w2, h2 = im_B.size
                # Get images in good format
                ws = self.w_resized
                hs = self.h_resized

                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True, clahe=False
                )
                im_A, im_B = test_transform((im_A, im_B))
                batch = {"im_A": im_A[None].to(device), "im_B": im_B[None].to(device), "domainid": torch.tensor(domainid).to(device)}
            else:
                b, c, h, w = im_A.shape
                b, c, h2, w2 = im_B.shape
                assert w == w2 and h == h2, "For batched images we assume same size"
                batch = {"im_A": im_A.to(device), "im_B": im_B.to(device), "domainid": domainid.to(device)}
                if h != self.h_resized or self.w_resized != w:
                    warn("Model resolution and batch resolution differ, may produce unexpected results")
                hs, ws = h, w
            finest_scale = 1
            # Run matcher
            if symmetric:
                corresps = self.forward_symmetric(batch)
            else:
                corresps = self.forward(batch, batched=True)

            if self.upsample_preds:
                hs, ws = self.upsample_res

            if self.attenuate_cert:
                low_res_certainty = F.interpolate(
                    corresps[16]["certainty"], size=(hs, ws), align_corners=False, mode="bilinear"
                )
                cert_clamp = 0
                factor = 0.5
                low_res_certainty = factor * low_res_certainty * (low_res_certainty < cert_clamp)

            if self.upsample_preds:
                finest_corresps = corresps[finest_scale]
                torch.cuda.empty_cache()
                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True
                )
                if isinstance(im_A_input, (str, os.PathLike)):
                    im_A, im_B = test_transform(
                        (Image.open(im_A_input).convert('RGB'), Image.open(im_B_input).convert('RGB')))
                else:
                    im_A, im_B = test_transform((im_A_input, im_B_input))

                im_A, im_B = im_A[None].to(device), im_B[None].to(device)
                scale_factor = math.sqrt(self.upsample_res[0] * self.upsample_res[1] / (self.w_resized * self.h_resized))
                batch = {"im_A": im_A, "im_B": im_B, "domainid": torch.tensor(domainid).to(device), "corresps": finest_corresps}
                if symmetric:
                    corresps = self.forward_symmetric(batch, upsample=True, batched=True, scale_factor=scale_factor)
                else:
                    corresps = self.forward(batch, batched=True, upsample=True, scale_factor=scale_factor)

            im_A_to_im_B = corresps[finest_scale]["flow"]
            certainty = corresps[finest_scale]["certainty"] - (low_res_certainty if self.attenuate_cert else 0)
            if finest_scale != 1:
                im_A_to_im_B = F.interpolate(
                    im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
                )
                certainty = F.interpolate(
                    certainty, size=(hs, ws), align_corners=False, mode="bilinear"
                )
            im_A_to_im_B = im_A_to_im_B.permute(
                0, 2, 3, 1
            )
            # Create im_A meshgrid
            im_A_coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                    torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
                ),
                indexing='ij'
            )
            im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
            im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
            certainty = certainty.sigmoid()  # logits -> probs
            im_A_coords = im_A_coords.permute(0, 2, 3, 1)
            if (im_A_to_im_B.abs() > 1).any() and True:
                wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
                certainty[wrong[:, None]] = 0
            im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
            if symmetric:
                A_to_B, B_to_A = im_A_to_im_B.chunk(2)
                q_warp = torch.cat((im_A_coords, A_to_B), dim=-1)
                im_B_coords = im_A_coords
                s_warp = torch.cat((B_to_A, im_B_coords), dim=-1)
                warp = torch.cat((q_warp, s_warp), dim=2)
                certainty = torch.cat(certainty.chunk(2), dim=3)
            else:
                warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
            if batched:
                return (
                    warp,
                    certainty[:, 0]
                )
            else:
                return (
                    warp[0],
                    certainty[0, 0],
                )
    
class DomainAdapter(nn.Module):
    def __init__(self, num_domains, feat_dim, scale=0.1):
        super().__init__()
        self.domain_emb = nn.Embedding(num_domains, feat_dim)
        self.adapter_mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid()  # gate ∈ [0,1]
        )
        self.scale = scale

    def forward(self, x, domain_id):
        B, C, H, W = x.shape
        dtype = x.dtype

        # domain_id can be scalar or tensor [B]
        if isinstance(domain_id, int):
            domain_id = torch.LongTensor([domain_id]).to(x.device).expand(B)
        elif isinstance(domain_id, torch.Tensor) and domain_id.ndim == 0:
            domain_id = domain_id.expand(B)

        domain_feat = self.domain_emb(domain_id)  # [B, C]

        # Compute small adjustment and gate
        delta = self.adapter_mlp(domain_feat).view(B, C, 1, 1) * self.scale
        gate = self.gate_mlp(domain_feat).view(B, C, 1, 1)

        # Apply residual with gate
        x_out = x + gate * delta
        return x_out.to(dtype)

class PyramidDomainAdapter(nn.Module):
    def __init__(self, num_domains, feat_dims):
        """
        feat_dims: dict, e.g., {16: 1024+768, 8: 512, 4: 256, 2: 128, 1: 64}
        """
        super().__init__()
        self.adapters = nn.ModuleDict()
        for scale, dim in feat_dims.items():
            self.adapters[str(scale)] = DomainAdapter(num_domains, dim)

    def forward(self, feats, domain_id):
        """
        feats: dict of {scale: [B, C, H, W]}
        domain_id: int or [B]
        """
        out = {}
        for scale, feat in feats.items():
            adapter = self.adapters[str(scale)]
            out[scale] = adapter(feat, domain_id)
        return out

class knn_matcher(nn.Module):
    def __init__(
        self,
        encoder,
        h=448,
        w=448,
        sample_mode = "threshold_balanced",
        upsample_preds = False,
        upsample_res = (14*16*6, 14*16*6),
        symmetric = False,
        name = None,
        attenuate_cert = None,
    ):
        super().__init__()
        self.attenuate_cert = attenuate_cert
        self.encoder = encoder
        self.name = name
        self.w_resized = w
        self.h_resized = h
        self.og_transforms = get_tuple_transform_ops(resize=None, normalize=True)
        self.sample_mode = sample_mode
        self.upsample_preds = upsample_preds
        self.upsample_res = upsample_res
        self.symmetric = symmetric
        self.sample_thresh = 0.05

    def get_output_resolution(self):
        if not self.upsample_preds:
            return self.h_resized, self.w_resized
        else:
            return self.upsample_res
    
    def extract_backbone_features(self, batch, batched = True, upsample = False):
        x_q = batch["im_A"]
        x_s = batch["im_B"]
        if batched:
            X = torch.cat((x_q, x_s), dim = 0)
            feature_pyramid = self.encoder(X, upsample = upsample)
        else:
            feature_pyramid = self.encoder(x_q, upsample = upsample), self.encoder(x_s, upsample = upsample)
        return feature_pyramid
    
    def decoder(self, fs_1, fs_2, scale_factor=1, k=1):
        K = CosKernel(T=0.1)    
        b, d, h, w = fs_1.shape

        fs_1, fs_2 = reshape(fs_1), reshape(fs_2)#b d h w -> b (h w) d
        distances = K(fs_1, fs_2) #bnd, bmd -> bnm
        
        distances = distances.view(b, h, w, -1).permute(0, 3, 1, 2)#b,C,h,w
        max_vals, _ = distances.max(dim=1, keepdim=True)  # Shape: (b, 1, h, w)
        gm_warp_or_cls = (distances - max_vals) * 10
        
        flow = cls_to_flow(
                        gm_warp_or_cls,
                    ).permute(0,3,1,2)#b,2,h,w
        certainty = gm_warp_or_cls
        corresps = {}
        corresps['gm_cls'] = gm_warp_or_cls
        max_certainty, max_indices = certainty.max(dim=1, keepdim=True)
        corresps['gm_certainty'] = max_certainty
        corresps['flow'] = flow
        return corresps

    def forward(self, batch, batched = True, upsample = False, scale_factor = 1):
        feature_pyramid = self.extract_backbone_features(batch, batched=batched, upsample = upsample)
        if batched:
            f_q_pyramid = feature_pyramid.chunk(2)[0]
            f_s_pyramid = feature_pyramid.chunk(2)[1]
        else:
            f_q_pyramid, f_s_pyramid = feature_pyramid
        corresps = self.decoder(f_q_pyramid, 
                                f_s_pyramid)
        
        return corresps
    
    def forward_symmetric(self, batch, batched = True, upsample = False, scale_factor = 1):
        feature_pyramid = self.extract_backbone_features(batch, batched = batched, upsample = upsample)
        f_q_pyramid = feature_pyramid
        f_s_pyramid = torch.cat((feature_pyramid.chunk(2)[1], feature_pyramid.chunk(2)[0]), dim = 0)
        corresps = self.decoder(f_q_pyramid, 
                                f_s_pyramid, 
                                scale_factor=scale_factor)
        return corresps


    @torch.inference_mode()
    def match(self, im_A_input, im_B_input, *args, batched=False, device=None,):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(im_A_input, (str, os.PathLike)):
            im_A = Image.open(im_A_input).convert("RGB")
        else:
            im_A = im_A_input

        if isinstance(im_B_input, (str, os.PathLike)):
            im_B = Image.open(im_B_input).convert("RGB")
        else:
            im_B = im_B_input

        symmetric = self.symmetric
        self.train(False)

        with torch.no_grad():
            if not batched:
                b = 1
                w, h = im_A.size
                w2, h2 = im_B.size
                # Get images in good format
                ws = self.w_resized
                hs = self.h_resized

                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True, clahe=False
                )
                im_A, im_B = test_transform((im_A, im_B))
                batch = {"im_A": im_A[None].to(device), "im_B": im_B[None].to(device)}
            else:
                b, c, h, w = im_A.shape
                b, c, h2, w2 = im_B.shape
                assert w == w2 and h == h2, "For batched images we assume same size"
                batch = {"im_A": im_A.to(device), "im_B": im_B.to(device)}
                hs, ws = h, w

            if symmetric:
                corresps = self.forward_symmetric(batch)
            else:
                corresps = self.forward(batch, batched=True)
            
            if self.upsample_preds:
                hs, ws = self.upsample_res
                
            im_A_to_im_B = corresps["flow"]
            certainty = corresps["gm_certainty"]

            im_A_to_im_B = F.interpolate(
                    im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
                )
            im_A_to_im_B = im_A_to_im_B.permute(
                0, 2, 3, 1
            )
            certainty = F.interpolate(
                certainty, size=(hs, ws), align_corners=False, mode="bilinear"
                )
            # Create im_A meshgrid
            im_A_coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                    torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
                ),
                indexing='ij'
            )
            im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
            im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
            #certainty = certainty.sigmoid()  # logits -> probs
            im_A_coords = im_A_coords.permute(0, 2, 3, 1)
            if (im_A_to_im_B.abs() > 1).any() and True:
                wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
                certainty[wrong[:, None]] = 0
            im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
            #warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
            if symmetric:
                A_to_B, B_to_A = im_A_to_im_B.chunk(2)
                q_warp = torch.cat((im_A_coords, A_to_B), dim=-1)
                im_B_coords = im_A_coords
                s_warp = torch.cat((B_to_A, im_B_coords), dim=-1)
                warp = torch.cat((q_warp, s_warp), dim=2)
                certainty = torch.cat(certainty.chunk(2), dim=3)
            else:
                warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
            if batched:
                return (
                    warp,
                    certainty[:, 0]
                )
            else:
                return (
                    warp[0],
                    certainty[0, 0],
                )

class mnn_matcher(nn.Module): #knn for mast3r forward
    def __init__(
        self,
        encoder,
        h=448,
        w=448,
        sample_mode = "threshold_balanced",
        upsample_preds = False,
        upsample_res = (14*16*6, 14*16*6),
        symmetric = False,
        name = None,
        attenuate_cert = None,
    ):
        super().__init__()
        self.attenuate_cert = attenuate_cert
        self.encoder = encoder
        self.name = name
        self.w_resized = w
        self.h_resized = h
        self.og_transforms = get_tuple_transform_ops(resize=None, normalize=True)
        self.sample_mode = sample_mode
        self.upsample_preds = upsample_preds
        self.upsample_res = upsample_res
        self.symmetric = symmetric
        self.sample_thresh = 0.05

    def get_output_resolution(self):
        if not self.upsample_preds:
            return self.h_resized, self.w_resized
        else:
            return self.upsample_res
    
    def extract_backbone_features(self, batch, batched = True, upsample = False):
        x_q = batch["im_A"]
        x_s = batch["im_B"]
        feature_pyramid = self.encoder((x_q, x_s))#, upsample = upsample), self.encoder(x_s, upsample = upsample)
        return feature_pyramid
    
    
    def decoder(self, fs_1, fs_2, scale_factor=1, k=1):
        K = CosKernel(T=0.1)    
        b, d, h, w = fs_1.shape

        fs_1, fs_2 = reshape(fs_1), reshape(fs_2)#b d h w -> b (h w) d
        distances = K(fs_1, fs_2) #bnd, bmd -> bnm
        
        distances = distances.view(b, h, w, -1).permute(0, 3, 1, 2)#b,C,h,w
        max_vals, _ = distances.max(dim=1, keepdim=True)  # Shape: (b, 1, h, w)
        gm_warp_or_cls = (distances - max_vals) * 10

        
        flow = cls_to_flow(
                        gm_warp_or_cls,
                    ).permute(0,3,1,2)#b,2,h,w
        certainty = gm_warp_or_cls
        corresps = {}
        corresps['gm_cls'] = gm_warp_or_cls
        max_certainty, max_indices = certainty.max(dim=1, keepdim=True)
        corresps['gm_certainty'] = max_certainty
        corresps['flow'] = flow

        return corresps


    def forward(self, batch, batched = True, upsample = False, scale_factor = 1):
        feature_pyramid = self.extract_backbone_features(batch, batched=batched, upsample = upsample)
        f_q_pyramid, f_s_pyramid = feature_pyramid
        corresps = self.decoder(f_q_pyramid, 
                                f_s_pyramid)
        
        return corresps
    
    def forward_symmetric(self, batch, batched = True, upsample = False, scale_factor = 1):
        feature_pyramid = self.extract_backbone_features(batch, batched = batched, upsample = upsample)
        f_q_pyramid = torch.cat((feature_pyramid[0], feature_pyramid[1]), dim = 0)
        f_s_pyramid = torch.cat((feature_pyramid[1], feature_pyramid[0]), dim = 0)
        corresps = self.decoder(f_q_pyramid, 
                                f_s_pyramid, 
                                scale_factor=scale_factor)
        return corresps


    @torch.inference_mode()
    def match(self, im_A_input, im_B_input, *args, batched=False, device=None,):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(im_A_input, (str, os.PathLike)):
            im_A = Image.open(im_A_input).convert("RGB")
        else:
            im_A = im_A_input

        if isinstance(im_B_input, (str, os.PathLike)):
            im_B = Image.open(im_B_input).convert("RGB")
        else:
            im_B = im_B_input

        symmetric = self.symmetric
        self.train(False)

        with torch.no_grad():
            if not batched:
                b = 1
                w, h = im_A.size
                w2, h2 = im_B.size
                # Get images in good format
                ws = self.w_resized
                hs = self.h_resized

                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True, clahe=False
                )
                im_A, im_B = test_transform((im_A, im_B))
                batch = {"im_A": im_A[None].to(device), "im_B": im_B[None].to(device)}
            else:
                b, c, h, w = im_A.shape
                b, c, h2, w2 = im_B.shape
                assert w == w2 and h == h2, "For batched images we assume same size"
                batch = {"im_A": im_A.to(device), "im_B": im_B.to(device)}
                hs, ws = h, w

            finest_scale = 1
            if symmetric:
                corresps = self.forward_symmetric(batch)
            else:
                corresps = self.forward(batch, batched=True)
            
            if self.upsample_preds:
                hs, ws = self.upsample_res
                
            im_A_to_im_B = corresps["flow"]
            certainty = corresps["gm_certainty"]

            im_A_to_im_B = F.interpolate(
                    im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
                )
            im_A_to_im_B = im_A_to_im_B.permute(
                0, 2, 3, 1
            )
            certainty = F.interpolate(
                certainty, size=(hs, ws), align_corners=False, mode="bilinear"
                )
            # Create im_A meshgrid
            im_A_coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                    torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
                ),
                indexing='ij'
            )
            im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
            im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
            #certainty = certainty.sigmoid()  # logits -> probs
            im_A_coords = im_A_coords.permute(0, 2, 3, 1)
            if (im_A_to_im_B.abs() > 1).any() and True:
                wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
                certainty[wrong[:, None]] = 0
            im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
            #warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
            if symmetric:
                A_to_B, B_to_A = im_A_to_im_B.chunk(2)
                q_warp = torch.cat((im_A_coords, A_to_B), dim=-1)
                im_B_coords = im_A_coords
                s_warp = torch.cat((B_to_A, im_B_coords), dim=-1)
                warp = torch.cat((q_warp, s_warp), dim=2)
                certainty = torch.cat(certainty.chunk(2), dim=3)
            else:
                warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
            if batched:
                return (
                    warp,
                    certainty[:, 0]
                )
            else:
                return (
                    warp[0],
                    certainty[0, 0],
                )

def reshape(x):
    return rearrange(x, "b d h w -> b (h w) d")

class RegressionMatcher_mast3r(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        h=448,
        w=448,
        sample_mode = "threshold_balanced",
        upsample_preds = False,
        upsample_res = (14*16*6, 14*16*6),
        symmetric = False,
        name = None,
        attenuate_cert = None,
    ):
        super().__init__()
        self.attenuate_cert = attenuate_cert
        self.encoder = encoder
        self.decoder = decoder
        self.name = name
        self.w_resized = w
        self.h_resized = h
        self.og_transforms = get_tuple_transform_ops(resize=None, normalize=True)
        self.sample_mode = sample_mode
        self.upsample_preds = upsample_preds
        self.upsample_res = upsample_res
        self.symmetric = symmetric
        self.sample_thresh = 0.05
            
    def get_output_resolution(self):
        if not self.upsample_preds:
            return self.h_resized, self.w_resized
        else:
            return self.upsample_res

    def extract_backbone_features(self, batch, batched = True, upsample = False):
        x_q = batch["im_A"]
        x_s = batch["im_B"]
        feature_pyramid = self.encoder((x_q, x_s), upsample = upsample)
        return feature_pyramid

    def sample(
        self,
        matches,
        certainty,
        num=10000,
    ):
        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            certainty = certainty.clone()
            certainty[certainty > upper_thresh] = 1
        matches, certainty = (
            matches.reshape(-1, 4),
            certainty.reshape(-1),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        good_samples = torch.multinomial(certainty, 
                          num_samples = min(expansion_factor*num, len(certainty)), 
                          replacement=False)
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        if "balanced" not in self.sample_mode:
            return good_matches, good_certainty
        density = kde(good_matches, std=0.1)
        p = 1 / (density+1)
        p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        balanced_samples = torch.multinomial(p, 
                          num_samples = min(num,len(good_certainty)), 
                          replacement=False)
        return good_matches[balanced_samples], good_certainty[balanced_samples]

    def forward(self, batch, end2end=False, batched = True, upsample = False, scale_factor = 1, mode = 'default'):
        feature_pyramid = self.extract_backbone_features(batch, batched=batched, upsample = upsample)
        f_q_pyramid, f_s_pyramid = feature_pyramid
        corresps = self.decoder(f_q_pyramid, 
                                f_s_pyramid, 
                                upsample = upsample, 
                                **(batch["corresps"] if "corresps" in batch else {}),
                                scale_factor=scale_factor)
        
        return corresps

    def forward_symmetric(self, batch, batched = True, upsample = False, scale_factor = 1):
        x_q = batch["im_A"]
        x_s = batch["im_B"]
        feature_pyramid_q = self.encoder((x_q, x_s), upsample = upsample)
        feature_pyramid_s = self.encoder((x_s, x_q), upsample = upsample)
        f_q_pyramid = {key: torch.cat([feature_pyramid_q[0][key], feature_pyramid_s[0][key]], dim=0) 
                   for key in feature_pyramid_q[0]}
        f_s_pyramid = {key: torch.cat([feature_pyramid_q[1][key], feature_pyramid_s[1][key]], dim=0) 
                   for key in feature_pyramid_s[0]}
         
        corresps = self.decoder(f_q_pyramid, 
                                f_s_pyramid, 
                                upsample = upsample, 
                                **(batch["corresps"] if "corresps" in batch else {}),
                                scale_factor=scale_factor)
        return corresps
    
    def conf_from_fb_consistency(self, flow_forward, flow_backward, th = 2):
        # assumes that flow forward is of shape (..., H, W, 2)
        has_batch = False
        if len(flow_forward.shape) == 3:
            flow_forward, flow_backward = flow_forward[None], flow_backward[None]
        else:
            has_batch = True
        H,W = flow_forward.shape[-3:-1]
        th_n = 2 * th / max(H,W)
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1 + 1 / W, 1 - 1 / W, W), 
            torch.linspace(-1 + 1 / H, 1 - 1 / H, H), indexing = "xy"),
                             dim = -1).to(flow_forward.device)
        coords_fb = F.grid_sample(
            flow_backward.permute(0, 3, 1, 2), 
            flow_forward, 
            align_corners=False, mode="bilinear").permute(0, 2, 3, 1)
        diff = (coords - coords_fb).norm(dim=-1)
        in_th = (diff < th_n).float()
        if not has_batch:
            in_th = in_th[0]
        return in_th
         
    def to_pixel_coordinates(self, coords, H_A, W_A, H_B = None, W_B = None):
        if coords.shape[-1] == 2:
            return self._to_pixel_coordinates(coords, H_A, W_A) 
        
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[...,:2], coords[...,2:]
        return self._to_pixel_coordinates(kpts_A, H_A, W_A), self._to_pixel_coordinates(kpts_B, H_B, W_B)

    def _to_pixel_coordinates(self, coords, H, W):
        kpts = torch.stack((W/2 * (coords[...,0]+1), H/2 * (coords[...,1]+1)),axis=-1)
        return kpts
 
    def to_normalized_coordinates(self, coords, H_A, W_A, H_B, W_B):
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[...,:2], coords[...,2:]
        kpts_A = torch.stack((2/W_A * kpts_A[...,0] - 1, 2/H_A * kpts_A[...,1] - 1),axis=-1)
        kpts_B = torch.stack((2/W_B * kpts_B[...,0] - 1, 2/H_B * kpts_B[...,1] - 1),axis=-1)
        return kpts_A, kpts_B

    def match_keypoints(self, x_A, x_B, warp, certainty, return_tuple = True, return_inds = False):
        x_A_to_B = F.grid_sample(warp[...,-2:].permute(2,0,1)[None], x_A[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        cert_A_to_B = F.grid_sample(certainty[None,None,...], x_A[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        D = torch.cdist(x_A_to_B, x_B)
        inds_A, inds_B = torch.nonzero((D == D.min(dim=-1, keepdim = True).values) * (D == D.min(dim=-2, keepdim = True).values) * (cert_A_to_B[:,None] > self.sample_thresh), as_tuple = True)
        
        if return_tuple:
            if return_inds:
                return inds_A, inds_B
            else:
                return x_A[inds_A], x_B[inds_B]
        else:
            if return_inds:
                return torch.cat((inds_A, inds_B),dim=-1)
            else:
                return torch.cat((x_A[inds_A], x_B[inds_B]),dim=-1)

    def match_keypoints2(self, x_A, x_B, warp, certainty): 
        assert len(warp.shape) == 3 and int(warp.shape[2]) == 4, str(warp.shape)
        H,W2,_ = warp.shape
        print(H, W2)
        W = W2//2
        
        x_A_from_B = F.grid_sample(warp[:, W:, :2].permute(2,0,1)[None], x_B[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        cert_A_from_B = F.grid_sample(certainty[None, None, :, W:], x_B[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        # match in the coordinate system of A
        D = torch.cdist(x_A, x_A_from_B)
        inds_A, inds_B = torch.nonzero((D == D.min(dim=-1, keepdim = True).values) * (D == D.min(dim=-2, keepdim = True).values) * (cert_A_from_B[None,:] > 0.01), as_tuple = True)
        return inds_A, inds_B, cert_A_from_B[inds_B]

    @torch.inference_mode()
    def match(
        self,
        im_A_input,
        im_B_input,
        *args,
        batched=False,
        device=None,
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Check if inputs are file paths or already loaded images
        if isinstance(im_A_input, (str, os.PathLike)):
            im_A = Image.open(im_A_input).convert("RGB")
        else:
            im_A = im_A_input

        if isinstance(im_B_input, (str, os.PathLike)):
            im_B = Image.open(im_B_input).convert("RGB")
        else:
            im_B = im_B_input

        symmetric = self.symmetric
        self.train(False)
        with torch.no_grad():
            if not batched:
                b = 1
                w, h = im_A.size
                w2, h2 = im_B.size
                # Get images in good format
                ws = self.w_resized
                hs = self.h_resized

                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True, clahe=False
                )
                im_A, im_B = test_transform((im_A, im_B))
                batch = {"im_A": im_A[None].to(device), "im_B": im_B[None].to(device)}
            else:
                b, c, h, w = im_A.shape
                b, c, h2, w2 = im_B.shape
                assert w == w2 and h == h2, "For batched images we assume same size"
                batch = {"im_A": im_A.to(device), "im_B": im_B.to(device)}
                if h != self.h_resized or self.w_resized != w:
                    warn("Model resolution and batch resolution differ, may produce unexpected results")
                hs, ws = h, w
            finest_scale = 1
            # Run matcher
            if symmetric:
                corresps = self.forward_symmetric(batch)
            else:
                corresps = self.forward(batch, batched=True)

            if self.upsample_preds:
                hs, ws = self.upsample_res

            if self.attenuate_cert:
                low_res_certainty = F.interpolate(
                    corresps[16]["certainty"], size=(hs, ws), align_corners=False, mode="bilinear"
                )
                cert_clamp = 0
                factor = 0.5
                low_res_certainty = factor * low_res_certainty * (low_res_certainty < cert_clamp)

            if self.upsample_preds:
                finest_corresps = corresps[finest_scale]
                torch.cuda.empty_cache()
                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True
                )
                if isinstance(im_A_input, (str, os.PathLike)):
                    im_A, im_B = test_transform(
                        (Image.open(im_A_input).convert('RGB'), Image.open(im_B_input).convert('RGB')))
                else:
                    im_A, im_B = test_transform((im_A_input, im_B_input))

                im_A, im_B = im_A[None].to(device), im_B[None].to(device)
                scale_factor = math.sqrt(self.upsample_res[0] * self.upsample_res[1] / (self.w_resized * self.h_resized))
                batch = {"im_A": im_A, "im_B": im_B, "corresps": finest_corresps}
                if symmetric:
                    corresps = self.forward_symmetric(batch, upsample=True, batched=True, scale_factor=scale_factor)
                else:
                    corresps = self.forward(batch, batched=True, upsample=True, scale_factor=scale_factor)

            im_A_to_im_B = corresps[finest_scale]["flow"]
            certainty = corresps[finest_scale]["certainty"] - (low_res_certainty if self.attenuate_cert else 0)
            if finest_scale != 1:
                im_A_to_im_B = F.interpolate(
                    im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
                )
                certainty = F.interpolate(
                    certainty, size=(hs, ws), align_corners=False, mode="bilinear"
                )
            im_A_to_im_B = im_A_to_im_B.permute(
                0, 2, 3, 1
            )
            # Create im_A meshgrid
            im_A_coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                    torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
                ),
                indexing='ij'
            )
            im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
            im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
            certainty = certainty.sigmoid()  # logits -> probs
            im_A_coords = im_A_coords.permute(0, 2, 3, 1)
            if (im_A_to_im_B.abs() > 1).any() and True:
                wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
                certainty[wrong[:, None]] = 0
            im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
            if symmetric:
                A_to_B, B_to_A = im_A_to_im_B.chunk(2)
                q_warp = torch.cat((im_A_coords, A_to_B), dim=-1)
                im_B_coords = im_A_coords
                s_warp = torch.cat((B_to_A, im_B_coords), dim=-1)
                warp = torch.cat((q_warp, s_warp), dim=2)
                certainty = torch.cat(certainty.chunk(2), dim=3)
            else:
                warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
            if batched:
                return (
                    warp,
                    certainty[:, 0]
                )
            else:
                return (
                    warp[0],
                    certainty[0, 0],
                )

    def visualize_warp(self, warp, certainty, im_A = None, im_B = None, 
                       im_A_path = None, im_B_path = None, device = "cuda", symmetric = True, save_path = None, unnormalize = False):
        #assert symmetric == True, "Currently assuming bidirectional warp, might update this if someone complains ;)"
        H,W2,_ = warp.shape
        W = W2//2 if symmetric else W2
        if im_A is None:
            from PIL import Image
            im_A, im_B = Image.open(im_A_path).convert("RGB"), Image.open(im_B_path).convert("RGB")
        if not isinstance(im_A, torch.Tensor):
            im_A = im_A.resize((W,H))
            im_B = im_B.resize((W,H))    
            x_B = (torch.tensor(np.array(im_B)) / 255).to(device).permute(2, 0, 1)
            if symmetric:
                x_A = (torch.tensor(np.array(im_A)) / 255).to(device).permute(2, 0, 1)
        else:
            if symmetric:
                x_A = im_A
            x_B = im_B
        im_A_transfer_rgb = F.grid_sample(
        x_B[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
        )[0]
        if symmetric:
            im_B_transfer_rgb = F.grid_sample(
            x_A[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
            )[0]
            warp_im = torch.cat((im_A_transfer_rgb,im_B_transfer_rgb),dim=2)
            white_im = torch.ones((H,2*W),device=device)
        else:
            warp_im = im_A_transfer_rgb
            white_im = torch.ones((H, W), device = device)
        vis_im = certainty * warp_im + (1 - certainty) * white_im
        if save_path is not None:
            from romatch.utils import tensor_to_pil
            tensor_to_pil(vis_im, unnormalize=unnormalize).save(save_path)
        return vis_im


class RegressionMatcher_mast3r_adapter(nn.Module):
    def __init__(
        self,
        encoder,
        adapter,
        decoder,
        h=448,
        w=448,
        sample_mode = "threshold_balanced",
        upsample_preds = False,
        upsample_res = (14*16*6, 14*16*6),
        symmetric = False,
        name = None,
        attenuate_cert = None,
    ):
        super().__init__()
        self.attenuate_cert = attenuate_cert
        self.encoder = encoder
        self.adapter = adapter
        self.decoder = decoder
        self.name = name
        self.w_resized = w
        self.h_resized = h
        self.og_transforms = get_tuple_transform_ops(resize=None, normalize=True)
        self.sample_mode = sample_mode
        self.upsample_preds = upsample_preds
        self.upsample_res = upsample_res
        self.symmetric = symmetric
        self.sample_thresh = 0.05
            
    def get_output_resolution(self):
        if not self.upsample_preds:
            return self.h_resized, self.w_resized
        else:
            return self.upsample_res


    def extract_backbone_features(self, batch, batched = True, upsample = False):
        x_q = batch["im_A"]
        x_s = batch["im_B"]
        feature_pyramid = self.encoder((x_q, x_s), upsample = upsample)
        return feature_pyramid


    def sample(
        self,
        matches,
        certainty,
        num=10000,
    ):
        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            certainty = certainty.clone()
            certainty[certainty > upper_thresh] = 1
        matches, certainty = (
            matches.reshape(-1, 4),
            certainty.reshape(-1),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        good_samples = torch.multinomial(certainty, 
                          num_samples = min(expansion_factor*num, len(certainty)), 
                          replacement=False)
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        if "balanced" not in self.sample_mode:
            return good_matches, good_certainty
        density = kde(good_matches, std=0.1)
        p = 1 / (density+1)
        p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        balanced_samples = torch.multinomial(p, 
                          num_samples = min(num,len(good_certainty)), 
                          replacement=False)
        return good_matches[balanced_samples], good_certainty[balanced_samples]

    def forward(self, batch, end2end=False, batched = True, upsample = False, scale_factor = 1, mode = 'default'):
        feature_pyramid = self.extract_backbone_features(batch, batched=batched, upsample = upsample)
        f_q_pyramid, f_s_pyramid = feature_pyramid
        f_q_pyramid = self.adapter(f_q_pyramid, batch['domainid'])
        f_s_pyramid = self.adapter(f_q_pyramid, batch['domainid'])
        corresps = self.decoder(f_q_pyramid, 
                                f_s_pyramid, 
                                upsample = upsample, 
                                **(batch["corresps"] if "corresps" in batch else {}),
                                scale_factor=scale_factor)
        
        return corresps

    def forward_symmetric(self, batch, batched = True, upsample = False, scale_factor = 1):
        x_q = batch["im_A"]
        x_s = batch["im_B"]
        feature_pyramid_q = self.encoder((x_q, x_s), upsample = upsample)
        feature_pyramid_s = self.encoder((x_s, x_q), upsample = upsample)
        f_q_pyramid = {key: torch.cat([feature_pyramid_q[0][key], feature_pyramid_s[0][key]], dim=0) 
                   for key in feature_pyramid_q[0]}
        f_s_pyramid = {key: torch.cat([feature_pyramid_q[1][key], feature_pyramid_s[1][key]], dim=0) 
                   for key in feature_pyramid_s[0]}
        f_q_pyramid = self.adapter(f_q_pyramid, batch['domainid'])
        f_s_pyramid = self.adapter(f_s_pyramid, batch['domainid'])
        corresps = self.decoder(f_q_pyramid, 
                                f_s_pyramid, 
                                upsample = upsample, 
                                **(batch["corresps"] if "corresps" in batch else {}),
                                scale_factor=scale_factor)
        return corresps
    
    def conf_from_fb_consistency(self, flow_forward, flow_backward, th = 2):
        # assumes that flow forward is of shape (..., H, W, 2)
        has_batch = False
        if len(flow_forward.shape) == 3:
            flow_forward, flow_backward = flow_forward[None], flow_backward[None]
        else:
            has_batch = True
        H,W = flow_forward.shape[-3:-1]
        th_n = 2 * th / max(H,W)
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1 + 1 / W, 1 - 1 / W, W), 
            torch.linspace(-1 + 1 / H, 1 - 1 / H, H), indexing = "xy"),
                             dim = -1).to(flow_forward.device)
        coords_fb = F.grid_sample(
            flow_backward.permute(0, 3, 1, 2), 
            flow_forward, 
            align_corners=False, mode="bilinear").permute(0, 2, 3, 1)
        diff = (coords - coords_fb).norm(dim=-1)
        in_th = (diff < th_n).float()
        if not has_batch:
            in_th = in_th[0]
        return in_th
         
    def to_pixel_coordinates(self, coords, H_A, W_A, H_B = None, W_B = None):
        if coords.shape[-1] == 2:
            return self._to_pixel_coordinates(coords, H_A, W_A) 
        
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[...,:2], coords[...,2:]
        return self._to_pixel_coordinates(kpts_A, H_A, W_A), self._to_pixel_coordinates(kpts_B, H_B, W_B)

    def _to_pixel_coordinates(self, coords, H, W):
        kpts = torch.stack((W/2 * (coords[...,0]+1), H/2 * (coords[...,1]+1)),axis=-1)
        return kpts
 
    def to_normalized_coordinates(self, coords, H_A, W_A, H_B, W_B):
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[...,:2], coords[...,2:]
        kpts_A = torch.stack((2/W_A * kpts_A[...,0] - 1, 2/H_A * kpts_A[...,1] - 1),axis=-1)
        kpts_B = torch.stack((2/W_B * kpts_B[...,0] - 1, 2/H_B * kpts_B[...,1] - 1),axis=-1)
        return kpts_A, kpts_B

    def match_keypoints(self, x_A, x_B, warp, certainty, return_tuple = True, return_inds = False):
        x_A_to_B = F.grid_sample(warp[...,-2:].permute(2,0,1)[None], x_A[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        cert_A_to_B = F.grid_sample(certainty[None,None,...], x_A[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        D = torch.cdist(x_A_to_B, x_B)
        inds_A, inds_B = torch.nonzero((D == D.min(dim=-1, keepdim = True).values) * (D == D.min(dim=-2, keepdim = True).values) * (cert_A_to_B[:,None] > self.sample_thresh), as_tuple = True)
        
        if return_tuple:
            if return_inds:
                return inds_A, inds_B
            else:
                return x_A[inds_A], x_B[inds_B]
        else:
            if return_inds:
                return torch.cat((inds_A, inds_B),dim=-1)
            else:
                return torch.cat((x_A[inds_A], x_B[inds_B]),dim=-1)

    def match_keypoints2(self, x_A, x_B, warp, certainty): 
        assert len(warp.shape) == 3 and int(warp.shape[2]) == 4, str(warp.shape)
        H,W2,_ = warp.shape
        print(H, W2)
        W = W2//2
        # warp B keypoints into image A
        #x_A_from_B = F.grid_sample(warp[:, :, :2].permute(2,0,1)[None], x_B[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        #cert_A_from_B = F.grid_sample(certainty[None, None, :, :], x_B[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        
        x_A_from_B = F.grid_sample(warp[:, W:, :2].permute(2,0,1)[None], x_B[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        cert_A_from_B = F.grid_sample(certainty[None, None, :, W:], x_B[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        # match in the coordinate system of A
        D = torch.cdist(x_A, x_A_from_B)
        inds_A, inds_B = torch.nonzero((D == D.min(dim=-1, keepdim = True).values) * (D == D.min(dim=-2, keepdim = True).values) * (cert_A_from_B[None,:] > 0.01), as_tuple = True)
        return inds_A, inds_B, cert_A_from_B[inds_B]
        '''
        x_A_to_B = F.grid_sample(warp[...,-2:].permute(2,0,1)[None], x_A[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        cert_A_to_B = F.grid_sample(certainty[None,None,...], x_A[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        D = torch.cdist(x_A_to_B, x_B)
        inds_A, inds_B = torch.nonzero((D == D.min(dim=-1, keepdim = True).values) * (D == D.min(dim=-2, keepdim = True).values) * (cert_A_to_B[:,None] > self.sample_thresh), as_tuple = True)
        return inds_A, inds_B, cert_A_to_B[inds_A]
        '''
    @torch.inference_mode()
    def match(
        self,
        im_A_input,
        im_B_input,
        domainid,
        *args,
        batched=False,
        device=None,
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Check if inputs are file paths or already loaded images
        if isinstance(im_A_input, (str, os.PathLike)):
            im_A = Image.open(im_A_input).convert("RGB")
        else:
            im_A = im_A_input

        if isinstance(im_B_input, (str, os.PathLike)):
            im_B = Image.open(im_B_input).convert("RGB")
        else:
            im_B = im_B_input

        symmetric = self.symmetric
        self.train(False)
        with torch.no_grad():
            if not batched:
                b = 1
                w, h = im_A.size
                w2, h2 = im_B.size
                # Get images in good format
                ws = self.w_resized
                hs = self.h_resized

                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True, clahe=False
                )
                im_A, im_B = test_transform((im_A, im_B))
                batch = {"im_A": im_A[None].to(device), "im_B": im_B[None].to(device), 'domainid': domainid.to(device)}
            else:
                b, c, h, w = im_A.shape
                b, c, h2, w2 = im_B.shape
                assert w == w2 and h == h2, "For batched images we assume same size"
                batch = {"im_A": im_A.to(device), "im_B": im_B.to(device), 'domainid': domainid.to(device)}
                if h != self.h_resized or self.w_resized != w:
                    warn("Model resolution and batch resolution differ, may produce unexpected results")
                hs, ws = h, w
            finest_scale = 1
            # Run matcher
            if symmetric:
                corresps = self.forward_symmetric(batch)
            else:
                corresps = self.forward(batch, batched=True)

            if self.upsample_preds:
                hs, ws = self.upsample_res

            if self.attenuate_cert:
                low_res_certainty = F.interpolate(
                    corresps[16]["certainty"], size=(hs, ws), align_corners=False, mode="bilinear"
                )
                cert_clamp = 0
                factor = 0.5
                low_res_certainty = factor * low_res_certainty * (low_res_certainty < cert_clamp)

            if self.upsample_preds:
                finest_corresps = corresps[finest_scale]
                torch.cuda.empty_cache()
                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True
                )
                if isinstance(im_A_input, (str, os.PathLike)):
                    im_A, im_B = test_transform(
                        (Image.open(im_A_input).convert('RGB'), Image.open(im_B_input).convert('RGB')))
                else:
                    im_A, im_B = test_transform((im_A_input, im_B_input))

                im_A, im_B = im_A[None].to(device), im_B[None].to(device)
                scale_factor = math.sqrt(self.upsample_res[0] * self.upsample_res[1] / (self.w_resized * self.h_resized))
                batch = {"im_A": im_A, "im_B": im_B, "domainid": domainid.to(device), "corresps": finest_corresps}
                if symmetric:
                    corresps = self.forward_symmetric(batch, upsample=True, batched=True, scale_factor=scale_factor)
                else:
                    corresps = self.forward(batch, batched=True, upsample=True, scale_factor=scale_factor)

            im_A_to_im_B = corresps[finest_scale]["flow"]
            certainty = corresps[finest_scale]["certainty"] - (low_res_certainty if self.attenuate_cert else 0)
            if finest_scale != 1:
                im_A_to_im_B = F.interpolate(
                    im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
                )
                certainty = F.interpolate(
                    certainty, size=(hs, ws), align_corners=False, mode="bilinear"
                )
            im_A_to_im_B = im_A_to_im_B.permute(
                0, 2, 3, 1
            )
            # Create im_A meshgrid
            im_A_coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                    torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
                ),
                indexing='ij'
            )
            im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
            im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
            certainty = certainty.sigmoid()  # logits -> probs
            im_A_coords = im_A_coords.permute(0, 2, 3, 1)
            if (im_A_to_im_B.abs() > 1).any() and True:
                wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
                certainty[wrong[:, None]] = 0
            im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
            if symmetric:
                A_to_B, B_to_A = im_A_to_im_B.chunk(2)
                q_warp = torch.cat((im_A_coords, A_to_B), dim=-1)
                im_B_coords = im_A_coords
                s_warp = torch.cat((B_to_A, im_B_coords), dim=-1)
                warp = torch.cat((q_warp, s_warp), dim=2)
                certainty = torch.cat(certainty.chunk(2), dim=3)
            else:
                warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
            if batched:
                return (
                    warp,
                    certainty[:, 0]
                )
            else:
                return (
                    warp[0],
                    certainty[0, 0],
                )

    def visualize_warp(self, warp, certainty, im_A = None, im_B = None, 
                       im_A_path = None, im_B_path = None, device = "cuda", symmetric = True, save_path = None, unnormalize = False):
        #assert symmetric == True, "Currently assuming bidirectional warp, might update this if someone complains ;)"
        H,W2,_ = warp.shape
        W = W2//2 if symmetric else W2
        if im_A is None:
            from PIL import Image
            im_A, im_B = Image.open(im_A_path).convert("RGB"), Image.open(im_B_path).convert("RGB")
        if not isinstance(im_A, torch.Tensor):
            im_A = im_A.resize((W,H))
            im_B = im_B.resize((W,H))    
            x_B = (torch.tensor(np.array(im_B)) / 255).to(device).permute(2, 0, 1)
            if symmetric:
                x_A = (torch.tensor(np.array(im_A)) / 255).to(device).permute(2, 0, 1)
        else:
            if symmetric:
                x_A = im_A
            x_B = im_B
        im_A_transfer_rgb = F.grid_sample(
        x_B[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
        )[0]
        if symmetric:
            im_B_transfer_rgb = F.grid_sample(
            x_A[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
            )[0]
            warp_im = torch.cat((im_A_transfer_rgb,im_B_transfer_rgb),dim=2)
            white_im = torch.ones((H,2*W),device=device)
        else:
            warp_im = im_A_transfer_rgb
            white_im = torch.ones((H, W), device = device)
        vis_im = certainty * warp_im + (1 - certainty) * white_im
        if save_path is not None:
            from romatch.utils import tensor_to_pil
            tensor_to_pil(vis_im, unnormalize=unnormalize).save(save_path)
        return vis_im


class RegressionMatcher_mast3r_coarse(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        h=448,
        w=448,
        sample_mode = "threshold_balanced",
        upsample_preds = False,
        upsample_res = (14*16*6, 14*16*6),
        symmetric = False,
        name = None,
        attenuate_cert = None,
    ):
        super().__init__()
        self.attenuate_cert = attenuate_cert
        self.encoder = encoder
        self.decoder = decoder
        self.name = name
        self.w_resized = w
        self.h_resized = h
        self.og_transforms = get_tuple_transform_ops(resize=None, normalize=True)
        self.sample_mode = sample_mode
        self.upsample_preds = upsample_preds
        self.upsample_res = upsample_res
        self.symmetric = symmetric
        self.sample_thresh = 0.05
            
    def get_output_resolution(self):
        if not self.upsample_preds:
            return self.h_resized, self.w_resized
        else:
            return self.upsample_res
    
    def extract_backbone_features(self, batch, batched = True, upsample = False):
        x_q = batch["im_A"]
        x_s = batch["im_B"]
        feature_pyramid = self.encoder((x_q, x_s), upsample = upsample)
        return feature_pyramid


    def sample(
        self,
        matches,
        certainty,
        num=10000,
    ):
        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            certainty = certainty.clone()
            certainty[certainty > upper_thresh] = 1
        matches, certainty = (
            matches.reshape(-1, 4),
            certainty.reshape(-1),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        good_samples = torch.multinomial(certainty, 
                          num_samples = min(expansion_factor*num, len(certainty)), 
                          replacement=False)
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        if "balanced" not in self.sample_mode:
            return good_matches, good_certainty
        density = kde(good_matches, std=0.1)
        p = 1 / (density+1)
        p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        balanced_samples = torch.multinomial(p, 
                          num_samples = min(num,len(good_certainty)), 
                          replacement=False)
        return good_matches[balanced_samples], good_certainty[balanced_samples]

    def forward(self, batch, end2end=False, batched = True, upsample = False, scale_factor = 1, mode = 'default'):
        feature_pyramid = self.extract_backbone_features(batch, batched=batched, upsample = upsample)
        f_q_pyramid, f_s_pyramid = feature_pyramid
        corresps = self.decoder(f_q_pyramid, 
                                f_s_pyramid, 
                                upsample = upsample, 
                                **(batch["corresps"] if "corresps" in batch else {}),
                                scale_factor=scale_factor)
        
        return corresps

    def forward_symmetric(self, batch, batched = True, upsample = False, scale_factor = 1):
        feature_pyramid = self.extract_backbone_features(batch, batched = batched, upsample = upsample)
        f_q_pyramid = {key: torch.cat([feature_pyramid[0][key], feature_pyramid[1][key]], dim=0) 
                   for key in feature_pyramid[0]}
        f_s_pyramid = {key: torch.cat([feature_pyramid[1][key], feature_pyramid[0][key]], dim=0) 
                   for key in feature_pyramid[0]}
         
        corresps = self.decoder(f_q_pyramid, 
                                f_s_pyramid, 
                                upsample = upsample, 
                                **(batch["corresps"] if "corresps" in batch else {}),
                                scale_factor=scale_factor)
        return corresps
    
    def conf_from_fb_consistency(self, flow_forward, flow_backward, th = 2):
        # assumes that flow forward is of shape (..., H, W, 2)
        has_batch = False
        if len(flow_forward.shape) == 3:
            flow_forward, flow_backward = flow_forward[None], flow_backward[None]
        else:
            has_batch = True
        H,W = flow_forward.shape[-3:-1]
        th_n = 2 * th / max(H,W)
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1 + 1 / W, 1 - 1 / W, W), 
            torch.linspace(-1 + 1 / H, 1 - 1 / H, H), indexing = "xy"),
                             dim = -1).to(flow_forward.device)
        coords_fb = F.grid_sample(
            flow_backward.permute(0, 3, 1, 2), 
            flow_forward, 
            align_corners=False, mode="bilinear").permute(0, 2, 3, 1)
        diff = (coords - coords_fb).norm(dim=-1)
        in_th = (diff < th_n).float()
        if not has_batch:
            in_th = in_th[0]
        return in_th
         
    def to_pixel_coordinates(self, coords, H_A, W_A, H_B = None, W_B = None):
        if coords.shape[-1] == 2:
            return self._to_pixel_coordinates(coords, H_A, W_A) 
        
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[...,:2], coords[...,2:]
        return self._to_pixel_coordinates(kpts_A, H_A, W_A), self._to_pixel_coordinates(kpts_B, H_B, W_B)

    def _to_pixel_coordinates(self, coords, H, W):
        kpts = torch.stack((W/2 * (coords[...,0]+1), H/2 * (coords[...,1]+1)),axis=-1)
        return kpts
 
    def to_normalized_coordinates(self, coords, H_A, W_A, H_B, W_B):
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[...,:2], coords[...,2:]
        kpts_A = torch.stack((2/W_A * kpts_A[...,0] - 1, 2/H_A * kpts_A[...,1] - 1),axis=-1)
        kpts_B = torch.stack((2/W_B * kpts_B[...,0] - 1, 2/H_B * kpts_B[...,1] - 1),axis=-1)
        return kpts_A, kpts_B

    def match_keypoints(self, x_A, x_B, warp, certainty, return_tuple = True, return_inds = False):
        x_A_to_B = F.grid_sample(warp[...,-2:].permute(2,0,1)[None], x_A[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        cert_A_to_B = F.grid_sample(certainty[None,None,...], x_A[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        D = torch.cdist(x_A_to_B, x_B)
        inds_A, inds_B = torch.nonzero((D == D.min(dim=-1, keepdim = True).values) * (D == D.min(dim=-2, keepdim = True).values) * (cert_A_to_B[:,None] > self.sample_thresh), as_tuple = True)
        
        if return_tuple:
            if return_inds:
                return inds_A, inds_B
            else:
                return x_A[inds_A], x_B[inds_B]
        else:
            if return_inds:
                return torch.cat((inds_A, inds_B),dim=-1)
            else:
                return torch.cat((x_A[inds_A], x_B[inds_B]),dim=-1)

    def match_keypoints2(self, x_A, x_B, warp, certainty): 
        assert len(warp.shape) == 3 and int(warp.shape[2]) == 4, str(warp.shape)
        H,W2,_ = warp.shape
        print(H, W2)
        W = W2//2
        x_A_from_B = F.grid_sample(warp[:, W:, :2].permute(2,0,1)[None], x_B[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        cert_A_from_B = F.grid_sample(certainty[None, None, :, W:], x_B[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        # match in the coordinate system of A
        D = torch.cdist(x_A, x_A_from_B)
        inds_A, inds_B = torch.nonzero((D == D.min(dim=-1, keepdim = True).values) * (D == D.min(dim=-2, keepdim = True).values) * (cert_A_from_B[None,:] > 0.01), as_tuple = True)
        return inds_A, inds_B, cert_A_from_B[inds_B]
        
    @torch.inference_mode()
    def match(
        self,
        im_A_input,
        im_B_input,
        *args,
        batched=False,
        device=None,
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Check if inputs are file paths or already loaded images
        if isinstance(im_A_input, (str, os.PathLike)):
            im_A = Image.open(im_A_input).convert("RGB")
        else:
            im_A = im_A_input

        if isinstance(im_B_input, (str, os.PathLike)):
            im_B = Image.open(im_B_input).convert("RGB")
        else:
            im_B = im_B_input

        symmetric = self.symmetric
        self.train(False)
        with torch.no_grad():
            if not batched:
                b = 1
                w, h = im_A.size
                w2, h2 = im_B.size
                # Get images in good format
                ws = self.w_resized
                hs = self.h_resized

                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True, clahe=False
                )
                im_A, im_B = test_transform((im_A, im_B))
                batch = {"im_A": im_A[None].to(device), "im_B": im_B[None].to(device)}
            else:
                b, c, h, w = im_A.shape
                b, c, h2, w2 = im_B.shape
                assert w == w2 and h == h2, "For batched images we assume same size"
                batch = {"im_A": im_A.to(device), "im_B": im_B.to(device)}
                if h != self.h_resized or self.w_resized != w:
                    warn("Model resolution and batch resolution differ, may produce unexpected results")
                hs, ws = h, w
            finest_scale = 16
            # Run matcher
            if symmetric:
                corresps = self.forward_symmetric(batch)
            else:
                corresps = self.forward(batch, batched=True)

            if self.upsample_preds:
                hs, ws = self.upsample_res

            if self.attenuate_cert:
                low_res_certainty = F.interpolate(
                    corresps[16]["certainty"], size=(hs, ws), align_corners=False, mode="bilinear"
                )
                cert_clamp = 0
                factor = 0.5
                low_res_certainty = factor * low_res_certainty * (low_res_certainty < cert_clamp)

            if self.upsample_preds:
                finest_corresps = corresps[finest_scale]
                torch.cuda.empty_cache()
                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True
                )
                if isinstance(im_A_input, (str, os.PathLike)):
                    im_A, im_B = test_transform(
                        (Image.open(im_A_input).convert('RGB'), Image.open(im_B_input).convert('RGB')))
                else:
                    im_A, im_B = test_transform((im_A_input, im_B_input))

                im_A, im_B = im_A[None].to(device), im_B[None].to(device)
                scale_factor = math.sqrt(self.upsample_res[0] * self.upsample_res[1] / (self.w_resized * self.h_resized))
                batch = {"im_A": im_A, "im_B": im_B, "corresps": finest_corresps}
                if symmetric:
                    corresps = self.forward_symmetric(batch, upsample=True, batched=True, scale_factor=scale_factor)
                else:
                    corresps = self.forward(batch, batched=True, upsample=True, scale_factor=scale_factor)

            im_A_to_im_B = corresps[finest_scale]["flow"]
            certainty = corresps[finest_scale]["certainty"] - (low_res_certainty if self.attenuate_cert else 0)
            if finest_scale != 1:
                im_A_to_im_B = F.interpolate(
                    im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
                )
                certainty = F.interpolate(
                    certainty, size=(hs, ws), align_corners=False, mode="bilinear"
                )
            im_A_to_im_B = im_A_to_im_B.permute(
                0, 2, 3, 1
            )
            # Create im_A meshgrid
            im_A_coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                    torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
                ),
                indexing='ij'
            )
            im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
            im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
            certainty = certainty.sigmoid()  # logits -> probs
            im_A_coords = im_A_coords.permute(0, 2, 3, 1)
            if (im_A_to_im_B.abs() > 1).any() and True:
                wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
                certainty[wrong[:, None]] = 0
            im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
            if symmetric:
                A_to_B, B_to_A = im_A_to_im_B.chunk(2)
                q_warp = torch.cat((im_A_coords, A_to_B), dim=-1)
                im_B_coords = im_A_coords
                s_warp = torch.cat((B_to_A, im_B_coords), dim=-1)
                warp = torch.cat((q_warp, s_warp), dim=2)
                certainty = torch.cat(certainty.chunk(2), dim=3)
            else:
                warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
            if batched:
                return (
                    warp,
                    certainty[:, 0]
                )
            else:
                return (
                    warp[0],
                    certainty[0, 0],
                )

    def visualize_warp(self, warp, certainty, im_A = None, im_B = None, 
                       im_A_path = None, im_B_path = None, device = "cuda", symmetric = True, save_path = None, unnormalize = False):
        #assert symmetric == True, "Currently assuming bidirectional warp, might update this if someone complains ;)"
        H,W2,_ = warp.shape
        W = W2//2 if symmetric else W2
        if im_A is None:
            from PIL import Image
            im_A, im_B = Image.open(im_A_path).convert("RGB"), Image.open(im_B_path).convert("RGB")
        if not isinstance(im_A, torch.Tensor):
            im_A = im_A.resize((W,H))
            im_B = im_B.resize((W,H))    
            x_B = (torch.tensor(np.array(im_B)) / 255).to(device).permute(2, 0, 1)
            if symmetric:
                x_A = (torch.tensor(np.array(im_A)) / 255).to(device).permute(2, 0, 1)
        else:
            if symmetric:
                x_A = im_A
            x_B = im_B
        im_A_transfer_rgb = F.grid_sample(
        x_B[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
        )[0]
        if symmetric:
            im_B_transfer_rgb = F.grid_sample(
            x_A[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
            )[0]
            warp_im = torch.cat((im_A_transfer_rgb,im_B_transfer_rgb),dim=2)
            white_im = torch.ones((H,2*W),device=device)
        else:
            warp_im = im_A_transfer_rgb
            white_im = torch.ones((H, W), device = device)
        vis_im = certainty * warp_im + (1 - certainty) * white_im
        if save_path is not None:
            from romatch.utils import tensor_to_pil
            tensor_to_pil(vis_im, unnormalize=unnormalize).save(save_path)
        return vis_im


class RegressionMatcher_mast3r_dpt(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        h=448,
        w=448,
        sample_mode = "threshold_balanced",
        upsample_preds = False,
        upsample_res = (14*16*6, 14*16*6),
        symmetric = False,
        name = None,
        attenuate_cert = None,
    ):
        super().__init__()
        self.attenuate_cert = attenuate_cert
        self.encoder = encoder
        self.decoder = decoder
        self.name = name
        self.w_resized = w
        self.h_resized = h
        self.og_transforms = get_tuple_transform_ops(resize=None, normalize=True)
        self.sample_mode = sample_mode
        self.upsample_preds = upsample_preds
        self.upsample_res = upsample_res
        self.symmetric = symmetric
        self.sample_thresh = 0.05
            
    def get_output_resolution(self):
        if not self.upsample_preds:
            return self.h_resized, self.w_resized
        else:
            return self.upsample_res
    
    def extract_backbone_features(self, batch, batched = True, upsample = False):
        x_q = batch["im_A"]
        x_s = batch["im_B"]
        feature_pyramid = self.encoder((x_q, x_s), upsample = upsample)
        return feature_pyramid


    def sample(
        self,
        matches,
        certainty,
        num=10000,
    ):
        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            certainty = certainty.clone()
            certainty[certainty > upper_thresh] = 1
        matches, certainty = (
            matches.reshape(-1, 4),
            certainty.reshape(-1),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        good_samples = torch.multinomial(certainty, 
                          num_samples = min(expansion_factor*num, len(certainty)), 
                          replacement=False)
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        if "balanced" not in self.sample_mode:
            return good_matches, good_certainty
        density = kde(good_matches, std=0.1)
        p = 1 / (density+1)
        p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        balanced_samples = torch.multinomial(p, 
                          num_samples = min(num,len(good_certainty)), 
                          replacement=False)
        return good_matches[balanced_samples], good_certainty[balanced_samples]

    def forward(self, batch, end2end=False, batched = True, upsample = False, scale_factor = 1, mode = 'default'):
        feature_pyramid = self.extract_backbone_features(batch, batched=batched, upsample = upsample)
        f_q_pyramid, f_s_pyramid = feature_pyramid
        corresps = self.decoder(f_q_pyramid, 
                                f_s_pyramid, 
                                upsample = upsample, 
                                **(batch["corresps"] if "corresps" in batch else {}),
                                scale_factor=scale_factor)
        
        return corresps

    def forward_symmetric(self, batch, batched = True, upsample = False, scale_factor = 1):
        feature_pyramid = self.extract_backbone_features(batch, batched = batched, upsample = upsample)
        f_q_pyramid = {key: torch.cat([feature_pyramid[0][key], feature_pyramid[1][key]], dim=0) 
                   for key in feature_pyramid[0]}
        f_s_pyramid = {key: torch.cat([feature_pyramid[1][key], feature_pyramid[0][key]], dim=0) 
                   for key in feature_pyramid[0]}
         
        corresps = self.decoder(f_q_pyramid, 
                                f_s_pyramid, 
                                upsample = upsample, 
                                **(batch["corresps"] if "corresps" in batch else {}),
                                scale_factor=scale_factor)
        return corresps
    
    def conf_from_fb_consistency(self, flow_forward, flow_backward, th = 2):
        # assumes that flow forward is of shape (..., H, W, 2)
        has_batch = False
        if len(flow_forward.shape) == 3:
            flow_forward, flow_backward = flow_forward[None], flow_backward[None]
        else:
            has_batch = True
        H,W = flow_forward.shape[-3:-1]
        th_n = 2 * th / max(H,W)
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1 + 1 / W, 1 - 1 / W, W), 
            torch.linspace(-1 + 1 / H, 1 - 1 / H, H), indexing = "xy"),
                             dim = -1).to(flow_forward.device)
        coords_fb = F.grid_sample(
            flow_backward.permute(0, 3, 1, 2), 
            flow_forward, 
            align_corners=False, mode="bilinear").permute(0, 2, 3, 1)
        diff = (coords - coords_fb).norm(dim=-1)
        in_th = (diff < th_n).float()
        if not has_batch:
            in_th = in_th[0]
        return in_th
         
    def to_pixel_coordinates(self, coords, H_A, W_A, H_B = None, W_B = None):
        if coords.shape[-1] == 2:
            return self._to_pixel_coordinates(coords, H_A, W_A) 
        
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[...,:2], coords[...,2:]
        return self._to_pixel_coordinates(kpts_A, H_A, W_A), self._to_pixel_coordinates(kpts_B, H_B, W_B)

    def _to_pixel_coordinates(self, coords, H, W):
        kpts = torch.stack((W/2 * (coords[...,0]+1), H/2 * (coords[...,1]+1)),axis=-1)
        return kpts
 
    def to_normalized_coordinates(self, coords, H_A, W_A, H_B, W_B):
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[...,:2], coords[...,2:]
        kpts_A = torch.stack((2/W_A * kpts_A[...,0] - 1, 2/H_A * kpts_A[...,1] - 1),axis=-1)
        kpts_B = torch.stack((2/W_B * kpts_B[...,0] - 1, 2/H_B * kpts_B[...,1] - 1),axis=-1)
        return kpts_A, kpts_B

    def match_keypoints(self, x_A, x_B, warp, certainty, return_tuple = True, return_inds = False):
        x_A_to_B = F.grid_sample(warp[...,-2:].permute(2,0,1)[None], x_A[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        cert_A_to_B = F.grid_sample(certainty[None,None,...], x_A[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        D = torch.cdist(x_A_to_B, x_B)
        inds_A, inds_B = torch.nonzero((D == D.min(dim=-1, keepdim = True).values) * (D == D.min(dim=-2, keepdim = True).values) * (cert_A_to_B[:,None] > self.sample_thresh), as_tuple = True)
        
        if return_tuple:
            if return_inds:
                return inds_A, inds_B
            else:
                return x_A[inds_A], x_B[inds_B]
        else:
            if return_inds:
                return torch.cat((inds_A, inds_B),dim=-1)
            else:
                return torch.cat((x_A[inds_A], x_B[inds_B]),dim=-1)

    def match_keypoints2(self, x_A, x_B, warp, certainty): 
        assert len(warp.shape) == 3 and int(warp.shape[2]) == 4, str(warp.shape)
        H,W2,_ = warp.shape
        print(H, W2)
        W = W2//2
        # warp B keypoints into image A
        #x_A_from_B = F.grid_sample(warp[:, :, :2].permute(2,0,1)[None], x_B[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        #cert_A_from_B = F.grid_sample(certainty[None, None, :, :], x_B[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        
        x_A_from_B = F.grid_sample(warp[:, W:, :2].permute(2,0,1)[None], x_B[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        cert_A_from_B = F.grid_sample(certainty[None, None, :, W:], x_B[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        # match in the coordinate system of A
        D = torch.cdist(x_A, x_A_from_B)
        inds_A, inds_B = torch.nonzero((D == D.min(dim=-1, keepdim = True).values) * (D == D.min(dim=-2, keepdim = True).values) * (cert_A_from_B[None,:] > 0.01), as_tuple = True)
        return inds_A, inds_B, cert_A_from_B[inds_B]
        '''
        x_A_to_B = F.grid_sample(warp[...,-2:].permute(2,0,1)[None], x_A[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        cert_A_to_B = F.grid_sample(certainty[None,None,...], x_A[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        D = torch.cdist(x_A_to_B, x_B)
        inds_A, inds_B = torch.nonzero((D == D.min(dim=-1, keepdim = True).values) * (D == D.min(dim=-2, keepdim = True).values) * (cert_A_to_B[:,None] > self.sample_thresh), as_tuple = True)
        return inds_A, inds_B, cert_A_to_B[inds_A]
        '''
    @torch.inference_mode()
    def match(
        self,
        im_A_input,
        im_B_input,
        *args,
        batched=False,
        device=None,
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Check if inputs are file paths or already loaded images
        if isinstance(im_A_input, (str, os.PathLike)):
            im_A = Image.open(im_A_input).convert("RGB")
        else:
            im_A = im_A_input

        if isinstance(im_B_input, (str, os.PathLike)):
            im_B = Image.open(im_B_input).convert("RGB")
        else:
            im_B = im_B_input

        symmetric = self.symmetric
        self.train(False)
        with torch.no_grad():
            if not batched:
                b = 1
                w, h = im_A.size
                w2, h2 = im_B.size
                # Get images in good format
                ws = self.w_resized
                hs = self.h_resized

                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True, clahe=False
                )
                im_A, im_B = test_transform((im_A, im_B))
                batch = {"im_A": im_A[None].to(device), "im_B": im_B[None].to(device)}
            else:
                b, c, h, w = im_A.shape
                b, c, h2, w2 = im_B.shape
                assert w == w2 and h == h2, "For batched images we assume same size"
                batch = {"im_A": im_A.to(device), "im_B": im_B.to(device)}
                if h != self.h_resized or self.w_resized != w:
                    warn("Model resolution and batch resolution differ, may produce unexpected results")
                hs, ws = h, w
            finest_scale = 1
            # Run matcher
            if symmetric:
                corresps = self.forward_symmetric(batch)
            else:
                corresps = self.forward(batch, batched=True)

            if self.upsample_preds:
                hs, ws = self.upsample_res

            if self.attenuate_cert:
                low_res_certainty = F.interpolate(
                    corresps[16]["certainty"], size=(hs, ws), align_corners=False, mode="bilinear"
                )
                cert_clamp = 0
                factor = 0.5
                low_res_certainty = factor * low_res_certainty * (low_res_certainty < cert_clamp)

            if self.upsample_preds:
                finest_corresps = corresps[finest_scale]
                torch.cuda.empty_cache()
                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True
                )
                if isinstance(im_A_input, (str, os.PathLike)):
                    im_A, im_B = test_transform(
                        (Image.open(im_A_input).convert('RGB'), Image.open(im_B_input).convert('RGB')))
                else:
                    im_A, im_B = test_transform((im_A_input, im_B_input))

                im_A, im_B = im_A[None].to(device), im_B[None].to(device)
                scale_factor = math.sqrt(self.upsample_res[0] * self.upsample_res[1] / (self.w_resized * self.h_resized))
                batch = {"im_A": im_A, "im_B": im_B, "corresps": finest_corresps}
                if symmetric:
                    corresps = self.forward_symmetric(batch, upsample=True, batched=True, scale_factor=scale_factor)
                else:
                    corresps = self.forward(batch, batched=True, upsample=True, scale_factor=scale_factor)

            im_A_to_im_B = corresps[finest_scale]["flow"]
            certainty = corresps[finest_scale]["certainty"] - (low_res_certainty if self.attenuate_cert else 0)
            pts1 = corresps[finest_scale]['pts1']
            pts_conf1 = corresps[finest_scale]['pts_conf1']
            pts2 = corresps[finest_scale]['pts2']
            pts_conf2 = corresps[finest_scale]['pts_conf2']
            if finest_scale != 1:
                im_A_to_im_B = F.interpolate(
                    im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
                )
                certainty = F.interpolate(
                    certainty, size=(hs, ws), align_corners=False, mode="bilinear"
                )
                pts1 = F.interpolate(
                    pts1, size=(hs, ws), align_corners=False, mode="bilinear"
                )
                pts2 = F.interpolate(
                    pts2, size=(hs, ws), align_corners=False, mode="bilinear"
                )
                pts_conf1 = F.interpolate(
                    pts_conf1, size=(hs, ws), align_corners=False, mode="bilinear"
                )
                pts_conf2 = F.interpolate(
                    pts_conf2, size=(hs, ws), align_corners=False, mode="bilinear"
                )
            im_A_to_im_B = im_A_to_im_B.permute(
                0, 2, 3, 1
            )
            pts1, pts_conf1 = reg(pts1, pts_conf1)
            pts2, pts_conf2 = reg(pts2, pts_conf2)
            # Create im_A meshgrid
            im_A_coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                    torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
                ),
                indexing='ij'
            )
            im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
            im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
            certainty = certainty.sigmoid()  # logits -> probs
            im_A_coords = im_A_coords.permute(0, 2, 3, 1)
            if (im_A_to_im_B.abs() > 1).any() and True:
                wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
                certainty[wrong[:, None]] = 0
            im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
            if symmetric:
                A_to_B, B_to_A = im_A_to_im_B.chunk(2)
                q_warp = torch.cat((im_A_coords, A_to_B), dim=-1)
                im_B_coords = im_A_coords
                s_warp = torch.cat((B_to_A, im_B_coords), dim=-1)
                warp = torch.cat((q_warp, s_warp), dim=2)
                certainty = torch.cat(certainty.chunk(2), dim=3)
                # t1_pts1, t2_pts2 = pts1.chunk(2)
                # t1_pts2, t2_pts1 = pts2.chunk(2) 
                # pts1 = torch.cat((t1_pts1, t2_pts1), dim=-1)
                # pts2 = torch.cat(t1_pts1, t2_pts2), dim = -1)
            else:
                warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
            if batched:
                return (
                    warp,
                    certainty[:, 0],
                    pts1,
                    pts_conf1,
                    pts2,
                    pts_conf2,
                )
            else:
                return (
                    warp[0],
                    certainty[0, 0],
                    pts1[0],
                    pts_conf1[0],
                    pts2[0],
                    pts_conf2[0],
                )

    def visualize_warp(self, warp, certainty, im_A = None, im_B = None, 
                       im_A_path = None, im_B_path = None, device = "cuda", symmetric = True, save_path = None, unnormalize = False):
        #assert symmetric == True, "Currently assuming bidirectional warp, might update this if someone complains ;)"
        H,W2,_ = warp.shape
        W = W2//2 if symmetric else W2
        if im_A is None:
            from PIL import Image
            im_A, im_B = Image.open(im_A_path).convert("RGB"), Image.open(im_B_path).convert("RGB")
        if not isinstance(im_A, torch.Tensor):
            im_A = im_A.resize((W,H))
            im_B = im_B.resize((W,H))    
            x_B = (torch.tensor(np.array(im_B)) / 255).to(device).permute(2, 0, 1)
            if symmetric:
                x_A = (torch.tensor(np.array(im_A)) / 255).to(device).permute(2, 0, 1)
        else:
            if symmetric:
                x_A = im_A
            x_B = im_B
        im_A_transfer_rgb = F.grid_sample(
        x_B[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
        )[0]
        if symmetric:
            im_B_transfer_rgb = F.grid_sample(
            x_A[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
            )[0]
            warp_im = torch.cat((im_A_transfer_rgb,im_B_transfer_rgb),dim=2)
            white_im = torch.ones((H,2*W),device=device)
        else:
            warp_im = im_A_transfer_rgb
            white_im = torch.ones((H, W), device = device)
        vis_im = certainty * warp_im + (1 - certainty) * white_im
        if save_path is not None:
            from romatch.utils import tensor_to_pil
            tensor_to_pil(vis_im, unnormalize=unnormalize).save(save_path)
        return vis_im

class RegressionMatcher_dpt(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        h=448,
        w=448,
        sample_mode = "threshold_balanced",
        upsample_preds = False,
        upsample_res = (14*16*6, 14*16*6),
        symmetric = False,
        name = None,
        attenuate_cert = None,
    ):
        super().__init__()
        self.attenuate_cert = attenuate_cert
        self.encoder = encoder
        self.decoder = decoder
        self.name = name
        self.w_resized = w
        self.h_resized = h
        self.og_transforms = get_tuple_transform_ops(resize=None, normalize=True)
        self.sample_mode = sample_mode
        self.upsample_preds = upsample_preds
        self.upsample_res = upsample_res
        self.symmetric = symmetric
        self.sample_thresh = 0.05
            
    def get_output_resolution(self):
        if not self.upsample_preds:
            return self.h_resized, self.w_resized
        else:
            return self.upsample_res
    
    def extract_backbone_features(self, batch, batched = True, upsample = False):
        x_q = batch["im_A"]
        x_s = batch["im_B"]
        if batched:
            X = torch.cat((x_q, x_s), dim = 0)
            feature_pyramid = self.encoder(X, upsample = upsample)
        else:
            feature_pyramid = self.encoder(x_q, upsample = upsample), self.encoder(x_s, upsample = upsample)
        return feature_pyramid


    def sample(
        self,
        matches,
        certainty,
        num=10000,
    ):
        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            certainty = certainty.clone()
            certainty[certainty > upper_thresh] = 1
        matches, certainty = (
            matches.reshape(-1, 4),
            certainty.reshape(-1),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        good_samples = torch.multinomial(certainty, 
                          num_samples = min(expansion_factor*num, len(certainty)), 
                          replacement=False)
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        if "balanced" not in self.sample_mode:
            return good_matches, good_certainty
        density = kde(good_matches, std=0.1)
        p = 1 / (density+1)
        p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        balanced_samples = torch.multinomial(p, 
                          num_samples = min(num,len(good_certainty)), 
                          replacement=False)
        return good_matches[balanced_samples], good_certainty[balanced_samples]
    
        
    def forward(self, batch, batched = True, upsample = False, scale_factor = 1):
        feature_pyramid = self.extract_backbone_features(batch, batched=batched, upsample = upsample)
        if batched:
            f_q_pyramid = {
                scale: f_scale.chunk(2)[0] for scale, f_scale in feature_pyramid.items()
            }
            f_s_pyramid = {
                scale: f_scale.chunk(2)[1] for scale, f_scale in feature_pyramid.items()
            }
        else:
            f_q_pyramid, f_s_pyramid = feature_pyramid
        corresps = self.decoder(f_q_pyramid, 
                                f_s_pyramid, 
                                upsample = upsample, 
                                **(batch["corresps"] if "corresps" in batch else {}),
                                scale_factor=scale_factor)
        return corresps
        
    def forward_symmetric(self, batch, batched = True, upsample = False, scale_factor = 1):
        feature_pyramid = self.extract_backbone_features(batch, batched = batched, upsample = upsample)
        f_q_pyramid = feature_pyramid
        f_s_pyramid = {
            scale: torch.cat((f_scale.chunk(2)[1], f_scale.chunk(2)[0]), dim = 0)
            for scale, f_scale in feature_pyramid.items()
        }
        corresps = self.decoder(f_q_pyramid, 
                                f_s_pyramid, 
                                upsample = upsample, 
                                **(batch["corresps"] if "corresps" in batch else {}),
                                scale_factor=scale_factor)
        return corresps
    
    def conf_from_fb_consistency(self, flow_forward, flow_backward, th = 2):
        # assumes that flow forward is of shape (..., H, W, 2)
        has_batch = False
        if len(flow_forward.shape) == 3:
            flow_forward, flow_backward = flow_forward[None], flow_backward[None]
        else:
            has_batch = True
        H,W = flow_forward.shape[-3:-1]
        th_n = 2 * th / max(H,W)
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1 + 1 / W, 1 - 1 / W, W), 
            torch.linspace(-1 + 1 / H, 1 - 1 / H, H), indexing = "xy"),
                             dim = -1).to(flow_forward.device)
        coords_fb = F.grid_sample(
            flow_backward.permute(0, 3, 1, 2), 
            flow_forward, 
            align_corners=False, mode="bilinear").permute(0, 2, 3, 1)
        diff = (coords - coords_fb).norm(dim=-1)
        in_th = (diff < th_n).float()
        if not has_batch:
            in_th = in_th[0]
        return in_th
         
    def to_pixel_coordinates(self, coords, H_A, W_A, H_B = None, W_B = None):
        if coords.shape[-1] == 2:
            return self._to_pixel_coordinates(coords, H_A, W_A) 
        
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[...,:2], coords[...,2:]
        return self._to_pixel_coordinates(kpts_A, H_A, W_A), self._to_pixel_coordinates(kpts_B, H_B, W_B)

    def _to_pixel_coordinates(self, coords, H, W):
        kpts = torch.stack((W/2 * (coords[...,0]+1), H/2 * (coords[...,1]+1)),axis=-1)
        return kpts
 
    def to_normalized_coordinates(self, coords, H_A, W_A, H_B, W_B):
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[...,:2], coords[...,2:]
        kpts_A = torch.stack((2/W_A * kpts_A[...,0] - 1, 2/H_A * kpts_A[...,1] - 1),axis=-1)
        kpts_B = torch.stack((2/W_B * kpts_B[...,0] - 1, 2/H_B * kpts_B[...,1] - 1),axis=-1)
        return kpts_A, kpts_B

    def match_keypoints(self, x_A, x_B, warp, certainty, return_tuple = True, return_inds = False):
        x_A_to_B = F.grid_sample(warp[...,-2:].permute(2,0,1)[None], x_A[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        cert_A_to_B = F.grid_sample(certainty[None,None,...], x_A[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        D = torch.cdist(x_A_to_B, x_B)
        inds_A, inds_B = torch.nonzero((D == D.min(dim=-1, keepdim = True).values) * (D == D.min(dim=-2, keepdim = True).values) * (cert_A_to_B[:,None] > self.sample_thresh), as_tuple = True)
        
        if return_tuple:
            if return_inds:
                return inds_A, inds_B
            else:
                return x_A[inds_A], x_B[inds_B]
        else:
            if return_inds:
                return torch.cat((inds_A, inds_B),dim=-1)
            else:
                return torch.cat((x_A[inds_A], x_B[inds_B]),dim=-1)

    def match_keypoints2(self, x_A, x_B, warp, certainty): 
        assert len(warp.shape) == 3 and int(warp.shape[2]) == 4, str(warp.shape)
        H,W2,_ = warp.shape
        print(H, W2)
        W = W2//2
        # warp B keypoints into image A
        #x_A_from_B = F.grid_sample(warp[:, :, :2].permute(2,0,1)[None], x_B[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        #cert_A_from_B = F.grid_sample(certainty[None, None, :, :], x_B[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        
        x_A_from_B = F.grid_sample(warp[:, W:, :2].permute(2,0,1)[None], x_B[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        cert_A_from_B = F.grid_sample(certainty[None, None, :, W:], x_B[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        # match in the coordinate system of A
        D = torch.cdist(x_A, x_A_from_B)
        inds_A, inds_B = torch.nonzero((D == D.min(dim=-1, keepdim = True).values) * (D == D.min(dim=-2, keepdim = True).values) * (cert_A_from_B[None,:] > 0.01), as_tuple = True)
        return inds_A, inds_B, cert_A_from_B[inds_B]
        '''
        x_A_to_B = F.grid_sample(warp[...,-2:].permute(2,0,1)[None], x_A[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        cert_A_to_B = F.grid_sample(certainty[None,None,...], x_A[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        D = torch.cdist(x_A_to_B, x_B)
        inds_A, inds_B = torch.nonzero((D == D.min(dim=-1, keepdim = True).values) * (D == D.min(dim=-2, keepdim = True).values) * (cert_A_to_B[:,None] > self.sample_thresh), as_tuple = True)
        return inds_A, inds_B, cert_A_to_B[inds_A]
        '''
    @torch.inference_mode()
    def match(
        self,
        im_A_input,
        im_B_input,
        *args,
        batched=False,
        device=None,
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Check if inputs are file paths or already loaded images
        if isinstance(im_A_input, (str, os.PathLike)):
            im_A = Image.open(im_A_input).convert("RGB")
        else:
            im_A = im_A_input

        if isinstance(im_B_input, (str, os.PathLike)):
            im_B = Image.open(im_B_input).convert("RGB")
        else:
            im_B = im_B_input

        symmetric = self.symmetric
        self.train(False)
        with torch.no_grad():
            if not batched:
                b = 1
                w, h = im_A.size
                w2, h2 = im_B.size
                # Get images in good format
                ws = self.w_resized
                hs = self.h_resized

                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True, clahe=False
                )
                im_A, im_B = test_transform((im_A, im_B))
                batch = {"im_A": im_A[None].to(device), "im_B": im_B[None].to(device)}
            else:
                b, c, h, w = im_A.shape
                b, c, h2, w2 = im_B.shape
                assert w == w2 and h == h2, "For batched images we assume same size"
                batch = {"im_A": im_A.to(device), "im_B": im_B.to(device)}
                if h != self.h_resized or self.w_resized != w:
                    warn("Model resolution and batch resolution differ, may produce unexpected results")
                hs, ws = h, w
            finest_scale = 1
            # Run matcher
            if symmetric:
                corresps = self.forward_symmetric(batch)
            else:
                corresps = self.forward(batch, batched=True)

            if self.upsample_preds:
                hs, ws = self.upsample_res

            if self.attenuate_cert:
                low_res_certainty = F.interpolate(
                    corresps[16]["certainty"], size=(hs, ws), align_corners=False, mode="bilinear"
                )
                cert_clamp = 0
                factor = 0.5
                low_res_certainty = factor * low_res_certainty * (low_res_certainty < cert_clamp)

            if self.upsample_preds:
                finest_corresps = corresps[finest_scale]
                torch.cuda.empty_cache()
                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True
                )
                if isinstance(im_A_input, (str, os.PathLike)):
                    im_A, im_B = test_transform(
                        (Image.open(im_A_input).convert('RGB'), Image.open(im_B_input).convert('RGB')))
                else:
                    im_A, im_B = test_transform((im_A_input, im_B_input))

                im_A, im_B = im_A[None].to(device), im_B[None].to(device)
                scale_factor = math.sqrt(self.upsample_res[0] * self.upsample_res[1] / (self.w_resized * self.h_resized))
                batch = {"im_A": im_A, "im_B": im_B, "corresps": finest_corresps}
                if symmetric:
                    corresps = self.forward_symmetric(batch, upsample=True, batched=True, scale_factor=scale_factor)
                else:
                    corresps = self.forward(batch, batched=True, upsample=True, scale_factor=scale_factor)

            im_A_to_im_B = corresps[finest_scale]["flow"]
            certainty = corresps[finest_scale]["certainty"] - (low_res_certainty if self.attenuate_cert else 0)
            pts1 = corresps[finest_scale]['pts1']
            pts_conf1 = corresps[finest_scale]['pts_conf1']
            pts2 = corresps[finest_scale]['pts2']
            pts_conf2 = corresps[finest_scale]['pts_conf2']
            if finest_scale != 1:
                im_A_to_im_B = F.interpolate(
                    im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
                )
                certainty = F.interpolate(
                    certainty, size=(hs, ws), align_corners=False, mode="bilinear"
                )
                pts1 = F.interpolate(
                    pts1, size=(hs, ws), align_corners=False, mode="bilinear"
                )
                pts2 = F.interpolate(
                    pts2, size=(hs, ws), align_corners=False, mode="bilinear"
                )
                pts_conf1 = F.interpolate(
                    pts_conf1, size=(hs, ws), align_corners=False, mode="bilinear"
                )
                pts_conf2 = F.interpolate(
                    pts_conf2, size=(hs, ws), align_corners=False, mode="bilinear"
                )
            im_A_to_im_B = im_A_to_im_B.permute(
                0, 2, 3, 1
            )
            pts1, pts_conf1 = reg(pts1, pts_conf1)
            pts2, pts_conf2 = reg(pts2, pts_conf2)
            # Create im_A meshgrid
            im_A_coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                    torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
                ),
                indexing='ij'
            )
            im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
            im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
            certainty = certainty.sigmoid()  # logits -> probs
            im_A_coords = im_A_coords.permute(0, 2, 3, 1)
            if (im_A_to_im_B.abs() > 1).any() and True:
                wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
                certainty[wrong[:, None]] = 0
            im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
            if symmetric:
                A_to_B, B_to_A = im_A_to_im_B.chunk(2)
                q_warp = torch.cat((im_A_coords, A_to_B), dim=-1)
                im_B_coords = im_A_coords
                s_warp = torch.cat((B_to_A, im_B_coords), dim=-1)
                warp = torch.cat((q_warp, s_warp), dim=2)
                certainty = torch.cat(certainty.chunk(2), dim=3)
                # t1_pts1, t2_pts2 = pts1.chunk(2)
                # t1_pts2, t2_pts1 = pts2.chunk(2) 
                # pts1 = torch.cat((t1_pts1, t2_pts1), dim=-1)
                # pts2 = torch.cat(t1_pts1, t2_pts2), dim = -1)
            else:
                warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
            if batched:
                return (
                    warp,
                    certainty[:, 0],
                    pts1,
                    pts_conf1,
                    pts2,
                    pts_conf2,
                )
            else:
                return (
                    warp[0],
                    certainty[0, 0],
                    pts1[0],
                    pts_conf1[0],
                    pts2[0],
                    pts_conf2[0],
                )

    def visualize_warp(self, warp, certainty, im_A = None, im_B = None, 
                       im_A_path = None, im_B_path = None, device = "cuda", symmetric = True, save_path = None, unnormalize = False):
        #assert symmetric == True, "Currently assuming bidirectional warp, might update this if someone complains ;)"
        H,W2,_ = warp.shape
        W = W2//2 if symmetric else W2
        if im_A is None:
            from PIL import Image
            im_A, im_B = Image.open(im_A_path).convert("RGB"), Image.open(im_B_path).convert("RGB")
        if not isinstance(im_A, torch.Tensor):
            im_A = im_A.resize((W,H))
            im_B = im_B.resize((W,H))    
            x_B = (torch.tensor(np.array(im_B)) / 255).to(device).permute(2, 0, 1)
            if symmetric:
                x_A = (torch.tensor(np.array(im_A)) / 255).to(device).permute(2, 0, 1)
        else:
            if symmetric:
                x_A = im_A
            x_B = im_B
        im_A_transfer_rgb = F.grid_sample(
        x_B[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
        )[0]
        if symmetric:
            im_B_transfer_rgb = F.grid_sample(
            x_A[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
            )[0]
            warp_im = torch.cat((im_A_transfer_rgb,im_B_transfer_rgb),dim=2)
            white_im = torch.ones((H,2*W),device=device)
        else:
            warp_im = im_A_transfer_rgb
            white_im = torch.ones((H, W), device = device)
        vis_im = certainty * warp_im + (1 - certainty) * white_im
        if save_path is not None:
            from romatch.utils import tensor_to_pil
            tensor_to_pil(vis_im, unnormalize=unnormalize).save(save_path)
        return vis_im


class RegressionMatcher_only_dpt(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        h=448,
        w=448,
        sample_mode = "threshold_balanced",
        upsample_preds = False,
        upsample_res = (14*16*6, 14*16*6),
        symmetric = False,
        name = None,
        attenuate_cert = None,
    ):
        super().__init__()
        self.attenuate_cert = attenuate_cert
        self.encoder = encoder
        self.decoder = decoder
        self.name = name
        self.w_resized = w
        self.h_resized = h
        self.og_transforms = get_tuple_transform_ops(resize=None, normalize=True)
        self.sample_mode = sample_mode
        self.upsample_preds = upsample_preds
        self.upsample_res = upsample_res
        self.symmetric = symmetric
        self.sample_thresh = 0.05
            
    def get_output_resolution(self):
        if not self.upsample_preds:
            return self.h_resized, self.w_resized
        else:
            return self.upsample_res
    
    def extract_backbone_features(self, batch, batched = True, upsample = False):
        x_q = batch["im_A"]
        x_s = batch["im_B"]
        if batched:
            X = torch.cat((x_q, x_s), dim = 0)
            feature_pyramid = self.encoder(X, upsample = upsample)
        else:
            feature_pyramid = self.encoder(x_q, upsample = upsample), self.encoder(x_s, upsample = upsample)
        return feature_pyramid
        
    def forward(self, batch, batched = True, upsample = False, scale_factor = 1):
        feature_pyramid = self.extract_backbone_features(batch, batched=batched, upsample = upsample)
        if batched:
            f_q_pyramid = {
                scale: f_scale.chunk(2)[0] for scale, f_scale in feature_pyramid.items()
            }
            f_s_pyramid = {
                scale: f_scale.chunk(2)[1] for scale, f_scale in feature_pyramid.items()
            }
        else:
            f_q_pyramid, f_s_pyramid = feature_pyramid
        corresps = self.decoder(f_q_pyramid, 
                                f_s_pyramid, 
                                upsample = upsample, 
                                **(batch["corresps"] if "corresps" in batch else {}),
                                scale_factor=scale_factor)
        return corresps
        
    def forward_symmetric(self, batch, batched = True, upsample = False, scale_factor = 1):
        feature_pyramid = self.extract_backbone_features(batch, batched = batched, upsample = upsample)
        f_q_pyramid = feature_pyramid
        f_s_pyramid = {
            scale: torch.cat((f_scale.chunk(2)[1], f_scale.chunk(2)[0]), dim = 0)
            for scale, f_scale in feature_pyramid.items()
        }
        corresps = self.decoder(f_q_pyramid, 
                                f_s_pyramid, 
                                upsample = upsample, 
                                **(batch["corresps"] if "corresps" in batch else {}),
                                scale_factor=scale_factor)
        return corresps
    
    def conf_from_fb_consistency(self, flow_forward, flow_backward, th = 2):
        # assumes that flow forward is of shape (..., H, W, 2)
        has_batch = False
        if len(flow_forward.shape) == 3:
            flow_forward, flow_backward = flow_forward[None], flow_backward[None]
        else:
            has_batch = True
        H,W = flow_forward.shape[-3:-1]
        th_n = 2 * th / max(H,W)
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1 + 1 / W, 1 - 1 / W, W), 
            torch.linspace(-1 + 1 / H, 1 - 1 / H, H), indexing = "xy"),
                             dim = -1).to(flow_forward.device)
        coords_fb = F.grid_sample(
            flow_backward.permute(0, 3, 1, 2), 
            flow_forward, 
            align_corners=False, mode="bilinear").permute(0, 2, 3, 1)
        diff = (coords - coords_fb).norm(dim=-1)
        in_th = (diff < th_n).float()
        if not has_batch:
            in_th = in_th[0]
        return in_th
         
    def to_pixel_coordinates(self, coords, H_A, W_A, H_B = None, W_B = None):
        if coords.shape[-1] == 2:
            return self._to_pixel_coordinates(coords, H_A, W_A) 
        
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[...,:2], coords[...,2:]
        return self._to_pixel_coordinates(kpts_A, H_A, W_A), self._to_pixel_coordinates(kpts_B, H_B, W_B)

    def _to_pixel_coordinates(self, coords, H, W):
        kpts = torch.stack((W/2 * (coords[...,0]+1), H/2 * (coords[...,1]+1)),axis=-1)
        return kpts
 
    def to_normalized_coordinates(self, coords, H_A, W_A, H_B, W_B):
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[...,:2], coords[...,2:]
        kpts_A = torch.stack((2/W_A * kpts_A[...,0] - 1, 2/H_A * kpts_A[...,1] - 1),axis=-1)
        kpts_B = torch.stack((2/W_B * kpts_B[...,0] - 1, 2/H_B * kpts_B[...,1] - 1),axis=-1)
        return kpts_A, kpts_B

    def match_keypoints(self, x_A, x_B, warp, certainty, return_tuple = True, return_inds = False):
        x_A_to_B = F.grid_sample(warp[...,-2:].permute(2,0,1)[None], x_A[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        cert_A_to_B = F.grid_sample(certainty[None,None,...], x_A[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        D = torch.cdist(x_A_to_B, x_B)
        inds_A, inds_B = torch.nonzero((D == D.min(dim=-1, keepdim = True).values) * (D == D.min(dim=-2, keepdim = True).values) * (cert_A_to_B[:,None] > self.sample_thresh), as_tuple = True)
        
        if return_tuple:
            if return_inds:
                return inds_A, inds_B
            else:
                return x_A[inds_A], x_B[inds_B]
        else:
            if return_inds:
                return torch.cat((inds_A, inds_B),dim=-1)
            else:
                return torch.cat((x_A[inds_A], x_B[inds_B]),dim=-1)

    def match_keypoints2(self, x_A, x_B, warp, certainty): 
        assert len(warp.shape) == 3 and int(warp.shape[2]) == 4, str(warp.shape)
        H,W2,_ = warp.shape
        print(H, W2)
        W = W2//2
        # warp B keypoints into image A
        #x_A_from_B = F.grid_sample(warp[:, :, :2].permute(2,0,1)[None], x_B[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        #cert_A_from_B = F.grid_sample(certainty[None, None, :, :], x_B[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        
        x_A_from_B = F.grid_sample(warp[:, W:, :2].permute(2,0,1)[None], x_B[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        cert_A_from_B = F.grid_sample(certainty[None, None, :, W:], x_B[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        # match in the coordinate system of A
        D = torch.cdist(x_A, x_A_from_B)
        inds_A, inds_B = torch.nonzero((D == D.min(dim=-1, keepdim = True).values) * (D == D.min(dim=-2, keepdim = True).values) * (cert_A_from_B[None,:] > 0.01), as_tuple = True)
        return inds_A, inds_B, cert_A_from_B[inds_B]
        '''
        x_A_to_B = F.grid_sample(warp[...,-2:].permute(2,0,1)[None], x_A[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
        cert_A_to_B = F.grid_sample(certainty[None,None,...], x_A[None,None], align_corners = False, mode = "bilinear")[0,0,0]
        D = torch.cdist(x_A_to_B, x_B)
        inds_A, inds_B = torch.nonzero((D == D.min(dim=-1, keepdim = True).values) * (D == D.min(dim=-2, keepdim = True).values) * (cert_A_to_B[:,None] > self.sample_thresh), as_tuple = True)
        return inds_A, inds_B, cert_A_to_B[inds_A]
        '''
    @torch.inference_mode()
    def match(
        self,
        im_A_input,
        im_B_input,
        *args,
        batched=False,
        device=None,
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Check if inputs are file paths or already loaded images
        if isinstance(im_A_input, (str, os.PathLike)):
            im_A = Image.open(im_A_input).convert("RGB")
        else:
            im_A = im_A_input

        if isinstance(im_B_input, (str, os.PathLike)):
            im_B = Image.open(im_B_input).convert("RGB")
        else:
            im_B = im_B_input

        symmetric = self.symmetric
        self.train(False)
        with torch.no_grad():
            if not batched:
                b = 1
                w, h = im_A.size
                w2, h2 = im_B.size
                # Get images in good format
                ws = self.w_resized
                hs = self.h_resized

                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True, clahe=False
                )
                im_A, im_B = test_transform((im_A, im_B))
                batch = {"im_A": im_A[None].to(device), "im_B": im_B[None].to(device)}
            else:
                b, c, h, w = im_A.shape
                b, c, h2, w2 = im_B.shape
                assert w == w2 and h == h2, "For batched images we assume same size"
                batch = {"im_A": im_A.to(device), "im_B": im_B.to(device)}
                if h != self.h_resized or self.w_resized != w:
                    warn("Model resolution and batch resolution differ, may produce unexpected results")
                hs, ws = h, w
            finest_scale = 1
            # Run matcher
            if symmetric:
                corresps = self.forward_symmetric(batch)
            else:
                corresps = self.forward(batch, batched=True)

            if self.upsample_preds:
                hs, ws = self.upsample_res

            if self.attenuate_cert:
                low_res_certainty = F.interpolate(
                    corresps[16]["certainty"], size=(hs, ws), align_corners=False, mode="bilinear"
                )
                cert_clamp = 0
                factor = 0.5
                low_res_certainty = factor * low_res_certainty * (low_res_certainty < cert_clamp)

            if self.upsample_preds:
                finest_corresps = corresps[finest_scale]
                torch.cuda.empty_cache()
                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True
                )
                if isinstance(im_A_input, (str, os.PathLike)):
                    im_A, im_B = test_transform(
                        (Image.open(im_A_input).convert('RGB'), Image.open(im_B_input).convert('RGB')))
                else:
                    im_A, im_B = test_transform((im_A_input, im_B_input))

                im_A, im_B = im_A[None].to(device), im_B[None].to(device)
                scale_factor = math.sqrt(self.upsample_res[0] * self.upsample_res[1] / (self.w_resized * self.h_resized))
                batch = {"im_A": im_A, "im_B": im_B, "corresps": finest_corresps}
                if symmetric:
                    corresps = self.forward_symmetric(batch, upsample=True, batched=True, scale_factor=scale_factor)
                else:
                    corresps = self.forward(batch, batched=True, upsample=True, scale_factor=scale_factor)

            im_A_to_im_B = corresps[finest_scale]["flow"]
            certainty = corresps[finest_scale]["certainty"] - (low_res_certainty if self.attenuate_cert else 0)
            pts1 = corresps[finest_scale]['pts1']
            pts_conf1 = corresps[finest_scale]['pts_conf1']
            pts2 = corresps[finest_scale]['pts2']
            pts_conf2 = corresps[finest_scale]['pts_conf2']
            if finest_scale != 1:
                im_A_to_im_B = F.interpolate(
                    im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
                )
                certainty = F.interpolate(
                    certainty, size=(hs, ws), align_corners=False, mode="bilinear"
                )
                pts1 = F.interpolate(
                    pts1, size=(hs, ws), align_corners=False, mode="bilinear"
                )
                pts2 = F.interpolate(
                    pts2, size=(hs, ws), align_corners=False, mode="bilinear"
                )
                pts_conf1 = F.interpolate(
                    pts_conf1, size=(hs, ws), align_corners=False, mode="bilinear"
                )
                pts_conf2 = F.interpolate(
                    pts_conf2, size=(hs, ws), align_corners=False, mode="bilinear"
                )
            im_A_to_im_B = im_A_to_im_B.permute(
                0, 2, 3, 1
            )
            pts1, pts_conf1 = reg(pts1, pts_conf1)
            pts2, pts_conf2 = reg(pts2, pts_conf2)
            # Create im_A meshgrid
            im_A_coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                    torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
                ),
                indexing='ij'
            )
            im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
            im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
            certainty = certainty.sigmoid()  # logits -> probs
            im_A_coords = im_A_coords.permute(0, 2, 3, 1)
            if (im_A_to_im_B.abs() > 1).any() and True:
                wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
                certainty[wrong[:, None]] = 0
            im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
            if symmetric:
                A_to_B, B_to_A = im_A_to_im_B.chunk(2)
                q_warp = torch.cat((im_A_coords, A_to_B), dim=-1)
                im_B_coords = im_A_coords
                s_warp = torch.cat((B_to_A, im_B_coords), dim=-1)
                warp = torch.cat((q_warp, s_warp), dim=2)
                certainty = torch.cat(certainty.chunk(2), dim=3)
                # t1_pts1, t2_pts2 = pts1.chunk(2)
                # t1_pts2, t2_pts1 = pts2.chunk(2) 
                # pts1 = torch.cat((t1_pts1, t2_pts1), dim=-1)
                # pts2 = torch.cat(t1_pts1, t2_pts2), dim = -1)
            else:
                warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
            if batched:
                return (
                    warp,
                    certainty[:, 0],
                    pts1,
                    pts_conf1,
                    pts2,
                    pts_conf2,
                )
            else:
                return (
                    warp[0],
                    certainty[0, 0],
                    pts1[0],
                    pts_conf1[0],
                    pts2[0],
                    pts_conf2[0],
                )

    def visualize_warp(self, warp, certainty, im_A = None, im_B = None, 
                       im_A_path = None, im_B_path = None, device = "cuda", symmetric = True, save_path = None, unnormalize = False):
        #assert symmetric == True, "Currently assuming bidirectional warp, might update this if someone complains ;)"
        H,W2,_ = warp.shape
        W = W2//2 if symmetric else W2
        if im_A is None:
            from PIL import Image
            im_A, im_B = Image.open(im_A_path).convert("RGB"), Image.open(im_B_path).convert("RGB")
        if not isinstance(im_A, torch.Tensor):
            im_A = im_A.resize((W,H))
            im_B = im_B.resize((W,H))    
            x_B = (torch.tensor(np.array(im_B)) / 255).to(device).permute(2, 0, 1)
            if symmetric:
                x_A = (torch.tensor(np.array(im_A)) / 255).to(device).permute(2, 0, 1)
        else:
            if symmetric:
                x_A = im_A
            x_B = im_B
        im_A_transfer_rgb = F.grid_sample(
        x_B[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
        )[0]
        if symmetric:
            im_B_transfer_rgb = F.grid_sample(
            x_A[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
            )[0]
            warp_im = torch.cat((im_A_transfer_rgb,im_B_transfer_rgb),dim=2)
            white_im = torch.ones((H,2*W),device=device)
        else:
            warp_im = im_A_transfer_rgb
            white_im = torch.ones((H, W), device = device)
        vis_im = certainty * warp_im + (1 - certainty) * white_im
        if save_path is not None:
            from romatch.utils import tensor_to_pil
            tensor_to_pil(vis_im, unnormalize=unnormalize).save(save_path)
        return vis_im


def reg(xyz, conf):
    xyz = xyz.permute(0, 2, 3, 1)
    conf = conf.permute(0, 2, 3, 1)
    conf = conf[...,0]
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)
    xyz = xyz * torch.expm1(d)
    conf = 1 + conf.exp().clip(max=float('inf')-1)
    return xyz, conf