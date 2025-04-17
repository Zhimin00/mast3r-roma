# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R heads
# --------------------------------------------------------
import torch
import torch.nn.functional as F

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.heads.postprocess import reg_dense_depth, reg_dense_conf  # noqa
from dust3r.heads.dpt_head import PixelwiseTaskWithDPT, PixelwiseTaskWithDPT_resnet  # noqa
import dust3r.utils.path_to_croco  # noqa
from models.blocks import Mlp  # noqa
import pdb
from romatch.models.transformer import Block, TransformerDecoder, MemEffAttention
from romatch.utils.utils import cls_to_flow_refine, get_autocast_params, cls_to_flow
from romatch.utils.local_correlation import local_correlation
from romatch.utils.kde import kde
from einops import rearrange
import torch.nn as nn
import warnings
import math

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
            decoder
    ):
        super().__init__()
        self.decoder = decoder

    def forward(self, f_q_pyramid, f_s_pyramid, upsample = False, scale_factor = 1):
        corresps = self.decoder(f_q_pyramid, 
                                f_s_pyramid, 
                                upsample = upsample, 
                                scale_factor=scale_factor)
        return corresps
    
    
def reg_desc(desc, mode):
    if 'norm' in mode:
        desc = desc / desc.norm(dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unknown desc mode {mode}")
    return desc


def postprocess(out, depth_mode, conf_mode, desc_dim=None, desc_mode='norm', two_confs=False, desc_conf_mode=None):
    if desc_conf_mode is None:
        desc_conf_mode = conf_mode
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,D
    res = dict(pts3d=reg_dense_depth(fmap[..., 0:3], mode=depth_mode))
    if conf_mode is not None:
        res['conf'] = reg_dense_conf(fmap[..., 3], mode=conf_mode)
    if desc_dim is not None:
        start = 3 + int(conf_mode is not None)
        res['desc'] = reg_desc(fmap[..., start:start + desc_dim], mode=desc_mode)
        if two_confs:
            res['desc_conf'] = reg_dense_conf(fmap[..., start + desc_dim], mode=desc_conf_mode)
        else:
            res['desc_conf'] = res['conf'].clone()
    return res


class Cat_MLP_LocalFeatures_DPT_Pts3d(PixelwiseTaskWithDPT):
    """ Mixture between MLP and DPT head that outputs 3d points and local features (with MLP).
    The input for both heads is a concatenation of Encoder and Decoder outputs
    """

    def __init__(self, net, has_conf=False, local_feat_dim=16, hidden_dim_factor=4., hooks_idx=None, dim_tokens=None,
                 num_channels=1, postprocess=None, feature_dim=256, last_dim=32, depth_mode=None, conf_mode=None, head_type="regression", **kwargs):
        super().__init__(num_channels=num_channels, feature_dim=feature_dim, last_dim=last_dim, hooks_idx=hooks_idx,
                         dim_tokens=dim_tokens, depth_mode=depth_mode, postprocess=postprocess, conf_mode=conf_mode, head_type=head_type)
        self.local_feat_dim = local_feat_dim

        patch_size = net.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2 and isinstance(patch_size[0], int) and isinstance(
                patch_size[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
            assert patch_size[0] == patch_size[1], "Error, non square patches not managed"
            patch_size = patch_size[0]
        self.patch_size = patch_size

        self.desc_mode = net.desc_mode
        self.has_conf = has_conf
        self.two_confs = net.two_confs  # independent confs for 3D regr and descs
        self.desc_conf_mode = net.desc_conf_mode
        idim = net.enc_embed_dim + net.dec_embed_dim

        self.head_local_features = Mlp(in_features=idim,
                                       hidden_features=int(hidden_dim_factor * idim),
                                       out_features=(self.local_feat_dim + self.two_confs) * self.patch_size**2)

    def forward(self, decout, img_shape, mode = 'default'):
        # pass through the heads
        if mode == 'mlp_with_conf':
            # recover encoder and decoder outputs
            enc_output, dec_output = decout[0], decout[-1]
            cat_output = torch.cat([enc_output, dec_output], dim=-1)  # concatenate
            H, W = img_shape
            B, S, D = cat_output.shape
            # extract local_features
            local_features = self.head_local_features(cat_output)  # B,S,D
            local_features = local_features.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
            
            return local_features
        elif mode == 'mlp_without_conf':
            # recover encoder and decoder outputs
            pts3d = self.dpt(decout, image_size=(img_shape[0], img_shape[1]))

            # recover encoder and decoder outputs
            enc_output, dec_output = decout[0], decout[-1]
            cat_output = torch.cat([enc_output, dec_output], dim=-1)  # concatenate
            H, W = img_shape
            B, S, D = cat_output.shape

            # extract local_features
            local_features = self.head_local_features(cat_output)  # B,S,D
            local_features = local_features.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
            local_features = F.pixel_shuffle(local_features, self.patch_size)  # B,d,H,W

            # post process 3D pts, descriptors and confidences
            out = torch.cat([pts3d, local_features], dim=1)
            if self.postprocess:
                out = self.postprocess(out,
                                    depth_mode=self.depth_mode,
                                    conf_mode=self.conf_mode,
                                    desc_dim=self.local_feat_dim,
                                    desc_mode=self.desc_mode,
                                    two_confs=self.two_confs,
                                    desc_conf_mode=self.desc_conf_mode)
            res = out['desc'].permute(0,3,1,2)
            res = F.pixel_unshuffle(res, self.patch_size)
            return res
        elif mode == 'refine':
            pts3d = self.dpt(decout, image_size=(img_shape[0], img_shape[1]))
            # recover encoder and decoder outputs
            enc_output, dec_output = decout[0], decout[-1]
            cat_output = torch.cat([enc_output, dec_output], dim=-1)  # concatenate
            H, W = img_shape
            B, S, D = cat_output.shape

            # extract local_features
            local_features = self.head_local_features(cat_output)  # B,S,D
            local_features = local_features.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
            local_features = F.pixel_shuffle(local_features, self.patch_size)  # B,d,H,W
            out = torch.cat([pts3d, local_features], dim=1)
            if self.postprocess:
                out = self.postprocess(out,
                                    depth_mode=self.depth_mode,
                                    conf_mode=self.conf_mode,
                                    desc_dim=self.local_feat_dim,
                                    desc_mode=self.desc_mode,
                                    two_confs=self.two_confs,
                                    desc_conf_mode=self.desc_conf_mode)
            return out, pts3d
        else: 
            pts3d = self.dpt(decout, image_size=(img_shape[0], img_shape[1]))
            # recover encoder and decoder outputs
            enc_output, dec_output = decout[0], decout[-1]
            cat_output = torch.cat([enc_output, dec_output], dim=-1)  # concatenate
            H, W = img_shape
            B, S, D = cat_output.shape

            # extract local_features
            local_features = self.head_local_features(cat_output)  # B,S,D
            local_features = local_features.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
            local_features = F.pixel_shuffle(local_features, self.patch_size)  # B,d,H,W

            # post process 3D pts, descriptors and confidences
            out = torch.cat([pts3d, local_features], dim=1)
            if self.postprocess:
                out = self.postprocess(out,
                                    depth_mode=self.depth_mode,
                                    conf_mode=self.conf_mode,
                                    desc_dim=self.local_feat_dim,
                                    desc_mode=self.desc_mode,
                                    two_confs=self.two_confs,
                                    desc_conf_mode=self.desc_conf_mode)
            return out

class PixelwiseTaskWithDPT_cat_cnn(PixelwiseTaskWithDPT_resnet):
    """ cat cnn features to DPT head that outputs 3d points
    The input for both heads is a concatenation of Encoder and Decoder outputs
    """

    def __init__(self, net, **kwargs):
        super().__init__(**kwargs)
        patch_size = net.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2 and isinstance(patch_size[0], int) and isinstance(
                patch_size[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
            assert patch_size[0] == patch_size[1], "Error, non square patches not managed"
            patch_size = patch_size[0]
        self.patch_size = patch_size

    def forward(self, decout, cnn_feats, img_shape):
        feat1, feat2, feat4, feat8 = cnn_feats
        out = self.dpt(decout, feat4, feat8, image_size=(img_shape[0], img_shape[1]))
        if self.postprocess:
            out = self.postprocess(out, self.depth_mode, self.conf_mode)
        enc_output1, dec_output1 = decout[0], decout[-1]
        
        H, W = img_shape[-2:]
        N_Hs1 = [H // 1, H // 2, H // 4, H // 8]
        N_Ws1 = [W // 1, W // 2, W // 4, W // 8]
        cnn_feats = [rearrange(cnn_feats[i], 'b (nh nw) c -> b nh nw c', nh = N_Hs1[i], nw=N_Ws1[i]) for i in range(len(N_Hs1))]
        feat1, feat2, feat4, feat8 = cnn_feats
        feat16 = torch.cat([enc_output1, dec_output1], dim=-1)
        # feat16 = decout[-1]
        B, S, D = feat16.shape
        feat16 = feat16.view(B, H // self.patch_size, W // self.patch_size, D)
        out['feat1'] = feat1
        out['feat2'] = feat2
        out['feat4'] = feat4
        out['feat8'] = feat8
        out['feat16'] = feat16
        return out

class PixelwiseTaskWithDPT_catwarp(PixelwiseTaskWithDPT):
    """ cat cnn features to DPT head that outputs 3d points
    The input for both heads is a concatenation of Encoder and Decoder outputs
    """

    def __init__(self, net, hooks_idx=None, dim_tokens=None,num_channels=1, postprocess=None, feature_dim=256, last_dim=32, depth_mode=None, conf_mode=None, head_type="regression", **kwargs):
        super().__init__(num_channels=num_channels, feature_dim=feature_dim, last_dim=last_dim, hooks_idx=hooks_idx,
                         dim_tokens=dim_tokens, depth_mode=depth_mode, postprocess=postprocess, conf_mode=conf_mode, head_type=head_type)
        
        patch_size = net.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2 and isinstance(patch_size[0], int) and isinstance(
                patch_size[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
            assert patch_size[0] == patch_size[1], "Error, non square patches not managed"
            patch_size = patch_size[0]
        self.patch_size = patch_size

    def forward(self, decout, cnn_feats, img_shape):
        out = self.dpt(decout, image_size=(img_shape[0], img_shape[1]))
        if self.postprocess:
            out = self.postprocess(out, self.depth_mode, self.conf_mode)
        
        H, W = img_shape[-2:]
        N_Hs1 = [H // 1, H // 2, H // 4, H // 8]
        N_Ws1 = [W // 1, W // 2, W // 4, W // 8]
        cnn_feats = [rearrange(cnn_feats[i], 'b (nh nw) c -> b nh nw c', nh = N_Hs1[i], nw=N_Ws1[i]) for i in range(len(N_Hs1))]
        feat1, feat2, feat4, feat8 = cnn_feats
        del cnn_feats

        enc_output1, dec_output1 = decout[0], decout[-1]
        feat16 = torch.cat([enc_output1, dec_output1], dim=-1)
        del decout, enc_output1, dec_output1
        #feat16 = decout[-1]
        B, S, D = feat16.shape
        feat16 = feat16.view(B, H // self.patch_size, W // self.patch_size, D)

        out['feat1'] = feat1
        out['feat2'] = feat2
        out['feat4'] = feat4
        out['feat8'] = feat8
        out['feat16'] = feat16
        return out

class Only_Warp(nn.Module):
    def __init__(
            self, net
    ):
        super().__init__()
        patch_size = net.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2 and isinstance(patch_size[0], int) and isinstance(
                patch_size[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
            assert patch_size[0] == patch_size[1], "Error, non square patches not managed"
            patch_size = patch_size[0]
        self.patch_size = patch_size
        net.dpt = None

    def forward(self, decout, cnn_feats, img_shape):
        H, W = img_shape[-2:]
        N_Hs1 = [H // 1, H // 2, H // 4, H // 8]
        N_Ws1 = [W // 1, W // 2, W // 4, W // 8]
        cnn_feats = [rearrange(cnn_feats[i], 'b (nh nw) c -> b nh nw c', nh = N_Hs1[i], nw=N_Ws1[i]) for i in range(len(N_Hs1))]
        feat1, feat2, feat4, feat8 = cnn_feats
        del cnn_feats

        enc_output1, dec_output1 = decout[0], decout[-1]
        feat16 = torch.cat([enc_output1, dec_output1], dim=-1)
        del decout, enc_output1, dec_output1
        #feat16 = decout[-1]
        B, S, D = feat16.shape
        feat16 = feat16.view(B, H // self.patch_size, W // self.patch_size, D)
        return {
        'feat1': feat1,
        'feat2': feat2,
        'feat4': feat4,
        'feat8': feat8,
        'feat16': feat16
        }

def mast3r_head_factory(head_type, output_mode, net, has_conf=False):
    """" build a prediction head for the decoder 
    """
    if head_type == 'catmlp+dpt' and output_mode.startswith('pts3d+desc'):
        local_feat_dim = int(output_mode[10:])
        assert net.dec_depth > 9
        l2 = net.dec_depth
        feature_dim = 256
        last_dim = feature_dim // 2
        out_nchan = 3
        ed = net.enc_embed_dim
        dd = net.dec_embed_dim
        return Cat_MLP_LocalFeatures_DPT_Pts3d(net, local_feat_dim=local_feat_dim, has_conf=has_conf,
                                               num_channels=out_nchan + has_conf,
                                               feature_dim=feature_dim,
                                               last_dim=last_dim,
                                               hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
                                               dim_tokens=[ed, dd, dd, dd],
                                               postprocess=postprocess,
                                               depth_mode=net.depth_mode,
                                               conf_mode=net.conf_mode,
                                               head_type='regression')
    elif head_type =='warp+dpt_cnn' and output_mode.startswith('pts3d'):
        assert net.dec_depth > 9
        l2 = net.dec_depth
        feature_dim = 256
        last_dim = feature_dim // 2
        out_nchan = 3
        ed = net.enc_embed_dim
        dd = net.dec_embed_dim
        return  PixelwiseTaskWithDPT_cat_cnn(net, num_channels=out_nchan + has_conf,
                                            feature_dim=feature_dim,
                                            last_dim=last_dim,
                                            hooks_idx=[l2 * 3 // 4, l2],
                                            dim_tokens=dd,
                                            postprocess=postprocess,
                                            depth_mode=net.depth_mode,
                                            conf_mode=net.conf_mode,
                                            head_type='regression',
                                            layer_dims = [256, 512, 384, 768])
    elif head_type =='warp+dpt' and output_mode.startswith('pts3d'):
        assert net.dec_depth > 9
        l2 = net.dec_depth
        feature_dim = 256
        last_dim = feature_dim // 2
        out_nchan = 3
        ed = net.enc_embed_dim
        dd = net.dec_embed_dim
        return  PixelwiseTaskWithDPT_catwarp(net, 
                                            num_channels=out_nchan + has_conf,
                                               feature_dim=feature_dim,
                                               last_dim=last_dim,
                                               hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
                                               dim_tokens=[ed, dd, dd, dd],
                                               postprocess=postprocess,
                                               depth_mode=net.depth_mode,
                                               conf_mode=net.conf_mode,
                                               head_type='regression')
    elif head_type == 'only_warp':
        return Only_Warp(net)

    elif head_type == 'warp':       
        gp_dim = 512
        feat_dim = 512
        decoder_dim = gp_dim + feat_dim
        cls_to_coord_res = 64
        coordinate_decoder = TransformerDecoder(
        nn.Sequential(*[Block(decoder_dim, 8, attn_class=MemEffAttention) for _ in range(5)]), 
        decoder_dim, 
        cls_to_coord_res**2 + 1,
        is_classifier=True,
        amp = True,
        pos_enc = False,)

        dw = True
        hidden_blocks = 8
        kernel_size = 5
        displacement_emb = "linear"
        disable_local_corr_grad = True

        conv_refiner = nn.ModuleDict(
        {
            "16": ConvRefiner(
                2 * 512+128+(2*7+1)**2,
                2 * 512+128+(2*7+1)**2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=128,
                local_corr_radius = 7,
                corr_in_other = True,
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
            "8": ConvRefiner(
                2 * 512+64+(2*3+1)**2,
                2 * 512+64+(2*3+1)**2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=64,
                local_corr_radius = 3,
                corr_in_other = True,
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
            "4": ConvRefiner(
                2 * 256+32+(2*2+1)**2,
                2 * 256+32+(2*2+1)**2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=32,
                local_corr_radius = 2,
                corr_in_other = True,
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
            "2": ConvRefiner(
                2 * 64+16,
                128+16,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=16,
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
            "1": ConvRefiner(
                2 * 9 + 6,
                24,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks = hidden_blocks,
                displacement_emb = displacement_emb,
                displacement_emb_dim = 6,
                amp = True,
                disable_local_corr_grad = disable_local_corr_grad,
                bn_momentum = 0.01,
            ),
        }
        )
        kernel_temperature = 0.2
        learn_temperature = False
        no_cov = True
        kernel = CosKernel
        only_attention = False
        basis = "fourier"
        gp16 = GP(
            kernel,
            T=kernel_temperature,
            learn_temperature=learn_temperature,
            only_attention=only_attention,
            gp_dim=gp_dim,
            basis=basis,
            no_cov=no_cov,
        )
        gps = nn.ModuleDict({"16": gp16})
        proj16 = nn.Sequential(nn.Conv2d(1024+768, 512, 1, 1), nn.BatchNorm2d(512))
        proj8 = nn.Sequential(nn.Conv2d(512, 512, 1, 1), nn.BatchNorm2d(512))
        proj4 = nn.Sequential(nn.Conv2d(256, 256, 1, 1), nn.BatchNorm2d(256))
        proj2 = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.BatchNorm2d(64))
        proj1 = nn.Sequential(nn.Conv2d(64, 9, 1, 1), nn.BatchNorm2d(9))
        proj = nn.ModuleDict({
            "16": proj16,
            "8": proj8,
            "4": proj4,
            "2": proj2,
            "1": proj1,
            })
        displacement_dropout_p = 0.0
        gm_warp_dropout_p = 0.0
        decoder = Decoder(coordinate_decoder, 
                        gps,
                        proj, 
                        conv_refiner, 
                        detach=True, 
                        scales=["16", "8", "4", "2", "1"], 
                        displacement_dropout_p = displacement_dropout_p,
                        gm_warp_dropout_p = gm_warp_dropout_p)
        return WarpHead(decoder)
    else:
        raise NotImplementedError(
            f"unexpected {head_type=} and {output_mode=}")
