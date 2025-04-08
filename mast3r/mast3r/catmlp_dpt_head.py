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
from romatch.models.matcher import *
from romatch.models.transformer import Block, TransformerDecoder, MemEffAttention
import torch.nn as nn

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

class Cat_Warp_LocalFeatures_DPT_Pts3d_resnet(PixelwiseTaskWithDPT_resnet):
    """ Mixture between MLP and DPT head that outputs 3d points and local features (with MLP).
    The input for both heads is a concatenation of Encoder and Decoder outputs
    """

    def __init__(self, net, has_conf=False, local_feat_dim=16, hidden_dim_factor=4., hooks_idx=None, dim_tokens=None,
                 num_channels=1, postprocess=None, feature_dim=256, last_dim=32, depth_mode=None, conf_mode=None, head_type="regression", layer_dims = [256, 512, 384, 768], **kwargs):
        super().__init__(num_channels=num_channels, feature_dim=feature_dim, last_dim=last_dim, hooks_idx=hooks_idx,
                         dim_tokens=dim_tokens, depth_mode=depth_mode, postprocess=postprocess, conf_mode=conf_mode, head_type=head_type, layer_dims = layer_dims)
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
        proj16 = nn.Sequential(nn.Conv2d(768, 512, 1, 1), nn.BatchNorm2d(512))
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
                        scales=["16", "8"],#, "4", "2", "1"], 
                        displacement_dropout_p = displacement_dropout_p,
                        gm_warp_dropout_p = gm_warp_dropout_p)
        
        

        self.head_local_features = Mlp(in_features=idim,
                                       hidden_features=int(hidden_dim_factor * idim),
                                       out_features=(self.local_feat_dim + self.two_confs) * self.patch_size**2)

    def forward(self, decout, vgg_features, img_shape, mode = 'default'):
        pts3d = self.dpt(decout, vgg_features, image_size=(img_shape[0], img_shape[1]))
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
    elif head_type =='catmlp+dpt_resnet' and output_mode.startwith('pts3d+desc'):
        local_feat_dim = int(output_mode[10:])
        assert net.dec_depth > 9
        l2 = net.dec_depth
        feature_dim = 256
        last_dim = feature_dim // 2
        out_nchan = 3
        ed = net.enc_embed_dim
        dd = net.dec_embed_dim
        return Cat_MLP_LocalFeatures_DPT_Pts3d_resnet(net, local_feat_dim=local_feat_dim, has_conf=has_conf,
                                               num_channels=out_nchan + has_conf,
                                               feature_dim=feature_dim,
                                               last_dim=last_dim,
                                               hooks_idx=[l2 * 3 // 4, l2],
                                               dim_tokens=dd,
                                               postprocess=postprocess,
                                               depth_mode=net.depth_mode,
                                               conf_mode=net.conf_mode,
                                               head_type='regression',
                                               layer_dims = [256, 512, 384, 768])
    else:
        raise NotImplementedError(
            f"unexpected {head_type=} and {output_mode=}")
