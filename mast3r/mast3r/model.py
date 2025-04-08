# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R model class
# --------------------------------------------------------
import torch
import torch.nn.functional as F
import os
import torch.nn as nn
from mast3r.catmlp_dpt_head import mast3r_head_factory
import torchvision.models as tvm

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.model import AsymmetricCroCo3DStereo, AsymmetricCroCo3DStereo_DINOv2, AsymmetricCroCo3DStereo_cnn # noqa
from dust3r.utils.misc import transpose_to_landscape  # noqa


inf = float('inf')


def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class AsymmetricMASt3R(AsymmetricCroCo3DStereo):
    def __init__(self, desc_mode=('norm'), two_confs=False, desc_conf_mode=None, **kwargs):
        self.desc_mode = desc_mode
        self.two_confs = two_confs
        self.desc_conf_mode = desc_conf_mode
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricMASt3R, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size, **kw):
        assert img_size[0] % patch_size == 0 and img_size[
            1] % patch_size == 0, f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        if self.desc_conf_mode is None:
            self.desc_conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

class AsymmetricMASt3R_DINOv2(AsymmetricCroCo3DStereo_DINOv2):
    def __init__(self, desc_mode=('norm'), two_confs=False, desc_conf_mode=None, **kwargs):
        self.desc_mode = desc_mode
        self.two_confs = two_confs
        self.desc_conf_mode = desc_conf_mode
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricMASt3R_DINOv2, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size, **kw):
        assert img_size[0] % patch_size == 0 and img_size[
            1] % patch_size == 0, f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        if self.desc_conf_mode is None:
            self.desc_conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

class AsymmetricMASt3R_cnn(AsymmetricCroCo3DStereo_cnn):
    def __init__(self, desc_mode=('norm'), two_confs=False, desc_conf_mode=None, **kwargs):
        self.desc_mode = desc_mode
        self.two_confs = two_confs
        self.desc_conf_mode = desc_conf_mode
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricMASt3R_DINOv2, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size, **kw):
        assert img_size[0] % patch_size == 0 and img_size[
            1] % patch_size == 0, f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        if self.desc_conf_mode is None:
            self.desc_conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)




def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


class Mast3rAndRefiner(nn.Module):
    def __init__(self, mast3r_model_name ="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
     cnn_kwargs = None, 
     amp = False, 
     use_vgg = False, 
    amp_dtype = torch.float16, 
    scales = ['16', '8', '4', '2', '1'],
    local_refiner = None):
        super(Mast3rAndRefiner, self).__init__()
        base_model = AsymmetricMASt3R.from_pretrained(mast3r_model_name)
        for param in base_model.parameters():
            param.requires_grad = False 
        self.base_model = base_model 

        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        if not use_vgg:
            self.cnn = ResNet50(**cnn_kwargs)
        else:
            self.cnn = VGG19(**cnn_kwargs)
        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.amp:
            model = model.to("cuda", self.amp_dtype)
        self.scales = scales 
        self.conv_refiner1 = nn.ModuleDict({
                "16": ConvRefiner(
                    1024+768,
                    512,
                    2 * 512+128,
                    2 * 512+128,
                    4,
                    displacement_emb_dim=128,
                    scale = 16,
                    amp = True,
                    bn_momentum = 0.01,
                ),
                "8": ConvRefiner(
                    512,
                    512,
                    2 * 512+64,
                    2 * 512+64,
                    4,
                    displacement_emb_dim=64,
                    scale = 8,
                    amp = True,
                    bn_momentum = 0.01,
                ),
                "4": ConvRefiner(
                    256,
                    256,
                    2 * 256 + 32,
                    2 * 256 + 32,
                    4,
                    displacement_emb_dim=32,
                    scale = 4,
                    amp = True,
                    bn_momentum = 0.01,
                ),
                "2": ConvRefiner(
                    128,
                    64,
                    2 * 64 + 16,
                    2 * 64 + 16,
                    4,
                    displacement_emb_dim=16,
                    scale = 2,
                    amp = True,
                    bn_momentum = 0.01,
                ),
                "1": ConvRefiner(
                    64,
                    9,
                    2 * 9 + 6,
                    2 * 9 + 6,
                    4,
                    displacement_emb_dim = 6,
                    scale = 1,
                    amp = True,
                    bn_momentum = 0.01,
                ),}
        )
        self.conv_refiner2 = nn.ModuleDict(
            {
                "16": ConvRefiner(
                    2 * 512+128,
                    2 * 512+128,
                    4,
                    displacement_emb_dim=128,
                    scale = 16,
                    amp = True,
                    bn_momentum = 0.01,
                ),
                "8": ConvRefiner(
                    2 * 512+64,
                    2 * 512+64,
                    4,
                    displacement_emb_dim=64,
                    scale = 8,
                    amp = True,
                    bn_momentum = 0.01,
                ),
                "4": ConvRefiner(
                    2 * 256 + 32,
                    2 * 256 + 32,
                    4,
                    displacement_emb_dim=32,
                    scale = 4,
                    amp = True,
                    bn_momentum = 0.01,
                ),
                "2": ConvRefiner(
                    2 * 64 + 16,
                    2 * 64 + 16,
                    4,
                    displacement_emb_dim=16,
                    scale = 2,
                    amp = True,
                    bn_momentum = 0.01,
                ),
                "1": ConvRefiner(
                    2 * 9 + 6,
                    2 * 9 + 6,
                    4,
                    displacement_emb_dim = 6,
                    scale = 1,
                    amp = True,
                    bn_momentum = 0.01,
                ),
            }
        )
        
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
        self.proj = proj
        self.mlp1 = local_refiner
        self.mlp2 = local_refiner

    def forward(self, view1, view2):
        results1 = {}
        results2 = {}
        all_scales = self.scales
        coarse_scales = int(all_scales[0])
        img1, img2 = view1['img'], view2['img']
        f1 = self.cnn(img1)
        f2 = self.cnn(img2)
        
        for new_scale in all_scales:
            ins = int(new_scale)
            results1[ins] = {}
            if ins in coarse_scales:
                res1, res2, pts3d1, pts3d2, feat1, feat2 = self.base_model.refine_forward(view1, view2)
                f1[ins] = feat1
                f2[ins] = feat2
                results1[ins].update(res1)
                results2[ins].update(res2)

            f1_s, f2_s = f1[ins], f2[ins]
            
            autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(f1_s.device, str(f1_s)=='cuda', self.amp_dtype)
            with torch.autocast(autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
                if not autocast_enabled:
                    f1_s, f2_s = f1_s.to(torch.float32), f2_s.to(torch.float32)
            
            if new_scale in self.mlp1:
                if ins in coarse_scales:
                    local_features1 = self.mlp1[new_scale](f1_s)
                    local_features2 = self.mlp2[new_scale](f2_s)
                else:
                    local_features1 = self.mlp1[new_scale](f1_s, local_features1)
                    local_features2 = self.mlp2[new_scale](f2_s, local_features2)

            if new_scale in self.conv_refiner1:
                results1[ins].update({
                    "pts_pre_delta": pts3d1
                })

                delta_pts3d1 = self.conv_refiner1[new_scale](
                    f1_s, f2_s, pts3d1
                )
                pts3d1 = pts3d1 + delta_pts3d1

                out1 = torch.cat([pts3d1, local_features1], dim=1)
                out1 = self.base_model.downstream_head1.postprocess(out1,
                                    depth_mode=self.base_model.downstream_head1.depth_mode,
                                    conf_mode=self.base_model.downstream_head1.conf_mode,
                                    desc_dim=self.base_model.downstream_head1.local_feat_dim,
                                    desc_mode=self.base_model.downstream_head1.desc_mode,
                                    two_confs=self.base_model.downstream_head1.two_confs,
                                    desc_conf_mode=self.base_model.downstream_head1.desc_conf_mode)

                results1[ins].update(out1)
                
                results2[ins].update({
                    "pts_pre_delta": pts3d2
                })

                delta_pts3d2 = self.conv_refiner2[new_scale](
                    f1_s, f2_s, pts3d1
                )
                pts3d2 = pts3d2 + delta_pts3d2
                local_features2 = self.mlp2(feat2)
                out2 = torch.cat([pts3d2, local_features2], dim=1)
                out2 = self.base_model.downstream_head2.postprocess(out2,
                                    depth_mode=self.base_model.downstream_head2.depth_mode,
                                    conf_mode=self.base_model.downstream_head2.conf_mode,
                                    desc_dim=self.base_model.downstream_head2.local_feat_dim,
                                    desc_mode=self.base_model.downstream_head2.desc_mode,
                                    two_confs=self.base_model.downstream_head2.two_confs,
                                    desc_conf_mode=self.base_model.downstream_head2.desc_conf_mode)
                results2[ins].update(out2)

        finest_scale = 1
        return results1[finest_scale], results2[finest_scale] 



class ResNet50(nn.Module):
    def __init__(self, pretrained=False, high_res = False, weights = None, 
                 dilation = None, freeze_bn = True, anti_aliased = False, early_exit = False, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        if dilation is None:
            dilation = [False,False,False]
        if anti_aliased:
            pass
        else:
            if weights is not None:
                self.net = tvm.resnet50(weights = weights,replace_stride_with_dilation=dilation)
            else:
                self.net = tvm.resnet50(pretrained=pretrained,replace_stride_with_dilation=dilation)
            
        self.high_res = high_res
        self.freeze_bn = freeze_bn
        self.early_exit = early_exit
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        with torch.autocast(autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
            net = self.net
            feats = {1:x}
            x = net.conv1(x)
            x = net.bn1(x)
            x = net.relu(x)
            feats[2] = x 
            x = net.maxpool(x)
            x = net.layer1(x)
            feats[4] = x 
            x = net.layer2(x)
            feats[8] = x
            if self.early_exit:
                return feats
            x = net.layer3(x)
            feats[16] = x
            x = net.layer4(x)
            feats[32] = x
            return feats

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                pass

class VGG19(nn.Module): #scale 8,4,2,1
    def __init__(self, pretrained=False, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])#40
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        with torch.autocast(device_type=autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
            feats = []
            scale = 1
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale](x)
                    scale = scale*2
                x = layer(x)
            return feats

def get_autocast_params(device=None, enabled=False, dtype=None):
    if device is None:
        autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        #strip :X from device
        autocast_device = str(device).split(":")[0]
    if 'cuda' in str(device):
        out_dtype = dtype
        enabled = True
    else:
        out_dtype = torch.bfloat16
        enabled = False
        # mps is not supported
        autocast_device = "cpu"
    return autocast_device, enabled, out_dtype

class ConvRefiner(nn.Module):
    def __init__(
        self,
        proj_in_dim = 1024,
        proj_out_dim = 512,
        in_dim=6,
        hidden_dim=16,
        out_dim=4,
        dw=True,
        kernel_size=5,
        hidden_blocks=8,
        displacement_emb_dim = 128,
        amp = False,
        use_bias_block_1 = True,
        scale = 8,
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
        self.disp_emb = nn.Conv2d(3, displacement_emb_dim, scale, scale)
        self.scale = scale
        self.amp = amp
        self.sample_mode = sample_mode
        self.amp_dtype = amp_dtype
        self.proj = nn.ModuleList([nn.Conv2d(proj_in_dim, proj_out_dim, 1, 1), nn.BatchNorm2d(proj_out_dim)])
        
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
        
    def forward(self, x, y, pts):
        b,c,hs,ws = x.shape
        emb_pts = self.disp_emb(pts[:,:3])
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, enabled=self.amp, dtype=self.amp_dtype)
        with torch.autocast(autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):            
            d = torch.cat((x, y, emb_pts), dim=1)
            d = self.block1(d)
            d = self.hidden_blocks(d)
        out = self.out_conv(d.float())
        return out