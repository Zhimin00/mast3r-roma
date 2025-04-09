from typing import Optional, Union
import torch
from torch import device
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import gc
from romatch.utils.utils import get_autocast_params
import pdb
from diffusers import StableDiffusionPipeline
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers import DDIMScheduler
import numpy as np
import os 
import sys

mast3r_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../mast3r"))
sys.path.insert(0, mast3r_path)

class FeatureExtractor_woProj(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True, 
                 amp=True, amp_dtype=torch.float16, 
                 sd_id='stabilityai/stable-diffusion-2-1', t=261,
                 up_ft_index=1,
                 ensemble_size=1,
                 alpha = 0.5):
        super(FeatureExtractor_woProj, self).__init__()
        
        # Set model parameters based on the chosen backbone
        self.model_name = model_name
        self.amp = amp
        self.amp_dtype = amp_dtype
        

        if model_name == 'resnet50':
            self.backbone = tvm.resnet50(pretrained=pretrained, replace_stride_with_dilation=[False, False, False])
            feature_dim = self.backbone.layer3[-1].conv3.out_channels  # Output dim of ResNet at layer3 (scale 16)
        elif model_name == 'vgg19':
            self.backbone = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:53])  # Up to scale 16
            feature_dim = 512  # Output dim of VGG19 after scale 16
        elif model_name == 'googlenet':
            self.backbone = tvm.googlenet(pretrained=pretrained)
            feature_dim = 832
        elif model_name == 'dinov2':
            dinov2_weights = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", map_location="cpu")
            from .transformer import vit_large
            vit_kwargs = dict(img_size= 518,
                patch_size= 14,
                init_values = 1.0,
                ffn_layer = "mlp",
                block_chunks = 0,
            )
            dinov2_vitl14 = vit_large(**vit_kwargs).eval()
            dinov2_vitl14.load_state_dict(dinov2_weights)
            self.backbone = dinov2_vitl14
            feature_dim = 1024
        elif model_name == 'diffusion':
            unet = MyUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")
            onestep_pipe = OneStepSDPipeline.from_pretrained(sd_id, unet=unet, safety_checker=None)
            onestep_pipe.vae.decoder = None
            onestep_pipe.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")
            gc.collect()
            onestep_pipe.enable_attention_slicing()
            onestep_pipe.enable_xformers_memory_efficient_attention()
            null_prompt = ''
            null_prompt_embeds = onestep_pipe._encode_prompt(
                prompt=null_prompt,
                device='cpu',
                num_images_per_prompt=1,
                do_classifier_free_guidance=False) # [1, 77, dim]

            self.null_prompt_embeds = null_prompt_embeds.to('cuda', dtype=amp_dtype)
            self.null_prompt = null_prompt
            onestep_pipe.text_encoder = onestep_pipe.text_encoder.to("cpu")
            onestep_pipe.tokenizer = None  # Free tokenizer if not needed
            self.t = t
            self.up_ft_index = up_ft_index
            self.ensemble_size =ensemble_size

            if self.amp:
                onestep_pipe = onestep_pipe.to("cuda", self.amp_dtype)
            self.backbone = onestep_pipe 
            feature_dim = 1280
        elif model_name == 'amradio':
            model_version="radio_v2.5-b"
            #model_version="e-radio_v2"
            model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_version, progress=True, skip_validation=True)
            self.backbone = model
            feature_dim = 768
        elif model_name == 'amradio_g':
            model_version="radio_v2.5-g"
            #model_version="e-radio_v2"
            model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_version, progress=True, skip_validation=True)
            self.backbone = model
            feature_dim = 1536
        elif model_name == 'dino_SD':
            dinov2_weights = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", map_location="cpu")
            from .transformer import vit_large
            vit_kwargs = dict(img_size= 518,
                patch_size= 14,
                init_values = 1.0,
                ffn_layer = "mlp",
                block_chunks = 0,
            )
            dinov2_vitl14 = vit_large(**vit_kwargs).eval()
            dinov2_vitl14.load_state_dict(dinov2_weights)
            self.backbone1 = dinov2_vitl14
            unet = MyUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")
            onestep_pipe = OneStepSDPipeline.from_pretrained(sd_id, unet=unet, safety_checker=None)
            onestep_pipe.vae.decoder = None
            onestep_pipe.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")
            gc.collect()
            onestep_pipe.enable_attention_slicing()
            onestep_pipe.enable_xformers_memory_efficient_attention()
            onestep_pipe.text_encoder = onestep_pipe.text_encoder.to("cuda")
           

            null_prompt = ''
            null_prompt_embeds = onestep_pipe._encode_prompt(
                prompt=null_prompt,
                device='cuda',
                num_images_per_prompt=1,
                do_classifier_free_guidance=False) # [1, 77, dim]

            self.null_prompt_embeds = null_prompt_embeds.to('cuda', dtype=amp_dtype)
            self.null_prompt = null_prompt
            onestep_pipe.tokenizer = None  # Free tokenizer if not needed

            self.t = t
            self.up_ft_index = up_ft_index
            self.ensemble_size =ensemble_size
            self.alpha = alpha

            if self.amp:
                onestep_pipe = onestep_pipe.to("cuda", self.amp_dtype)
            self.backbone2 = onestep_pipe 
            feature_dim = 1024 + 1280
        
        elif model_name == 'mast3r':
            from mast3r.model import AsymmetricMASt3R
            mast3r_model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
                # you can put the path to a local checkpoint in model_name if needed
            model = AsymmetricMASt3R.from_pretrained(mast3r_model_name)
            self.backbone = model
            feature_dim = 1024
            #backbone = AsymmetricMASt3R.from_pretrained(ckpt_args.pretrained)
        elif model_name == 'dust3r':
            from dust3r.model import AsymmetricCroCo3DStereo
            dust3r_model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
            model = AsymmetricCroCo3DStereo.from_pretrained(dust3r_model_name)
            self.backbone = model
            feature_dim = 1024

        elif model_name == 'mast3r_decoder':
            from mast3r.model import AsymmetricMASt3R
            mast3r_model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
                # you can put the path to a local checkpoint in model_name if needed
            model = AsymmetricMASt3R.from_pretrained(mast3r_model_name)
            self.backbone = model
            feature_dim = 1024 + 768
        elif model_name == 'mast3r_decoder_mlp_without_conf':
            from mast3r.model import AsymmetricMASt3R
            mast3r_model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
                # you can put the path to a local checkpoint in model_name if needed
            model = AsymmetricMASt3R.from_pretrained(mast3r_model_name)
            self.backbone = model
            feature_dim = 24*16*16
        elif model_name == 'mast3r_decoder_mlp_with_conf':
            from mast3r.model import AsymmetricMASt3R
            mast3r_model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
                # you can put the path to a local checkpoint in model_name if needed
            model = AsymmetricMASt3R.from_pretrained(mast3r_model_name)
            self.backbone = model
            feature_dim = 25*16*16
        elif model_name == 'dust3r_decoder':
            from dust3r.model import AsymmetricCroCo3DStereo
            dust3r_model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
            model = AsymmetricCroCo3DStereo.from_pretrained(dust3r_model_name)
            self.backbone = model
            feature_dim = 1024 + 768
        
        self.feature_dim = feature_dim

        if model_name == 'dino_SD':
            for param in self.backbone2.unet.parameters():
                param.requires_grad = False
            if hasattr(self.backbone2, 'vae') and self.backbone2.vae is not None:
                for param in self.backbone2.vae.parameters():
                    param.requires_grad = False
            for param in self.backbone1.parameters():
                param.requires_grad = False

        elif model_name == 'diffusion':
            for param in self.backbone.unet.parameters():
                param.requires_grad = False
            if hasattr(self.backbone, 'vae') and self.backbone.vae is not None:
                for param in self.backbone.vae.parameters():
                    param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x, upsample=False):
        if 'decoder' not in self.model_name:
            B,C,H,W = x.shape
        else:
            img1, img2 = x
            x = img1
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        if self.model_name == 'diffusion':
            self.backbone.unet.eval()
            if hasattr(self.backbone, 'vae') and self.backbone.vae is not None:
                self.backbone.vae.eval()
        elif self.model_name == 'dino_SD':
            self.backbone2.unet.eval()
            if hasattr(self.backbone2, 'vae') and self.backbone2.vae is not None:
                self.backbone2.vae.eval()
            self.backbone1.eval()
        else:
            self.backbone.eval()
        with torch.autocast(device_type=autocast_device, enabled=autocast_enabled, dtype=self.amp_dtype):
            if self.model_name == 'resnet50': 
                net = self.backbone
                x = net.conv1(x)
                x = net.bn1(x)
                x = net.relu(x)
                x = net.maxpool(x)
                x = net.layer1(x)
                x = net.layer2(x)
                x = net.layer3(x)
                feat = x
            elif self.model_name == 'vgg19':
                net = self.backbone
                scale = 1
                for layer in net:
                    if isinstance(layer, nn.MaxPool2d):
                        if scale == 16:
                            feat = x
                        scale = scale*2
                    x = layer(x)
            elif self.model_name == 'googlenet':
                net = self.backbone
                x = net.conv1(x)
                x = net.maxpool1(x) # N x 64 x 56 x 56
                x = net.conv2(x) # N x 64 x 56 x 56
                x = net.conv3(x) # N x 192 x 56 x 56
                x = net.maxpool2(x) # N x 192 x 28 x 28
                x = net.inception3a(x)# N x 256 x 28 x 28
                x = net.inception3b(x)# N x 480 x 28 x 28
                x = net.maxpool3(x) # N x 480 x 14 x 14
                x = net.inception4a(x) # N x 512 x 14 x 14
                x = net.inception4b(x)# N x 512 x 14 x 14
                x = net.inception4c(x)# N x 512 x 14 x 14
                x = net.inception4d(x) # N x 528 x 14 x 14
                x = net.inception4e(x) # N x 832 x 14 x 14
                feat = x
            elif self.model_name == 'dinov2':
                dinov2_features_16 = self.backbone.forward_features(x.to(self.amp_dtype))
                feat = dinov2_features_16['x_norm_patchtokens'].permute(0,2,1).reshape(B,1024,H//14, W//14)
            elif self.model_name == 'diffusion':
                img_tensor = x.cuda()
                prompt_embeds = self.null_prompt_embeds.repeat(B, 1, 1).to(self.amp_dtype)
                if self.backbone.device != x.device:
                    self.backbone = self.backbone.to(x.device, self.amp_dtype)
                sd_features = self.backbone(img_tensor.to(self.amp_dtype),
                                            t=self.t, 
                                            up_ft_indices=[self.up_ft_index,2],
                                            prompt_embeds=prompt_embeds)
                feat = sd_features['up_ft'][self.up_ft_index]
                del sd_features

            elif self.model_name == 'amradio' or self.model_name == 'amradio_g':
                #self.backbone.model.set_optimal_window_size(x.shape[2:])
                summary, feat = self.backbone(x, feature_fmt='NCHW')
            elif self.model_name == 'dino_SD':
                dinov2_features_16 = self.backbone1.forward_features(x.to(self.amp_dtype))
                dinov2_feat = dinov2_features_16['x_norm_patchtokens'].permute(0,2,1).reshape(B,1024,H//14, W//14)
                
                img_tensor =  x.cuda()
                prompt_embeds = self.null_prompt_embeds.repeat(B, 1, 1).to(self.amp_dtype)
                if self.backbone2.device != x.device:
                    self.backbone2 = self.backbone2.to(x.device, self.amp_dtype)
                sd_features = self.backbone2(img_tensor.to(self.amp_dtype),
                                            t=self.t, 
                                            up_ft_indices=[self.up_ft_index,2],
                                            prompt_embeds=prompt_embeds)
                SD_feat = sd_features['up_ft'][self.up_ft_index]
                SD_feat_resized = F.interpolate(SD_feat, size=(H//14, W//14), mode='bilinear', align_corners=False)
                SD_feat_norm = SD_feat_resized / SD_feat_resized.norm(dim=1, keepdim=True)
                dinov2_feat_norm = dinov2_feat / dinov2_feat.norm(dim=1, keepdim=True)
                feat = torch.cat([self.alpha * dinov2_feat_norm, self.alpha * SD_feat_norm], dim=1)
            elif self.model_name == 'mast3r':
                feat = self.backbone._encode_image(x, true_shape = [H,W])[0]
                B, _, C = feat.shape
                H_s, W_s = H//16, W//16
                feat = feat.transpose(1, 2).reshape(B, C, H_s, W_s)
            elif self.model_name == 'dust3r':
                feat = self.backbone._encode_image(x, true_shape = [H,W])[0]
                B, _, C = feat.shape
                H_s, W_s = H//16, W//16
                feat = feat.transpose(1, 2).reshape(B, C, H_s, W_s)
            elif self.model_name == 'mast3r_decoder':
                feat1, feat2 = self.backbone.decoder_forward(img1, img2)
                feat = (feat1, feat2)
            elif self.model_name == 'mast3r_decoder_mlp_without_conf':
                feat1, feat2 = self.backbone.mlp_forward(img1, img2, mode='mlp_without_conf')
                feat = (feat1, feat2)
            elif self.model_name == 'mast3r_decoder_mlp_with_conf':
                feat1, feat2 = self.backbone.mlp_forward(img1, img2, mode='mlp_with_conf')
                feat = (feat1, feat2)
            elif self.model_name == 'dust3r_decoder':
                feat1, feat2 = self.backbone.decoder_forward(img1, img2)
                feat = (feat1, feat2)
        return feat

class FeatureExtractor_s16(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True, 
                 amp=True, amp_dtype=torch.float16, 
                 sd_id='stabilityai/stable-diffusion-2-1', t=261,
                 up_ft_index=1,
                 ensemble_size=1,
                 alpha = 0.5):
        super(FeatureExtractor_s16, self).__init__()
        
        # Set model parameters based on the chosen backbone
        self.model_name = model_name
        self.amp = amp
        self.amp_dtype = amp_dtype
        

        if model_name == 'resnet50':
            self.backbone = tvm.resnet50(pretrained=pretrained, replace_stride_with_dilation=[False, False, False])
            feature_dim = self.backbone.layer3[-1].conv3.out_channels  # Output dim of ResNet at layer3 (scale 16)
        elif model_name == 'vgg19':
            self.backbone = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:53])  # Up to scale 16
            feature_dim = 512  # Output dim of VGG19 after scale 16
        elif model_name == 'googlenet':
            self.backbone = tvm.googlenet(pretrained=pretrained)
            feature_dim = 832
        elif model_name == 'dinov2':
            dinov2_weights = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", map_location="cpu")
            from .transformer import vit_large
            vit_kwargs = dict(img_size= 518,
                patch_size= 14,
                init_values = 1.0,
                ffn_layer = "mlp",
                block_chunks = 0,
            )
            dinov2_vitl14 = vit_large(**vit_kwargs).eval()
            dinov2_vitl14.load_state_dict(dinov2_weights)
            self.backbone = dinov2_vitl14
            feature_dim = 1024
        elif model_name == 'diffusion':
            unet = MyUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")
            onestep_pipe = OneStepSDPipeline.from_pretrained(sd_id, unet=unet, safety_checker=None)
            onestep_pipe.vae.decoder = None
            onestep_pipe.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")
            gc.collect()
            onestep_pipe.enable_attention_slicing()
            onestep_pipe.enable_xformers_memory_efficient_attention()
            null_prompt = ''
            null_prompt_embeds = onestep_pipe._encode_prompt(
                prompt=null_prompt,
                device='cpu',
                num_images_per_prompt=1,
                do_classifier_free_guidance=False) # [1, 77, dim]

            self.null_prompt_embeds = null_prompt_embeds.to('cuda', dtype=amp_dtype)
            self.null_prompt = null_prompt
            onestep_pipe.text_encoder = onestep_pipe.text_encoder.to("cpu")
            onestep_pipe.tokenizer = None  # Free tokenizer if not needed
            self.t = t
            self.up_ft_index = up_ft_index
            self.ensemble_size =ensemble_size

            if self.amp:
                onestep_pipe = onestep_pipe.to("cuda", self.amp_dtype)
            self.backbone = onestep_pipe 
            feature_dim = 1280
        elif model_name == 'amradio':
            model_version="radio_v2.5-b"
            #model_version="e-radio_v2"
            model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_version, progress=True, skip_validation=True)
            self.backbone = model
            feature_dim = 768
        elif model_name == 'amradio_g':
            model_version="radio_v2.5-g"
            #model_version="e-radio_v2"
            model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_version, progress=True, skip_validation=True)
            self.backbone = model
            feature_dim = 1536
        elif model_name == 'dino_SD':
            dinov2_weights = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", map_location="cpu")
            from .transformer import vit_large
            vit_kwargs = dict(img_size= 518,
                patch_size= 14,
                init_values = 1.0,
                ffn_layer = "mlp",
                block_chunks = 0,
            )
            dinov2_vitl14 = vit_large(**vit_kwargs).eval()
            dinov2_vitl14.load_state_dict(dinov2_weights)
            self.backbone1 = dinov2_vitl14
            unet = MyUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")
            onestep_pipe = OneStepSDPipeline.from_pretrained(sd_id, unet=unet, safety_checker=None)
            onestep_pipe.vae.decoder = None
            onestep_pipe.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")
            gc.collect()
            onestep_pipe.enable_attention_slicing()
            onestep_pipe.enable_xformers_memory_efficient_attention()
            onestep_pipe.text_encoder = onestep_pipe.text_encoder.to("cuda")
            null_prompt = ''
            null_prompt_embeds = onestep_pipe._encode_prompt(
                prompt=null_prompt,
                device='cuda',
                num_images_per_prompt=1,
                do_classifier_free_guidance=False) # [1, 77, dim]

            self.null_prompt_embeds = null_prompt_embeds.to('cuda', dtype=amp_dtype)
            self.null_prompt = null_prompt
            onestep_pipe.tokenizer = None  # Free tokenizer if not needed

            self.t = t
            self.up_ft_index = up_ft_index
            self.ensemble_size =ensemble_size
            self.alpha = alpha

            if self.amp:
                onestep_pipe = onestep_pipe.to("cuda", self.amp_dtype)
            self.backbone2 = onestep_pipe 
            feature_dim = 1024 + 1280
        
        elif model_name == 'mast3r':
            from mast3r.model import AsymmetricMASt3R
            mast3r_model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
                # you can put the path to a local checkpoint in model_name if needed
            model = AsymmetricMASt3R.from_pretrained(mast3r_model_name)
            self.backbone = model
            feature_dim = 1024
            #backbone = AsymmetricMASt3R.from_pretrained(ckpt_args.pretrained)
        elif model_name == 'dust3r':
            from dust3r.model import AsymmetricCroCo3DStereo
            dust3r_model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
            model = AsymmetricCroCo3DStereo.from_pretrained(dust3r_model_name)
            self.backbone = model
            feature_dim = 1024
        elif model_name == 'mast3r_decoder':
            from mast3r.model import AsymmetricMASt3R
            mast3r_model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
                # you can put the path to a local checkpoint in model_name if needed
            model = AsymmetricMASt3R.from_pretrained(mast3r_model_name)
            self.backbone = model
            feature_dim = 1024+768
        elif model_name == 'mast3r_decoder_mlp_without_conf':
            from mast3r.model import AsymmetricMASt3R
            mast3r_model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
                # you can put the path to a local checkpoint in model_name if needed
            model = AsymmetricMASt3R.from_pretrained(mast3r_model_name)
            self.backbone = model
            feature_dim = 24*16*16
        elif model_name == 'mast3r_decoder_mlp_with_conf':
            from mast3r.model import AsymmetricMASt3R
            mast3r_model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
                # you can put the path to a local checkpoint in model_name if needed
            model = AsymmetricMASt3R.from_pretrained(mast3r_model_name)
            self.backbone = model
            feature_dim = 25*16*16
        elif model_name == 'dust3r_decoder':
            from dust3r.model import AsymmetricCroCo3DStereo
            dust3r_model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
            model = AsymmetricCroCo3DStereo.from_pretrained(dust3r_model_name)
            self.backbone = model
            feature_dim = 1024+768
        # self.proj_layer =  nn.Sequential(nn.Conv2d(feature_dim, 512, kernel_size=1, stride=1),
        #                                 nn.BatchNorm2d(512),
        #                                 nn.ReLU(inplace=True),  # Non-linear activation
        #                                 nn.Conv2d(512, 512, kernel_size=1, stride=1),
        #                                 nn.BatchNorm2d(512),
        #                                 nn.ReLU(inplace=True)   # Non-linear activation
        #                             )
        self.proj_layer = nn.Sequential(nn.Conv2d(feature_dim, 512, 1, 1), nn.BatchNorm2d(512))

        if model_name == 'dino_SD':
            for param in self.backbone2.unet.parameters():
                param.requires_grad = False
            if hasattr(self.backbone2, 'vae') and self.backbone2.vae is not None:
                for param in self.backbone2.vae.parameters():
                    param.requires_grad = False
            for param in self.backbone1.parameters():
                param.requires_grad = False

        elif model_name == 'diffusion':
            for param in self.backbone.unet.parameters():
                param.requires_grad = False
            if hasattr(self.backbone, 'vae') and self.backbone.vae is not None:
                for param in self.backbone.vae.parameters():
                    param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False
        for param in self.proj_layer.parameters():
            param.requires_grad = True

    def forward(self, x, upsample=False):
        if 'decoder' not in self.model_name:
            B,C,H,W = x.shape
        else:
            img1, img2 = x
            x = img1
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        if self.model_name == 'diffusion':
            self.backbone.unet.eval()
            if hasattr(self.backbone, 'vae') and self.backbone.vae is not None:
                self.backbone.vae.eval()
        elif self.model_name == 'dino_SD':
            self.backbone2.unet.eval()
            if hasattr(self.backbone2, 'vae') and self.backbone2.vae is not None:
                self.backbone2.vae.eval()
            self.backbone1.eval()
        else:
            self.backbone.eval()
        with torch.autocast(device_type=autocast_device, enabled=autocast_enabled, dtype=self.amp_dtype):
            if self.model_name == 'resnet50': 
                net = self.backbone
                x = net.conv1(x)
                x = net.bn1(x)
                x = net.relu(x)
                x = net.maxpool(x)
                x = net.layer1(x)
                x = net.layer2(x)
                x = net.layer3(x)
                feat = x
            elif self.model_name == 'vgg19':
                net = self.backbone
                scale = 1
                for layer in net:
                    if isinstance(layer, nn.MaxPool2d):
                        if scale == 16:
                            feat = x
                        scale = scale*2
                    x = layer(x)
            elif self.model_name == 'googlenet':
                net = self.backbone
                x = net.conv1(x)
                x = net.maxpool1(x) # N x 64 x 56 x 56
                x = net.conv2(x) # N x 64 x 56 x 56
                x = net.conv3(x) # N x 192 x 56 x 56
                x = net.maxpool2(x) # N x 192 x 28 x 28
                x = net.inception3a(x)# N x 256 x 28 x 28
                x = net.inception3b(x)# N x 480 x 28 x 28
                x = net.maxpool3(x) # N x 480 x 14 x 14
                x = net.inception4a(x) # N x 512 x 14 x 14
                x = net.inception4b(x)# N x 512 x 14 x 14
                x = net.inception4c(x)# N x 512 x 14 x 14
                x = net.inception4d(x) # N x 528 x 14 x 14
                x = net.inception4e(x) # N x 832 x 14 x 14
                feat = x
                #feats[16] = x
                #x = net.maxpool4(x)# N x 832 x 7 x 7
                #x = net.inception5a(x) # N x 832 x 7 x 7
                #x = net.inception5b(x)# N x 1024 x 7 x 7
                #feats[32] = x
            elif self.model_name == 'dinov2':
                dinov2_features_16 = self.backbone.forward_features(x.to(self.amp_dtype))
                feat = dinov2_features_16['x_norm_patchtokens'].permute(0,2,1).reshape(B,1024,H//14, W//14)
            elif self.model_name == 'diffusion':
                img_tensor = x.cuda()
                prompt_embeds = self.null_prompt_embeds.repeat(B, 1, 1).to(self.amp_dtype)
                if self.backbone.device != x.device:
                    self.backbone = self.backbone.to(x.device, self.amp_dtype)
                sd_features = self.backbone(img_tensor.to(self.amp_dtype),
                                            t=self.t, 
                                            up_ft_indices=[self.up_ft_index,2],
                                            prompt_embeds=prompt_embeds)
                feat = sd_features['up_ft'][self.up_ft_index]
                del sd_features

            elif self.model_name == 'amradio' or self.model_name == 'amradio_g':
                #self.backbone.model.set_optimal_window_size(x.shape[2:])
                summary, feat = self.backbone(x, feature_fmt='NCHW')
            elif self.model_name == 'dino_SD':
                dinov2_features_16 = self.backbone1.forward_features(x.to(self.amp_dtype))
                dinov2_feat = dinov2_features_16['x_norm_patchtokens'].permute(0,2,1).reshape(B,1024,H//14, W//14)
                
                img_tensor =  x.cuda()
                prompt_embeds = self.null_prompt_embeds.repeat(B, 1, 1).to(self.amp_dtype)
                if self.backbone2.device != x.device:
                    self.backbone2 = self.backbone2.to(x.device, self.amp_dtype)
                sd_features = self.backbone2(img_tensor.to(self.amp_dtype),
                                            t=self.t, 
                                            up_ft_indices=[self.up_ft_index,2],
                                            prompt_embeds=prompt_embeds)
                SD_feat = sd_features['up_ft'][self.up_ft_index]
                SD_feat_resized = F.interpolate(SD_feat, size=(H//14, W//14), mode='bilinear', align_corners=False)
                SD_feat_norm = SD_feat_resized / SD_feat_resized.norm(dim=1, keepdim=True)
                dinov2_feat_norm = dinov2_feat / dinov2_feat.norm(dim=1, keepdim=True)
                feat = torch.cat([self.alpha * dinov2_feat_norm, self.alpha * SD_feat_norm], dim=1)
            elif self.model_name == 'mast3r':
                feat = self.backbone._encode_image(x, true_shape = [H,W])[0]
                B, _, C = feat.shape
                H_s, W_s = H//16, W//16
                feat = feat.transpose(1, 2).reshape(B, C, H_s, W_s)
            elif self.model_name == 'dust3r':
                feat = self.backbone._encode_image(x, true_shape = [H,W])[0]
                B, _, C = feat.shape
                H_s, W_s = H//16, W//16
                feat = feat.transpose(1, 2).reshape(B, C, H_s, W_s)
            elif self.model_name == 'mast3r_decoder':
                feat1, feat2 = self.backbone.decoder_forward(img1, img2)
            elif self.model_name == 'mast3r_decoder_mlp_without_conf':
                feat1, feat2 = self.backbone.mlp_forward(img1, img2, mode='mlp_without_conf')
            elif self.model_name == 'mast3r_decoder_mlp_with_conf':
                feat1, feat2 = self.backbone.mlp_forward(img1, img2, mode='mlp_with_conf')
                
            elif self.model_name == 'dust3r_decoder':
                feat1, feat2 = self.backbone.decoder_forward(img1, img2)
            if 'decoder' in self.model_name:
                feat1 = self.proj_layer(feat1)
                feat2 = self.proj_layer(feat2)
                feat = (feat1, feat2)
            else:
                feat = self.proj_layer(feat)
        return feat

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

class VGG19_2(nn.Module): ##only scale 4,2,1
    def __init__(self, pretrained=False, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:27])#40
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        with torch.autocast(device_type=autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
            feats = {}
            scale = 1
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale] = x
                    scale = scale*2
                x = layer(x)
            return feats

class VGG19(nn.Module): #scale 8,4,2,1
    def __init__(self, pretrained=False, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])#40
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        with torch.autocast(device_type=autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
            feats = {}
            scale = 1
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale] = x
                    scale = scale*2
                x = layer(x)
            return feats

class GoogleNet(nn.Module): #scale 8,4,2,1
    def __init__(self, pretrained=False, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        self.net = tvm.googlenet(pretrained=pretrained)
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        with torch.autocast(device_type=autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
            net = self.net
            feats = {1:x}
            x = net.conv1(x)
            feats[2] = x        # N x 64 x 112 x 112
            x = net.maxpool1(x) # N x 64 x 56 x 56
            x = net.conv2(x) # N x 64 x 56 x 56
            x = net.conv3(x) # N x 192 x 56 x 56
            feats[4] = x
            x = net.maxpool2(x) # N x 192 x 28 x 28
            x = net.inception3a(x)# N x 256 x 28 x 28
            x = net.inception3b(x)# N x 480 x 28 x 28
            feats[8] = x
            x = net.maxpool3(x) # N x 480 x 14 x 14
            x = net.inception4a(x) # N x 512 x 14 x 14
            x = net.inception4b(x)# N x 512 x 14 x 14
            x = net.inception4c(x)# N x 512 x 14 x 14
            x = net.inception4d(x) # N x 528 x 14 x 14
            x = net.inception4e(x) # N x 832 x 14 x 14
            feats[16] = x
            x = net.maxpool4(x)# N x 832 x 7 x 7
            x = net.inception5a(x) # N x 832 x 7 x 7
            x = net.inception5b(x)# N x 1024 x 7 x 7
            feats[32] = x
            return feats


class CNNandDinov2(nn.Module):
    def __init__(self, cnn_kwargs = None, amp = False, use_vgg = False, dinov2_weights = None, amp_dtype = torch.float16):
        super().__init__()
        if dinov2_weights is None:
            dinov2_weights = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", map_location="cpu")
        from .transformer import vit_large
        vit_kwargs = dict(img_size= 518,
            patch_size= 14,
            init_values = 1.0,
            ffn_layer = "mlp",
            block_chunks = 0,
        )

        dinov2_vitl14 = vit_large(**vit_kwargs).eval()
        dinov2_vitl14.load_state_dict(dinov2_weights)
        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        if not use_vgg:
            self.cnn = ResNet50(**cnn_kwargs)
        else:
            self.cnn = VGG19(**cnn_kwargs)
        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.amp:
            dinov2_vitl14 = dinov2_vitl14.to(self.amp_dtype)
        self.dinov2_vitl14 = [dinov2_vitl14] # ugly hack to not show parameters to DDP
    
    
    def train(self, mode: bool = True):
        return self.cnn.train(mode)
    
    def forward(self, x, upsample = False):
        B,C,H,W = x.shape
        feature_pyramid = self.cnn(x)
        
        if not upsample:
            with torch.no_grad():
                if self.dinov2_vitl14[0].device != x.device:
                    self.dinov2_vitl14[0] = self.dinov2_vitl14[0].to(x.device).to(self.amp_dtype)
                dinov2_features_16 = self.dinov2_vitl14[0].forward_features(x.to(self.amp_dtype))
                features_16 = dinov2_features_16['x_norm_patchtokens'].permute(0,2,1).reshape(B,1024,H//14, W//14)
                del dinov2_features_16
                feature_pyramid[16] = features_16
        return feature_pyramid




class MyUNet2DConditionModel(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None):
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            # logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        # 5. up
        up_ft = {}
        for i, upsample_block in enumerate(self.up_blocks):

            if i > np.max(up_ft_indices):
                break

            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

            if i in up_ft_indices:
                up_ft[i] = sample.detach()

        output = {}
        output['up_ft'] = up_ft
        return output

class OneStepSDPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        img_tensor,
        t,
        up_ft_indices,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None
    ):

        device = self._execution_device
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        unet_output = self.unet(latents_noisy,
                               t,
                               up_ft_indices,
                               encoder_hidden_states=prompt_embeds,
                               cross_attention_kwargs=cross_attention_kwargs)
        return unet_output

class CNNandSD(nn.Module):
    def __init__(self, cnn_kwargs = None, amp = False, use_vgg = False, 
                sd_id = 'stabilityai/stable-diffusion-2-1', 
                amp_dtype = torch.float16, t=0,
                up_ft_index=1,
                ensemble_size=1):
        super().__init__()
        unet = MyUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")
        onestep_pipe = OneStepSDPipeline.from_pretrained(sd_id, unet=unet, safety_checker=None)
        onestep_pipe.vae.decoder = None
        onestep_pipe.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")
        gc.collect()
        onestep_pipe.enable_attention_slicing()
        onestep_pipe.enable_xformers_memory_efficient_attention()
        null_prompt = ''
        null_prompt_embeds = onestep_pipe._encode_prompt(
            prompt=null_prompt,
            device='cpu',
            num_images_per_prompt=1,
            do_classifier_free_guidance=False) # [1, 77, dim]

        self.null_prompt_embeds = null_prompt_embeds.to('cuda', dtype=amp_dtype)
        self.null_prompt = null_prompt
        onestep_pipe.text_encoder = onestep_pipe.text_encoder.to("cpu")
        onestep_pipe.tokenizer = None  # Free tokenizer if not needed

        self.t = t
        self.up_ft_index = up_ft_index
        self.ensemble_size =ensemble_size

        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        if not use_vgg:
            self.cnn = ResNet50(**cnn_kwargs)
        else:
            self.cnn = VGG19(**cnn_kwargs)
        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.amp:
            onestep_pipe = onestep_pipe.to("cuda", self.amp_dtype)
        self.pipe = [onestep_pipe] # ugly hack to not show parameters to DDP
    
    def train(self, mode: bool = True):
        return self.cnn.train(mode)
    
    def forward(self, x, upsample = False):
        B,C,H,W = x.shape
        feature_pyramid = self.cnn(x)

        img_tensor = x.unsqueeze(1).repeat(1, self.ensemble_size, 1, 1, 1)
        img_tensor = img_tensor.view(B * self.ensemble_size, C, H, W).cuda()
        if not upsample:
            with torch.no_grad():
                prompt_embeds = self.null_prompt_embeds.repeat(B*self.ensemble_size, 1, 1).to(self.amp_dtype)
                if self.pipe[0].device != x.device:
                    self.pipe[0] = self.pipe[0].to(x.device, self.amp_dtype)
                sd_features = self.pipe[0](img_tensor.to(self.amp_dtype),
                                            t=self.t, 
                                            up_ft_indices=[self.up_ft_index],
                                            prompt_embeds=prompt_embeds)
                features_16 = sd_features['up_ft'][self.up_ft_index]
                _, c, h, w = features_16.shape
                features_16 = features_16.view(B, self.ensemble_size, c, h, w)
                features_16 = features_16.mean(dim=1, keepdim=False)  
                del sd_features
                feature_pyramid[16] = features_16
        return feature_pyramid

class CNNandSD16_8(nn.Module):
    def __init__(self, cnn_kwargs = None, amp = False, use_vgg = False, 
                sd_id = 'stabilityai/stable-diffusion-2-1', 
                amp_dtype = torch.float16, t=0,
                up_ft_index=1,
                ensemble_size=1):
        super().__init__()
        unet = MyUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")
        onestep_pipe = OneStepSDPipeline.from_pretrained(sd_id, unet=unet, safety_checker=None)
        onestep_pipe.vae.decoder = None
        onestep_pipe.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")
        gc.collect()
        onestep_pipe.enable_attention_slicing()
        onestep_pipe.enable_xformers_memory_efficient_attention()
        null_prompt = ''
        null_prompt_embeds = onestep_pipe._encode_prompt(
            prompt=null_prompt,
            device='cpu',
            num_images_per_prompt=1,
            do_classifier_free_guidance=False) # [1, 77, dim]

        self.null_prompt_embeds = null_prompt_embeds.to('cuda', dtype=amp_dtype)
        self.null_prompt = null_prompt
        onestep_pipe.text_encoder = onestep_pipe.text_encoder.to("cpu")
        onestep_pipe.tokenizer = None  # Free tokenizer if not needed

        self.t = t
        self.up_ft_index = up_ft_index
        self.ensemble_size =ensemble_size

        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        if not use_vgg:
            self.cnn = ResNet50(**cnn_kwargs)
        else:
            self.cnn = VGG19_2(**cnn_kwargs)
        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.amp:
            onestep_pipe = onestep_pipe.to("cuda", self.amp_dtype)
        self.pipe = [onestep_pipe] # ugly hack to not show parameters to DDP
    
    def train(self, mode: bool = True):
        return self.cnn.train(mode)
    
    def forward(self, x, upsample = False):
        B,C,H,W = x.shape
        feature_pyramid = self.cnn(x)

        img_tensor = x.unsqueeze(1).repeat(1, self.ensemble_size, 1, 1, 1)
        img_tensor = img_tensor.view(B * self.ensemble_size, C, H, W).cuda()
        with torch.no_grad():
            prompt_embeds = self.null_prompt_embeds.repeat(B*self.ensemble_size, 1, 1).to(self.amp_dtype)
            if self.pipe[0].device != x.device:
                self.pipe[0] = self.pipe[0].to(x.device, self.amp_dtype)
            sd_features = self.pipe[0](img_tensor.to(self.amp_dtype),
                                        t=self.t, 
                                        up_ft_indices=[self.up_ft_index,2],
                                        prompt_embeds=prompt_embeds)
            features_16 = sd_features['up_ft'][self.up_ft_index]
            _, c, h, w = features_16.shape
            features_16 = features_16.view(B, self.ensemble_size, c, h, w)
            features_16 = features_16.mean(dim=1, keepdim=False)  
            features_8 = sd_features['up_ft'][2]
            _, c, h, w = features_8.shape
            features_8 = features_8.view(B, self.ensemble_size, c, h, w)
            features_8 = features_8.mean(dim=1, keepdim=False)  
            del sd_features
            feature_pyramid[8] = features_8
            if not upsample:
                feature_pyramid[16] = features_16
        return feature_pyramid

class CNNandDinov2_SD8(nn.Module):
    def __init__(self, cnn_kwargs = None, amp = False, use_vgg = False, 
                dinov2_weights = None, 
                sd_id = 'stabilityai/stable-diffusion-2-1', 
                amp_dtype = torch.float16, t=0,
                up_ft_index=1,
                ensemble_size=1):
        super().__init__()
        if dinov2_weights is None:
            dinov2_weights = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", map_location="cpu")
        from .transformer import vit_large
        vit_kwargs = dict(img_size= 518,
            patch_size= 14,
            init_values = 1.0,
            ffn_layer = "mlp",
            block_chunks = 0,
        )

        dinov2_vitl14 = vit_large(**vit_kwargs).eval()
        dinov2_vitl14.load_state_dict(dinov2_weights)

        unet = MyUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")
        onestep_pipe = OneStepSDPipeline.from_pretrained(sd_id, unet=unet, safety_checker=None)
        onestep_pipe.vae.decoder = None
        onestep_pipe.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")
        gc.collect()
        onestep_pipe.enable_attention_slicing()
        onestep_pipe.enable_xformers_memory_efficient_attention()
        null_prompt = ''
        null_prompt_embeds = onestep_pipe._encode_prompt(
            prompt=null_prompt,
            device='cpu',
            num_images_per_prompt=1,
            do_classifier_free_guidance=False) # [1, 77, dim]

        self.null_prompt_embeds = null_prompt_embeds.to('cuda', dtype=amp_dtype)
        self.null_prompt = null_prompt
        onestep_pipe.text_encoder = onestep_pipe.text_encoder.to("cpu")
        onestep_pipe.tokenizer = None  # Free tokenizer if not needed

        self.t = t
        self.up_ft_index = up_ft_index
        self.ensemble_size =ensemble_size

        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        if not use_vgg:
            self.cnn = ResNet50(**cnn_kwargs)
        else:
            self.cnn = VGG19_2(**cnn_kwargs)
        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.amp:
            onestep_pipe = onestep_pipe.to("cuda", self.amp_dtype)
            dinov2_vitl14 = dinov2_vitl14.to("cuda", self.amp_dtype)
        self.pipe = [onestep_pipe] # ugly hack to not show parameters to DDP
        self.dinov2_vitl14 = [dinov2_vitl14] # ugly hack to not show parameters to DDP
    
    def train(self, mode: bool = True):
        return self.cnn.train(mode)
    
    def forward(self, x, upsample = False):
        B,C,H,W = x.shape
        feature_pyramid = self.cnn(x)

        img_tensor = x.unsqueeze(1).repeat(1, self.ensemble_size, 1, 1, 1)
        img_tensor = img_tensor.view(B * self.ensemble_size, C, H, W).cuda()
        #if not upsample:
        with torch.no_grad():
            prompt_embeds = self.null_prompt_embeds.repeat(B*self.ensemble_size, 1, 1).to(self.amp_dtype)
            if self.pipe[0].device != x.device:
                self.pipe[0] = self.pipe[0].to(x.device, self.amp_dtype)
            sd_features = self.pipe[0](img_tensor.to(self.amp_dtype),
                                        t=self.t, 
                                        up_ft_indices=[2],
                                        prompt_embeds=prompt_embeds)
            features_8 = sd_features['up_ft'][2]
            _, c, h, w = features_8.shape
            features_8 = features_8.view(B, self.ensemble_size, c, h, w)
            features_8 = features_8.mean(dim=1, keepdim=False)  
            del sd_features
            feature_pyramid[8] = features_8

            if self.dinov2_vitl14[0].device != x.device:
                self.dinov2_vitl14[0] = self.dinov2_vitl14[0].to(x.device, self.amp_dtype)
            dinov2_features_16 = self.dinov2_vitl14[0].forward_features(x.to(self.amp_dtype))
            features_16 = dinov2_features_16['x_norm_patchtokens'].permute(0,2,1).reshape(B,1024,H//14, W//14)
            del dinov2_features_16
            feature_pyramid[16] = features_16
        return feature_pyramid

class CNNandDinov2_SD(nn.Module): #concatenate dinov2 and SD at scale 16
    def __init__(self, cnn_kwargs = None, amp = False, use_vgg = False, 
                dinov2_weights = None, 
                sd_id = 'stabilityai/stable-diffusion-2-1', 
                amp_dtype = torch.float16, t=0,
                up_ft_index=1,
                ensemble_size=1,
                alpha = 0.5):
        super().__init__()
        if dinov2_weights is None:
            dinov2_weights = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", map_location="cpu")
        from .transformer import vit_large
        vit_kwargs = dict(img_size= 518,
            patch_size= 14,
            init_values = 1.0,
            ffn_layer = "mlp",
            block_chunks = 0,
        )

        dinov2_vitl14 = vit_large(**vit_kwargs).eval()
        dinov2_vitl14.load_state_dict(dinov2_weights)

        unet = MyUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")
        onestep_pipe = OneStepSDPipeline.from_pretrained(sd_id, unet=unet, safety_checker=None)
        onestep_pipe.vae.decoder = None
        onestep_pipe.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")
        gc.collect()
        onestep_pipe.enable_attention_slicing()
        onestep_pipe.enable_xformers_memory_efficient_attention()
        null_prompt = ''
        null_prompt_embeds = onestep_pipe._encode_prompt(
            prompt=null_prompt,
            device='cpu',
            num_images_per_prompt=1,
            do_classifier_free_guidance=False) # [1, 77, dim]

        self.null_prompt_embeds = null_prompt_embeds.to('cuda', dtype=amp_dtype)
        self.null_prompt = null_prompt
        onestep_pipe.text_encoder = onestep_pipe.text_encoder.to("cpu")
        onestep_pipe.tokenizer = None  # Free tokenizer if not needed

        self.t = t
        self.up_ft_index = up_ft_index
        self.ensemble_size =ensemble_size
        self.alpha = alpha

        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        if not use_vgg:
            self.cnn = ResNet50(**cnn_kwargs)
        else:
            self.cnn = VGG19(**cnn_kwargs)
        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.amp:
            onestep_pipe = onestep_pipe.to("cuda", self.amp_dtype)
            dinov2_vitl14 = dinov2_vitl14.to("cuda", self.amp_dtype)
        self.pipe = [onestep_pipe] # ugly hack to not show parameters to DDP
        self.dinov2_vitl14 = [dinov2_vitl14] # ugly hack to not show parameters to DDP
    
    def train(self, mode: bool = True):
        return self.cnn.train(mode)
    
    def forward(self, x, upsample = False):
        B,C,H,W = x.shape
        feature_pyramid = self.cnn(x)

        img_tensor = x.unsqueeze(1).repeat(1, self.ensemble_size, 1, 1, 1)
        img_tensor = img_tensor.view(B * self.ensemble_size, C, H, W).cuda()
        #if not upsample:
        with torch.no_grad():
            prompt_embeds = self.null_prompt_embeds.repeat(B*self.ensemble_size, 1, 1).to(self.amp_dtype)
            if self.pipe[0].device != x.device:
                self.pipe[0] = self.pipe[0].to(x.device, self.amp_dtype)
            sd_features = self.pipe[0](img_tensor.to(self.amp_dtype),
                                        t=self.t, 
                                        up_ft_indices=[self.up_ft_index],
                                        prompt_embeds=prompt_embeds)
            SD_features_16 = sd_features['up_ft'][self.up_ft_index]
            _, c, h, w = SD_features_16.shape
            SD_features_16 = SD_features_16.view(B, self.ensemble_size, c, h, w)
            SD_features_16 = SD_features_16.mean(dim=1, keepdim=False)  
            del sd_features
           
            if self.dinov2_vitl14[0].device != x.device:
                self.dinov2_vitl14[0] = self.dinov2_vitl14[0].to(x.device, self.amp_dtype)
            dinov2_features_16 = self.dinov2_vitl14[0].forward_features(x.to(self.amp_dtype))
            features_16 = dinov2_features_16['x_norm_patchtokens'].permute(0,2,1).reshape(B,1024,H//14, W//14)
            del dinov2_features_16

            SD_features_resized = F.interpolate(SD_features_16, size=(H//14, W//14), mode='bilinear', align_corners=False)
            SD_features_norm = SD_features_resized / SD_features_resized.norm(dim=1, keepdim=True)
            Dino_features_norm = features_16 / features_16.norm(dim=1, keepdim=True)
            
            feature_pyramid[16] = torch.cat([self.alpha * Dino_features_norm, self.alpha * SD_features_norm], dim=1)
        return feature_pyramid

class CNNandMast3r(nn.Module):
    def __init__(self, cnn_kwargs = None, amp = False, use_vgg = False, 
                dinov2_weights = None, 
                sd_id = 'stabilityai/stable-diffusion-2-1', 
                amp_dtype = torch.float16, t=0,
                up_ft_index=1,
                ensemble_size=1,
                alpha = 0.5):
        super().__init__()
        from mast3r.model import AsymmetricMASt3R
        mast3r_model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
                # you can put the path to a local checkpoint in model_name if needed
        model = AsymmetricMASt3R.from_pretrained(mast3r_model_name)
        #self.backbone = model
        #feature_dim = 1024 + 768
        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        if not use_vgg:
            self.cnn = ResNet50(**cnn_kwargs)
        else:
            self.cnn = VGG19(**cnn_kwargs)
        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.amp:
            model = model.to("cuda", self.amp_dtype)
        self.pipe = [model] # ugly hack to not show parameters to DDP
        
    def train(self, mode: bool = True):
        return self.cnn.train(mode)
    
    def forward(self, x, upsample = False):
        img1, img2 = x
        B,C,H,W = img1.shape
        feature_q_pyramid = self.cnn(img1)
        feature_s_pyramid = self.cnn(img2)

        if not upsample:
            with torch.no_grad():
                self.pipe[0] = self.pipe[0].to(img1.device, self.amp_dtype)
                feat1, feat2 = self.pipe[0].decoder_forward(img1.to(self.amp_dtype), img2.to(self.amp_dtype))
                feature_q_pyramid[16] = feat1
                feature_s_pyramid[16] = feat2
        return feature_q_pyramid, feature_s_pyramid

class CNNandMast3r_decoder(nn.Module):
    def __init__(self, cnn_kwargs = None, amp = False, use_vgg = False, 
                amp_dtype = torch.float16):
        super().__init__()
        from mast3r.model import AsymmetricMASt3R
        mast3r_model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
                # you can put the path to a local checkpoint in model_name if needed
        model = AsymmetricMASt3R.from_pretrained(mast3r_model_name)
        #self.backbone = model
        #feature_dim = 1024 + 768
        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        if not use_vgg:
            self.cnn = ResNet50(**cnn_kwargs)
        else:
            self.cnn = VGG19(**cnn_kwargs)
        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.amp:
            model = model.to("cuda", self.amp_dtype)
        self.pipe = [model] # ugly hack to not show parameters to DDP
        
    def train(self, mode: bool = True):
        return self.cnn.train(mode)
    
    def forward(self, x, upsample = False):
        img1, img2 = x
        B,C,H,W = img1.shape
        feature_q_pyramid = self.cnn(img1)
        feature_s_pyramid = self.cnn(img2)

        if not upsample:
            with torch.no_grad():
                self.pipe[0] = self.pipe[0].to(img1.device, self.amp_dtype)
                feat1, feat2 = self.pipe[0].decoder_only_forward(img1.to(self.amp_dtype), img2.to(self.amp_dtype))
                feature_q_pyramid[16] = feat1
                feature_s_pyramid[16] = feat2
        return feature_q_pyramid, feature_s_pyramid
    
class CNNandDinov2_Mast3r(nn.Module):
    def __init__(self, cnn_kwargs = None, amp = False, use_vgg = False, 
                dinov2_weights = None, 
                amp_dtype = torch.float16,
                alpha = 0.5):
        super().__init__()
        if dinov2_weights is None:
            dinov2_weights = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", map_location="cpu")
        from .transformer import vit_large
        vit_kwargs = dict(img_size= 518,
            patch_size= 14,
            init_values = 1.0,
            ffn_layer = "mlp",
            block_chunks = 0,
        )
        dinov2_vitl14 = vit_large(**vit_kwargs).eval()
        dinov2_vitl14.load_state_dict(dinov2_weights)
        from mast3r.model import AsymmetricMASt3R
        mast3r_model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
                # you can put the path to a local checkpoint in model_name if needed
        mast3r_model = AsymmetricMASt3R.from_pretrained(mast3r_model_name)
        #self.backbone = model
        #feature_dim = 1024 + 768
        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        if not use_vgg:
            self.cnn = ResNet50(**cnn_kwargs)
        else:
            self.cnn = VGG19(**cnn_kwargs)
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.alpha = alpha
        if self.amp:
            mast3r_model = mast3r_model.to("cuda", self.amp_dtype)
            dinov2_vitl14 = dinov2_vitl14.to("cuda", self.amp_dtype)
        self.pipe = [dinov2_vitl14, mast3r_model] # ugly hack to not show parameters to DDP
        
    def train(self, mode: bool = True):
        return self.cnn.train(mode)
    
    def forward(self, x, upsample = False):
        img1, img2 = x
        B,C,H,W = img1.shape
        feature_q_pyramid = self.cnn(img1)
        feature_s_pyramid = self.cnn(img2)

        if not upsample:
            with torch.no_grad():
                self.pipe[0] = self.pipe[0].to(img1.device, self.amp_dtype)
                dinov2_features_1 = self.pipe[0].forward_features(img1.to(self.amp_dtype))
                feat1_dinov2 = dinov2_features_1['x_norm_patchtokens'].permute(0,2,1).reshape(B,1024,H//14, W//14)
                dinov2_features_2 = self.pipe[0].forward_features(img2.to(self.amp_dtype))
                feat2_dinov2 = dinov2_features_2['x_norm_patchtokens'].permute(0,2,1).reshape(B,1024,H//14, W//14)
                del dinov2_features_1, dinov2_features_2
                feat1_dinov2_norm = feat1_dinov2 / feat1_dinov2.norm(dim=1, keepdim=True)
                feat2_dinov2_norm = feat2_dinov2 / feat2_dinov2.norm(dim=1, keepdim=True)

                self.pipe[1] = self.pipe[1].to(img1.device, self.amp_dtype)
                feat1, feat2 = self.pipe[1].decoder_forward(img1.to(self.amp_dtype), img2.to(self.amp_dtype))
                feat1_resized = F.interpolate(feat1, size=(H//14, W//14), mode='bilinear', align_corners=False)
                feat1_norm = feat1_resized / feat1_resized.norm(dim=1, keepdim=True)
                feat2_resized = F.interpolate(feat2, size=(H//14, W//14), mode='bilinear', align_corners=False)
                feat2_norm = feat2_resized / feat2_resized.norm(dim=1, keepdim=True)
                del feat1, feat2

                feature_q_pyramid[16] = torch.cat([self.alpha * feat1_dinov2_norm, self.alpha * feat1_norm], dim=1)
                feature_s_pyramid[16] = torch.cat([self.alpha * feat2_dinov2_norm, self.alpha * feat2_norm], dim=1)
        
        return feature_q_pyramid, feature_s_pyramid
    
class CNNandMast3r_trainable(nn.Module): #concatenate dinov2 and SD at scale 16
    def __init__(self, cnn_kwargs = None, amp = False, use_vgg = False, 
                amp_dtype = torch.float16):
        super().__init__()
        from mast3r.model import AsymmetricMASt3R
        mast3r_model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
                # you can put the path to a local checkpoint in model_name if needed
        model = AsymmetricMASt3R.from_pretrained(mast3r_model_name)
        #self.backbone = model
        #feature_dim = 1024 + 768
        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        if not use_vgg:
            self.cnn = ResNet50(**cnn_kwargs)
        else:
            self.cnn = VGG19(**cnn_kwargs)
        self.amp = amp
        self.amp_dtype = amp_dtype
        model.downstream_head1 = None
        model.downstream_head2 = None
        model = model.to("cuda")
        self.base_model = model
        #self.pipe = [model] # ugly hack to not show parameters to DDP

    def train(self, mode: bool = True):
        return self.cnn.train(mode)
    
    def forward(self, x, upsample = False):
        img1, img2 = x
        B,C,H,W = img1.shape
        feature_q_pyramid = self.cnn(img1)
        feature_s_pyramid = self.cnn(img2)
        feat1, feat2 = self.base_model.decoder_forward(img1, img2)
        feature_q_pyramid[16] = feat1
        feature_s_pyramid[16] = feat2
        return feature_q_pyramid, feature_s_pyramid


class CNNandMast3r_s16(nn.Module): #only coarse scale
    def __init__(self, cnn_kwargs = None, amp = False, use_vgg = False, 
                dinov2_weights = None, 
                sd_id = 'stabilityai/stable-diffusion-2-1', 
                amp_dtype = torch.float16, t=0,
                up_ft_index=1,
                ensemble_size=1,
                alpha = 0.5):
        super().__init__()
        from mast3r.model import AsymmetricMASt3R
        mast3r_model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
                # you can put the path to a local checkpoint in model_name if needed
        model = AsymmetricMASt3R.from_pretrained(mast3r_model_name)
        #self.backbone = model
        #feature_dim = 1024 + 768
        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        if not use_vgg:
            self.cnn = ResNet50(**cnn_kwargs)
        else:
            self.cnn = VGG19(**cnn_kwargs)
        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.amp:
            model = model.to("cuda", self.amp_dtype)
        
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in model.parameters():
            param.requires_grad = False
        self.pipe = [model] # ugly hack to not show parameters to DDP

    def forward(self, x, upsample = False):
        img1, img2 = x
        B,C,H,W = img1.shape
        
        #if not upsample:
        with torch.no_grad():
            feature_q_pyramid = self.cnn(img1)
            feature_s_pyramid = self.cnn(img2)
            self.pipe[0] = self.pipe[0].to(img1.device, self.amp_dtype)
            feat1, feat2 = self.pipe[0].decoder_forward(img1.to(self.amp_dtype), img2.to(self.amp_dtype))
            feature_q_pyramid[16] = feat1
            feature_s_pyramid[16] = feat2
        return feature_q_pyramid, feature_s_pyramid
