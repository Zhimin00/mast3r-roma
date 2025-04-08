# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# PatchEmbed implementation for DUST3R,
# in particular ManyAR_PatchEmbed that Handle images with non-square aspect ratio
# --------------------------------------------------------
import torch
import dust3r.utils.path_to_croco  # noqa: F401
from models.blocks import PatchEmbed  # noqa
import torchvision.models as tvm
import torch.nn as nn

def get_patch_embed(patch_embed_cls, img_size, patch_size, enc_embed_dim, **kwargs):
    assert patch_embed_cls in ['PatchEmbedDust3R', 'ManyAR_PatchEmbed','PatchEmbedDust3R2', 'ManyAR_PatchEmbed2','PatchEmbedDust3R_ResNet', 'ManyAR_PatchEmbed_ResNet', 'ManyAR_PatchEmbed_VGG', 'PatchEmbedDust3R_cnn','ManyAR_PatchEmbed_cnn']
    patch_embed = eval(patch_embed_cls)(img_size, patch_size, 3, enc_embed_dim, **kwargs)
    return patch_embed


class PatchEmbedDust3R(PatchEmbed):
    def forward(self, x, **kw):
        B, C, H, W = x.shape
        assert H % self.patch_size[0] == 0, f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert W % self.patch_size[1] == 0, f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, pos


class ManyAR_PatchEmbed (PatchEmbed):
    """ Handle images with non-square aspect ratio.
        All images in the same batch have the same aspect ratio.
        true_shape = [(height, width) ...] indicates the actual shape of each image.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        self.embed_dim = embed_dim
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten)

    def forward(self, img, true_shape):
        B, C, H, W = img.shape
        assert W >= H, f'img should be in landscape mode, but got {W=} {H=}'
        assert H % self.patch_size[0] == 0, f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert W % self.patch_size[1] == 0, f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        assert true_shape.shape == (B, 2), f"true_shape has the wrong shape={true_shape.shape}"

        # size expressed in tokens
        W //= self.patch_size[0]
        H //= self.patch_size[1]
        n_tokens = H * W

        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape

        # allocate result
        x = img.new_zeros((B, n_tokens, self.embed_dim))
        pos = img.new_zeros((B, n_tokens, 2), dtype=torch.int64)

        # linear projection, transposed if necessary
        x[is_landscape] = self.proj(img[is_landscape]).permute(0, 2, 3, 1).flatten(1, 2).float()
        x[is_portrait] = self.proj(img[is_portrait].swapaxes(-1, -2)).permute(0, 2, 3, 1).flatten(1, 2).float()

        pos[is_landscape] = self.position_getter(1, H, W, pos.device)
        pos[is_portrait] = self.position_getter(1, W, H, pos.device)

        x = self.norm(x)
        return x, pos

class ResNet50(nn.Module):
    def __init__(self, pretrained=False, dilation = None, freeze_bn = True, anti_aliased = False) -> None:
        super().__init__()
        if dilation is None:
            dilation = [False,False,False]
        if anti_aliased:
            pass
        else:
            self.net = tvm.resnet50(pretrained=pretrained,replace_stride_with_dilation=dilation)
        self.freeze_bn = freeze_bn

    def forward(self, x, **kwargs):
        net = self.net
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)
        feat4 = net.layer1(x) # scale 4
        feat8 = net.layer2(feat4) #scale 8       
        return feat4.permute(0, 2, 3, 1).flatten(1, 2).float(), feat8.permute(0, 2, 3, 1).flatten(1, 2).float()

class VGG19(nn.Module): #scale 8,4
    def __init__(self, pretrained=False) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])#40

    def forward(self, x, **kwargs):
        scale = 1
        for layer in self.layers:
            if isinstance(layer, nn.MaxPool2d):
                if scale == 4:
                    feat4 = x
                elif scale == 8:
                    feat8 = x
                scale = scale*2
            x = layer(x)
        return feat4.permute(0, 2, 3, 1).flatten(1, 2).float(), feat8.permute(0, 2, 3, 1).flatten(1, 2).float()
    

class ResNet50_all(nn.Module):
    def __init__(self, pretrained=False, dilation = None, freeze_bn = True, anti_aliased = False) -> None:
        super().__init__()
        if dilation is None:
            dilation = [False,False,False]
        if anti_aliased:
            pass
        else:
            self.net = tvm.resnet50(pretrained=pretrained,replace_stride_with_dilation=dilation)
        self.freeze_bn = freeze_bn

    def forward(self, x, **kwargs):
        feats = []
        net = self.net
        feats.append(x) # scale 1
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        feats.append(x) #scale 2
        x = net.maxpool(x) 
        x = net.layer1(x) # scale 4
        feats.append(x)
        x = net.layer2(x) #scale 8 
        feats.append(x)
        return [feat.permute(0, 2, 3, 1).flatten(1, 2).float() for feat in feats]

class VGG19_all(nn.Module): #scale 8,4,2,1
    def __init__(self, pretrained=False) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])#40

    def forward(self, x, **kwargs):
        feats = []
        scale = 1
        for layer in self.layers:
            if isinstance(layer, nn.MaxPool2d):
                feats.append(x)
                scale = scale*2
            x = layer(x)
        return [feat.permute(0, 2, 3, 1).flatten(1, 2).float() for feat in feats]

class PatchEmbedDust3R_ResNet (PatchEmbed):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        self.embed_dim = embed_dim
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten)
        self.cnn = ResNet50()
        
    def forward(self, x, **kw):
        B, C, H, W = x.shape
        assert H % self.patch_size[0] == 0, f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert W % self.patch_size[1] == 0, f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."

        feat4, feat8 = self.cnn(x)
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        
        return x, pos, feat4, feat8

class ManyAR_PatchEmbed_ResNet (PatchEmbed):
    """ Handle images with non-square aspect ratio.
        All images in the same batch have the same aspect ratio.
        true_shape = [(height, width) ...] indicates the actual shape of each image.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        self.embed_dim = embed_dim
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten)
        self.cnn = ResNet50()

    def forward(self, img, true_shape):
        B, C, H, W = img.shape
        assert W >= H, f'img should be in landscape mode, but got {W=} {H=}'
        assert H % self.patch_size[0] == 0, f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert W % self.patch_size[1] == 0, f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        assert true_shape.shape == (B, 2), f"true_shape has the wrong shape={true_shape.shape}"
        n4 = (W//4) * (H//4) 
        n8 = (W//8) * (H//8)
        # size expressed in tokens
        W //= self.patch_size[0]
        H //= self.patch_size[1]
        n_tokens = H * W

        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape

        # allocate result
        x = img.new_zeros((B, n_tokens, self.embed_dim))
        pos = img.new_zeros((B, n_tokens, 2), dtype=torch.int64)
        feat4 = img.new_zeros((B, n4, 256))
        feat8 = img.new_zeros((B, n8, 512))

        # linear projection, transposed if necessary
        x[is_landscape] = self.proj(img[is_landscape]).permute(0, 2, 3, 1).flatten(1, 2).float()
        x[is_portrait] = self.proj(img[is_portrait].swapaxes(-1, -2)).permute(0, 2, 3, 1).flatten(1, 2).float()
        pos[is_landscape] = self.position_getter(1, H, W, pos.device)
        pos[is_portrait] = self.position_getter(1, W, H, pos.device)
        x = self.norm(x)
        
        feat4[is_landscape], feat8[is_landscape] = self.cnn(img[is_landscape])
        feat4[is_portrait], feat8[is_portrait] = self.cnn(img[is_portrait].swapaxes(-1, -2))
              
        return x, pos, feat4, feat8

class ManyAR_PatchEmbed_VGG (PatchEmbed):
    """ Handle images with non-square aspect ratio.
        All images in the same batch have the same aspect ratio.
        true_shape = [(height, width) ...] indicates the actual shape of each image.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        self.embed_dim = embed_dim
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten)
        self.cnn = VGG19()

    def forward(self, img, true_shape):
        B, C, H, W = img.shape
        assert W >= H, f'img should be in landscape mode, but got {W=} {H=}'
        assert H % self.patch_size[0] == 0, f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert W % self.patch_size[1] == 0, f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        assert true_shape.shape == (B, 2), f"true_shape has the wrong shape={true_shape.shape}"
        n4 = (W//4) * (H//4) 
        n8 = (W//8) * (H//8)
        # size expressed in tokens
        W //= self.patch_size[0]
        H //= self.patch_size[1]
        n_tokens = H * W

        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape

        # allocate result
        x = img.new_zeros((B, n_tokens, self.embed_dim))
        pos = img.new_zeros((B, n_tokens, 2), dtype=torch.int64)
        feat4 = img.new_zeros((B, n4, 256))
        feat8 = img.new_zeros((B, n8, 512))

        # linear projection, transposed if necessary
        x[is_landscape] = self.proj(img[is_landscape]).permute(0, 2, 3, 1).flatten(1, 2).float()
        x[is_portrait] = self.proj(img[is_portrait].swapaxes(-1, -2)).permute(0, 2, 3, 1).flatten(1, 2).float()
        pos[is_landscape] = self.position_getter(1, H, W, pos.device)
        pos[is_portrait] = self.position_getter(1, W, H, pos.device)
        x = self.norm(x)
        
        feat4[is_landscape], feat8[is_landscape] = self.cnn(img[is_landscape])
        feat4[is_portrait], feat8[is_portrait] = self.cnn(img[is_portrait].swapaxes(-1, -2))
              
        return x, pos, feat4, feat8
    

class PatchEmbedDust3R_cnn (PatchEmbed):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, cnn_type = 'resnet', norm_layer=None, flatten=True):
        self.embed_dim = embed_dim
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten)
        if cnn_type == 'resnet':
            self.cnn = ResNet50_all()
            self.cnn_feature_dims = [3, 64, 256, 512]
        elif cnn_type == 'vgg':
            self.cnn = VGG19_all()
            self.cnn_feature_dims = [64, 128, 256, 512]
        
    def forward(self, x, **kw):
        B, C, H, W = x.shape
        assert H % self.patch_size[0] == 0, f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert W % self.patch_size[1] == 0, f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."

        feat1, feat2, feat4, feat8 = self.cnn(x)
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        cnn_feats = [feat1, feat2, feat4, feat8]
        return x, pos, cnn_feats
    
class ManyAR_PatchEmbed_cnn (PatchEmbed):
    """ Handle images with non-square aspect ratio.
        All images in the same batch have the same aspect ratio.
        true_shape = [(height, width) ...] indicates the actual shape of each image.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, cnn_type = 'resnet', norm_layer=None, flatten=True):
        self.embed_dim = embed_dim
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten)
        if cnn_type == 'resnet':
            self.cnn = ResNet50_all()
            self.cnn_feature_dims = [3, 64, 256, 512]
        elif cnn_type == 'vgg':
            self.cnn = VGG19_all()
            self.cnn_feature_dims = [64, 128, 256, 512]

    def forward(self, img, true_shape):
        B, C, H, W = img.shape
        assert W >= H, f'img should be in landscape mode, but got {W=} {H=}'
        assert H % self.patch_size[0] == 0, f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert W % self.patch_size[1] == 0, f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        assert true_shape.shape == (B, 2), f"true_shape has the wrong shape={true_shape.shape}"
        ns = [W * H, (W//2) * (H//2), (W//4) * (H//4), (W//8) * (H//8)]
        # size expressed in tokens
        W //= self.patch_size[0]
        H //= self.patch_size[1]
        n_tokens = H * W

        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape

        # allocate result
        x = img.new_zeros((B, n_tokens, self.embed_dim))
        pos = img.new_zeros((B, n_tokens, 2), dtype=torch.int64)
        feats = [img.new_zeros((B, ns[idx], self.cnn_feature_dims[idx])) for idx in range(len(self.cnn_feature_dims))]
        feat1, feat2, feat4, feat8 = feats
        # linear projection, transposed if necessary
        x[is_landscape] = self.proj(img[is_landscape]).permute(0, 2, 3, 1).flatten(1, 2).float()
        x[is_portrait] = self.proj(img[is_portrait].swapaxes(-1, -2)).permute(0, 2, 3, 1).flatten(1, 2).float()
        pos[is_landscape] = self.position_getter(1, H, W, pos.device)
        pos[is_portrait] = self.position_getter(1, W, H, pos.device)
        x = self.norm(x)
        
        feat1[is_landscape], feat2[is_landscape], feat4[is_landscape], feat8[is_landscape] = self.cnn(img[is_landscape])
        feat1[is_portrait], feat2[is_portrait], feat4[is_portrait], feat8[is_portrait] = self.cnn(img[is_portrait].swapaxes(-1, -2))
        cnn_feats = [feat1, feat2, feat4, feat8]
        return x, pos, cnn_feats
    
class PatchEmbedDust3R2(PatchEmbed):
    def forward(self, x, **kw):
        B, C, H, W = x.shape
        assert H % self.patch_size[0] == 0, f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert W % self.patch_size[1] == 0, f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, pos, H, W

        # B, C, H, W = x.shape
        # assert H % self.patch_size[0] == 0, f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        # assert W % self.patch_size[1] == 0, f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        # W //= self.patch_size[0]
        # H //= self.patch_size[1]
        # pos = self.position_getter(B, H, W, x.device)
        # return x, pos


class ManyAR_PatchEmbed2 (PatchEmbed):
    """ Handle images with non-square aspect ratio.
        All images in the same batch have the same aspect ratio.
        true_shape = [(height, width) ...] indicates the actual shape of each image.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        self.embed_dim = embed_dim
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten)

    def forward(self, img, true_shape):
        B, C, H, W = img.shape
        h, w = H, W
        assert W >= H, f'img should be in landscape mode, but got {W=} {H=}'
        assert H % self.patch_size[0] == 0, f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert W % self.patch_size[1] == 0, f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        assert true_shape.shape == (B, 2), f"true_shape has the wrong shape={true_shape.shape}"

        # size expressed in tokens
        W //= self.patch_size[0]
        H //= self.patch_size[1]
        n_tokens = H * W

        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape

        # allocate result
        x = img.new_zeros((B, n_tokens, self.embed_dim))
        pos = img.new_zeros((B, n_tokens, 2), dtype=torch.int64)

        # linear projection, transposed if necessary
        x[is_landscape] = self.proj(img[is_landscape]).permute(0, 2, 3, 1).flatten(1, 2).float()
        x[is_portrait] = self.proj(img[is_portrait].swapaxes(-1, -2)).permute(0, 2, 3, 1).flatten(1, 2).float()

        pos[is_landscape] = self.position_getter(1, H, W, pos.device)
        pos[is_portrait] = self.position_getter(1, W, H, pos.device)

        x = self.norm(x)
        return x, pos, h, w