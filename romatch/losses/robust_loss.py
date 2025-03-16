from einops.einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from romatch.utils.utils import get_gt_warp, get_gt_pts
import wandb
import romatch
import math
import pdb
from dust3r.utils.geometry import inv, geotrf, normalize_pointcloud

def reg(xyz, conf):
    xyz = xyz.permute(0, 2, 3, 1)
    conf = conf.permute(0, 2, 3, 1)
    conf = conf[..., 0]
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)
    xyz = xyz * torch.expm1(d)
    conf = 1 + conf.exp().clip(max=float('inf')-1)
    return xyz, conf

def coords_grid(batch, ht, wd, device):#[B,2,H, W]
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

class RobustLosses(nn.Module):
    def __init__(
        self,
        robust=False,
        center_coords=False,
        scale_normalize=False,
        ce_weight=0.01,
        local_loss=True,
        local_dist=4.0,
        local_largest_scale=8,
        smooth_mask = False,
        depth_interpolation_mode = "bilinear",
        mask_depth_loss = False,
        relative_depth_error_threshold = 0.05,
        alpha = 1.,
        c = 1e-3,
    ):
        super().__init__()
        self.robust = robust  # measured in pixels
        self.center_coords = center_coords
        self.scale_normalize = scale_normalize
        self.ce_weight = ce_weight
        self.local_loss = local_loss
        self.local_dist = local_dist
        self.local_largest_scale = local_largest_scale
        self.smooth_mask = smooth_mask
        self.depth_interpolation_mode = depth_interpolation_mode
        self.mask_depth_loss = mask_depth_loss
        self.relative_depth_error_threshold = relative_depth_error_threshold
        self.avg_overlap = dict()
        self.alpha = alpha
        self.c = c

    def gm_cls_loss(self, x2, prob, scale_gm_cls, gm_certainty, scale):
        with torch.no_grad():
            B, C, H, W = scale_gm_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(C))
            G = torch.meshgrid(*[torch.linspace(-1+1/cls_res, 1 - 1/cls_res, steps = cls_res,device = device) for _ in range(2)], indexing='ij')
            G = torch.stack((G[1], G[0]), dim = -1).reshape(C,2)
            GT = (G[None,:,None,None,:]-x2[:,None]).norm(dim=-1).min(dim=1).indices ##ground coordinates in res scale [B,H,W,2]
        cls_loss = F.cross_entropy(scale_gm_cls, GT, reduction  = 'none')[prob > 0.99]
        certainty_loss = F.binary_cross_entropy_with_logits(gm_certainty[:,0], prob)
        if not torch.any(cls_loss):
            cls_loss = (certainty_loss * 0.0)  # Prevent issues where prob is 0 everywhere
            
        losses = {
            f"gm_certainty_loss_{scale}": certainty_loss.mean(),
            f"gm_cls_loss_{scale}": cls_loss.mean(),
        }
        wandb.log(losses, step = romatch.GLOBAL_STEP)
        return losses

    def delta_cls_loss(self, x2, prob, flow_pre_delta, delta_cls, certainty, scale, offset_scale):
        with torch.no_grad():
            B, C, H, W = delta_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(C))
            G = torch.meshgrid(*[torch.linspace(-1+1/cls_res, 1 - 1/cls_res, steps = cls_res,device = device) for _ in range(2)])
            G = torch.stack((G[1], G[0]), dim = -1).reshape(C,2) * offset_scale
            GT = (G[None,:,None,None,:] + flow_pre_delta[:,None] - x2[:,None]).norm(dim=-1).min(dim=1).indices
        cls_loss = F.cross_entropy(delta_cls, GT, reduction  = 'none')[prob > 0.99]
        certainty_loss = F.binary_cross_entropy_with_logits(certainty[:,0], prob)
        if not torch.any(cls_loss):
            cls_loss = (certainty_loss * 0.0)  # Prevent issues where prob is 0 everywhere
        losses = {
            f"delta_certainty_loss_{scale}": certainty_loss.mean(),
            f"delta_cls_loss_{scale}": cls_loss.mean(),
        }
        wandb.log(losses, step = romatch.GLOBAL_STEP)
        return losses

    def regression_loss(self, x2, prob, flow, certainty, scale, eps=1e-8, mode = "delta"):
        epe = (flow.permute(0,2,3,1) - x2).norm(dim=-1)
        if scale == 1:
            pck_05 = (epe[prob > 0.99] < 0.5 * (2/512)).float().mean()
            wandb.log({"train_pck_05": pck_05}, step = romatch.GLOBAL_STEP)

        ce_loss = F.binary_cross_entropy_with_logits(certainty[:, 0], prob)
        a = self.alpha[scale] if isinstance(self.alpha, dict) else self.alpha
        cs = self.c * scale
        x = epe[prob > 0.99]
        reg_loss = cs**a * ((x/(cs))**2 + 1**2)**(a/2)
        if not torch.any(reg_loss):
            reg_loss = (ce_loss * 0.0)  # Prevent issues where prob is 0 everywhere
        losses = {
            f"{mode}_certainty_loss_{scale}": ce_loss.mean(),
            f"{mode}_regression_loss_{scale}": reg_loss.mean(),
        }
        wandb.log(losses, step = romatch.GLOBAL_STEP)
        return losses
    
    

    def forward(self, corresps, batch):
        scales = list(corresps.keys())
        tot_loss = 0.0
        # scale_weights due to differences in scale for regression gradients and classification gradients
        scale_weights = {1:1, 2:1, 4:1, 8:1, 16:1}
        for scale in scales:
            scale_corresps = corresps[scale]
            scale_certainty, flow_pre_delta, delta_cls, offset_scale, scale_gm_cls, scale_gm_certainty, flow, scale_gm_flow = (
                scale_corresps["certainty"],
                scale_corresps.get("flow_pre_delta"),
                scale_corresps.get("delta_cls"),
                scale_corresps.get("offset_scale"),
                scale_corresps.get("gm_cls"),
                scale_corresps.get("gm_certainty"),
                scale_corresps["flow"],
                scale_corresps.get("gm_flow"),
                )
            if flow_pre_delta is not None:
                flow_pre_delta = rearrange(flow_pre_delta, "b d h w -> b h w d")
                b, h, w, d = flow_pre_delta.shape
            else:
                # _ = 1
                b, _, h, w = scale_certainty.shape
            gt_warp, gt_prob = get_gt_warp(                
            batch["im_A_depth"],
            batch["im_B_depth"],
            batch["T_1to2"],
            batch["K1"],
            batch["K2"],
            H=h,
            W=w,
        )
            x2 = gt_warp.float()
            prob = gt_prob
            
            if self.local_largest_scale >= scale:
                prob = prob * (
                        F.interpolate(prev_epe[:, None], size=(h, w), mode="nearest-exact")[:, 0]
                        < (2 / 512) * (self.local_dist[scale] * scale))
            
            if scale_gm_cls is not None:
                gm_cls_losses = self.gm_cls_loss(x2, prob, scale_gm_cls, scale_gm_certainty, scale)
                gm_loss = self.ce_weight * gm_cls_losses[f"gm_certainty_loss_{scale}"] + gm_cls_losses[f"gm_cls_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * gm_loss
            elif scale_gm_flow is not None:
                gm_flow_losses = self.regression_loss(x2, prob, scale_gm_flow, scale_gm_certainty, scale, mode = "gm")
                gm_loss = self.ce_weight * gm_flow_losses[f"gm_certainty_loss_{scale}"] + gm_flow_losses[f"gm_regression_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * gm_loss
            
            if delta_cls is not None:
                delta_cls_losses = self.delta_cls_loss(x2, prob, flow_pre_delta, delta_cls, scale_certainty, scale, offset_scale)
                delta_cls_loss = self.ce_weight * delta_cls_losses[f"delta_certainty_loss_{scale}"] + delta_cls_losses[f"delta_cls_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * delta_cls_loss
            else:
                delta_regression_losses = self.regression_loss(x2, prob, flow, scale_certainty, scale)
                reg_loss = self.ce_weight * delta_regression_losses[f"delta_certainty_loss_{scale}"] + delta_regression_losses[f"delta_regression_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * reg_loss
            prev_epe = (flow.permute(0,2,3,1) - x2).norm(dim=-1).detach()
        return tot_loss


class RobustLosses_dust3r(nn.Module):
    def __init__(
        self,
        robust=False,
        center_coords=False,
        scale_normalize=False,
        ce_weight=0.01,
        local_loss=True,
        local_dist=4.0,
        local_largest_scale=8,
        smooth_mask = False,
        depth_interpolation_mode = "bilinear",
        mask_depth_loss = False,
        relative_depth_error_threshold = 0.05,
        alpha = 1.,
        c = 1e-3,
        _alpha = 0.2,
    ):
        super().__init__()
        self.robust = robust  # measured in pixels
        self.center_coords = center_coords
        self.scale_normalize = scale_normalize
        self.ce_weight = ce_weight
        self.local_loss = local_loss
        self.local_dist = local_dist
        self.local_largest_scale = local_largest_scale
        self.smooth_mask = smooth_mask
        self.depth_interpolation_mode = depth_interpolation_mode
        self.mask_depth_loss = mask_depth_loss
        self.relative_depth_error_threshold = relative_depth_error_threshold
        self.avg_overlap = dict()
        self.alpha = alpha
        self.c = c
        self.criterion = L21Loss()
        self._alpha = _alpha

    def gm_cls_loss(self, x2, prob, scale_gm_cls, gm_certainty, scale):
        with torch.no_grad():
            B, C, H, W = scale_gm_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(C))
            G = torch.meshgrid(*[torch.linspace(-1+1/cls_res, 1 - 1/cls_res, steps = cls_res,device = device) for _ in range(2)], indexing='ij')
            G = torch.stack((G[1], G[0]), dim = -1).reshape(C,2)
            GT = (G[None,:,None,None,:]-x2[:,None]).norm(dim=-1).min(dim=1).indices ##ground coordinates in res scale [B,H,W,2]
        cls_loss = F.cross_entropy(scale_gm_cls, GT, reduction  = 'none')[prob > 0.99]
        certainty_loss = F.binary_cross_entropy_with_logits(gm_certainty[:,0], prob)
        if not torch.any(cls_loss):
            cls_loss = (certainty_loss * 0.0)  # Prevent issues where prob is 0 everywhere
            
        losses = {
            f"gm_certainty_loss_{scale}": certainty_loss.mean(),
            f"gm_cls_loss_{scale}": cls_loss.mean(),
        }
        wandb.log(losses, step = romatch.GLOBAL_STEP)
        return losses

    def delta_cls_loss(self, x2, prob, flow_pre_delta, delta_cls, certainty, scale, offset_scale):
        with torch.no_grad():
            B, C, H, W = delta_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(C))
            G = torch.meshgrid(*[torch.linspace(-1+1/cls_res, 1 - 1/cls_res, steps = cls_res,device = device) for _ in range(2)])
            G = torch.stack((G[1], G[0]), dim = -1).reshape(C,2) * offset_scale
            GT = (G[None,:,None,None,:] + flow_pre_delta[:,None] - x2[:,None]).norm(dim=-1).min(dim=1).indices
        cls_loss = F.cross_entropy(delta_cls, GT, reduction  = 'none')[prob > 0.99]
        certainty_loss = F.binary_cross_entropy_with_logits(certainty[:,0], prob)
        if not torch.any(cls_loss):
            cls_loss = (certainty_loss * 0.0)  # Prevent issues where prob is 0 everywhere
        losses = {
            f"delta_certainty_loss_{scale}": certainty_loss.mean(),
            f"delta_cls_loss_{scale}": cls_loss.mean(),
        }
        wandb.log(losses, step = romatch.GLOBAL_STEP)
        return losses

    def regression_loss(self, x2, prob, flow, certainty, scale, eps=1e-8, mode = "delta"):
        epe = (flow.permute(0,2,3,1) - x2).norm(dim=-1)
        if scale == 1:
            pck_05 = (epe[prob > 0.99] < 0.5 * (2/512)).float().mean()
            wandb.log({"train_pck_05": pck_05}, step = romatch.GLOBAL_STEP)

        ce_loss = F.binary_cross_entropy_with_logits(certainty[:, 0], prob)
        a = self.alpha[scale] if isinstance(self.alpha, dict) else self.alpha
        cs = self.c * scale
        x = epe[prob > 0.99]
        reg_loss = cs**a * ((x/(cs))**2 + 1**2)**(a/2)
        if not torch.any(reg_loss):
            reg_loss = (ce_loss * 0.0)  # Prevent issues where prob is 0 everywhere
        losses = {
            f"{mode}_certainty_loss_{scale}": ce_loss.mean(),
            f"{mode}_regression_loss_{scale}": reg_loss.mean(),
        }
        wandb.log(losses, step = romatch.GLOBAL_STEP)
        return losses

    def get_conf_log(self, x):
        return x, torch.log(x)

    def confloss(self, pred_pts3d1, pred_conf1, pred_pts3d2, pred_conf2, gt_pts3d1, gt_valid1, gt_pts3d2, gt_valid2, T1, scale, dist_clip = None, norm_mode = None, gt_scale = None):
        in_camera1 = inv(T1)
        gt_pts1 = geotrf(in_camera1, gt_pts3d1)  # B,H,W,3
        gt_pts2 = geotrf(in_camera1, gt_pts3d2)  # B,H,W,3
        valid1 = gt_valid1.clone()
        valid2 = gt_valid2.clone()
        if dist_clip is not None:
            dis1 = gt_pts1.norm(dim=-1)
            dis2 = gt_pts2.norm(dim=-1)
            valid1 = valid1 & (dis1 <= dist_clip)
            valid2 = valid2 & (dis2 <= dist_clip)
        
        pr_pts1, pr_pts2 = normalize_pointcloud(pred_pts3d1, pred_pts3d2, 'avg_dis', valid1, valid2)
        gt_pts1, gt_pts2 = normalize_pointcloud(gt_pts1, gt_pts2, self.norm_mode, valid1, valid2)
        l1 = self.criterion(pr_pts1[valid1], gt_pts1[valid1])
        l2 = self.criterion(pr_pts2[valid2], gt_pts2[valid2])

        if l1.numel() == 0:
            print('NO VALID POINTS in img1')
        if l2.numel() == 0:
            print('NO VALID POINTS in img2')
        # weight by confidence
        conf1, log_conf1 = self.get_conf_log(pred_conf1[valid1])
        conf2, log_conf2 = self.get_conf_log(pred_conf2[valid2])
        conf_loss1 = l1 * conf1 - self._alpha * log_conf1
        conf_loss2 = l2 * conf2 - self._alpha * log_conf2

        # average + nan protection (in case of no valid pixels at all)
        conf_loss1 = conf_loss1.mean() if conf_loss1.numel() > 0 else 0
        conf_loss2 = conf_loss2.mean() if conf_loss2.numel() > 0 else 0

        losses = {
            f"conf_loss1_{scale}": conf_loss1,
            f"conf_loss2_{scale}": conf_loss2,
        }
        wandb.log(losses, step = romatch.GLOBAL_STEP)
        return losses        

    def forward(self, corresps, batch):
        scales = list(corresps.keys())
        tot_loss = 0.0
        # scale_weights due to differences in scale for regression gradients and classification gradients
        scale_weights = {1:1, 2:1, 4:1, 8:1, 16:1}
        for scale in scales:
            scale_corresps = corresps[scale]
            scale_certainty, flow_pre_delta, delta_cls, offset_scale, scale_gm_cls, scale_gm_certainty, flow, scale_gm_flow, scale_pts3d1, scale_conf1, scale_pts3d2, scale_conf2 = (
                scale_corresps["certainty"],
                scale_corresps.get("flow_pre_delta"),
                scale_corresps.get("delta_cls"),
                scale_corresps.get("offset_scale"),
                scale_corresps.get("gm_cls"),
                scale_corresps.get("gm_certainty"),
                scale_corresps["flow"],
                scale_corresps.get("gm_flow"),
                scale_corresps.get('pts1'),
                scale_corresps.get('conf1'),
                scale_corresps.get('pts2'),
                scale_corresps.get('conf2'),
            )
            if flow_pre_delta is not None:
                flow_pre_delta = rearrange(flow_pre_delta, "b d h w -> b h w d")
                b, h, w, d = flow_pre_delta.shape
            else:
                # _ = 1
                b, _, h, w = scale_certainty.shape
            gt_warp, gt_prob = get_gt_warp(                
            batch["im_A_depth"],
            batch["im_B_depth"],
            batch["T_1to2"],
            batch["K1"],
            batch["K2"],
            H=h,
            W=w,
        )
            x2 = gt_warp.float()
            prob = gt_prob

            gt_pts1, gt_valid1 = get_gt_pts(
                batch["im_A_depth"],
                batch["K1"],
                batch["T1"],
                H=h,
                W=w,
            )
            gt_pts2, gt_valid2 = get_gt_pts(
                batch["im_VB_depth"],
                batch["K2"],
                batch["T2"],
                H=h,
                W=w,
            )
            T1 = batch['T1']
            if self.local_largest_scale >= scale:
                prob = prob * (
                        F.interpolate(prev_epe[:, None], size=(h, w), mode="nearest-exact")[:, 0]
                        < (2 / 512) * (self.local_dist[scale] * scale))
            
            if scale_gm_cls is not None:
                gm_cls_losses = self.gm_cls_loss(x2, prob, scale_gm_cls, scale_gm_certainty, scale)
                gm_loss = self.ce_weight * gm_cls_losses[f"gm_certainty_loss_{scale}"] + gm_cls_losses[f"gm_cls_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * gm_loss
            elif scale_gm_flow is not None:
                gm_flow_losses = self.regression_loss(x2, prob, scale_gm_flow, scale_gm_certainty, scale, mode = "gm")
                gm_loss = self.ce_weight * gm_flow_losses[f"gm_certainty_loss_{scale}"] + gm_flow_losses[f"gm_regression_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * gm_loss
            
            if delta_cls is not None:
                delta_cls_losses = self.delta_cls_loss(x2, prob, flow_pre_delta, delta_cls, scale_certainty, scale, offset_scale)
                delta_cls_loss = self.ce_weight * delta_cls_losses[f"delta_certainty_loss_{scale}"] + delta_cls_losses[f"delta_cls_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * delta_cls_loss
            else:
                delta_regression_losses = self.regression_loss(x2, prob, flow, scale_certainty, scale)
                reg_loss = self.ce_weight * delta_regression_losses[f"delta_certainty_loss_{scale}"] + delta_regression_losses[f"delta_regression_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * reg_loss
            prev_epe = (flow.permute(0,2,3,1) - x2).norm(dim=-1).detach()
            if scale_pts3d1 is not None:
                conf_loss = self.confloss(scale_pts3d1, scale_conf1, scale_pts3d2, scale_conf2, gt_pts1, gt_valid1, gt_pts2, gt_valid2, T1, scale)
                tot_loss = tot_loss + scale_weights[scale] * (conf_loss[f"conf_loss1_{scale}"]+conf_loss[f"conf_loss2_{scale}"])
        return tot_loss


class RobustLosses_Depth(nn.Module):
    def __init__(
        self,
        robust=False,
        center_coords=False,
        scale_normalize=False,
        ce_weight=0.01,
        local_loss=True,
        local_dist=4.0,
        local_largest_scale=8,
        smooth_mask = False,
        depth_interpolation_mode = "bilinear",
        mask_depth_loss = False,
        relative_depth_error_threshold = 0.05,
        alpha = 1.,
        c = 1e-3,
        _alpha = 0.2,
    ):
        super().__init__()
        self.robust = robust  # measured in pixels
        self.center_coords = center_coords
        self.scale_normalize = scale_normalize
        self.ce_weight = ce_weight
        self.local_loss = local_loss
        self.local_dist = local_dist
        self.local_largest_scale = local_largest_scale
        self.smooth_mask = smooth_mask
        self.depth_interpolation_mode = depth_interpolation_mode
        self.mask_depth_loss = mask_depth_loss
        self.relative_depth_error_threshold = relative_depth_error_threshold
        self.avg_overlap = dict()
        self.alpha = alpha
        self.c = c
        self.criterion = L21Loss()
        self._alpha = _alpha

    def gm_cls_loss(self, x2, prob, scale_gm_cls, gm_certainty, scale):
        with torch.no_grad():
            B, C, H, W = scale_gm_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(C))
            G = torch.meshgrid(*[torch.linspace(-1+1/cls_res, 1 - 1/cls_res, steps = cls_res,device = device) for _ in range(2)], indexing='ij')
            G = torch.stack((G[1], G[0]), dim = -1).reshape(C,2)
            GT = (G[None,:,None,None,:]-x2[:,None]).norm(dim=-1).min(dim=1).indices ##ground coordinates in res scale [B,H,W,2]
        cls_loss = F.cross_entropy(scale_gm_cls, GT, reduction  = 'none')[prob > 0.99]
        certainty_loss = F.binary_cross_entropy_with_logits(gm_certainty[:,0], prob)
        if not torch.any(cls_loss):
            cls_loss = (certainty_loss * 0.0)  # Prevent issues where prob is 0 everywhere
            
        losses = {
            f"gm_certainty_loss_{scale}": certainty_loss.mean(),
            f"gm_cls_loss_{scale}": cls_loss.mean(),
        }
        wandb.log(losses, step = romatch.GLOBAL_STEP)
        return losses

    def delta_cls_loss(self, x2, prob, flow_pre_delta, delta_cls, certainty, scale, offset_scale):
        with torch.no_grad():
            B, C, H, W = delta_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(C))
            G = torch.meshgrid(*[torch.linspace(-1+1/cls_res, 1 - 1/cls_res, steps = cls_res,device = device) for _ in range(2)])
            G = torch.stack((G[1], G[0]), dim = -1).reshape(C,2) * offset_scale
            GT = (G[None,:,None,None,:] + flow_pre_delta[:,None] - x2[:,None]).norm(dim=-1).min(dim=1).indices
        cls_loss = F.cross_entropy(delta_cls, GT, reduction  = 'none')[prob > 0.99]
        certainty_loss = F.binary_cross_entropy_with_logits(certainty[:,0], prob)
        if not torch.any(cls_loss):
            cls_loss = (certainty_loss * 0.0)  # Prevent issues where prob is 0 everywhere
        losses = {
            f"delta_certainty_loss_{scale}": certainty_loss.mean(),
            f"delta_cls_loss_{scale}": cls_loss.mean(),
        }
        wandb.log(losses, step = romatch.GLOBAL_STEP)
        return losses

    def regression_loss(self, x2, prob, flow, certainty, scale, eps=1e-8, mode = "delta"):
        epe = (flow.permute(0,2,3,1) - x2).norm(dim=-1)
        if scale == 1:
            pck_05 = (epe[prob > 0.99] < 0.5 * (2/512)).float().mean()
            wandb.log({"train_pck_05": pck_05}, step = romatch.GLOBAL_STEP)

        ce_loss = F.binary_cross_entropy_with_logits(certainty[:, 0], prob)
        a = self.alpha[scale] if isinstance(self.alpha, dict) else self.alpha
        cs = self.c * scale
        x = epe[prob > 0.99]
        reg_loss = cs**a * ((x/(cs))**2 + 1**2)**(a/2)
        if not torch.any(reg_loss):
            reg_loss = (ce_loss * 0.0)  # Prevent issues where prob is 0 everywhere
        losses = {
            f"{mode}_certainty_loss_{scale}": ce_loss.mean(),
            f"{mode}_regression_loss_{scale}": reg_loss.mean(),
        }
        wandb.log(losses, step = romatch.GLOBAL_STEP)
        return losses

    def get_conf_log(self, x):
        return x, torch.log(x)

    def Confloss(self, pred_pts3d1, pred_conf1, pred_pts3d2, pred_conf2, gt_pts3d1, gt_valid1, gt_pts3d2, gt_valid2, T1, scale, dist_clip = None, norm_mode = None, gt_scale = None):
        pred_pts3d1, pred_conf1 = reg(pred_pts3d1, pred_conf1)
        pred_pts3d2, pred_conf2 = reg(pred_pts3d2, pred_conf2)
        in_camera1 = inv(T1)
        gt_pts1 = geotrf(in_camera1, gt_pts3d1)  # B,H,W,3
        gt_pts2 = geotrf(in_camera1, gt_pts3d2)  # B,H,W,3
        valid1 = gt_valid1.clone() # B,H,W
        valid2 = gt_valid2.clone()
        if dist_clip is not None:
            dis1 = gt_pts1.norm(dim=-1)
            dis2 = gt_pts2.norm(dim=-1)
            valid1 = valid1 & (dis1 <= dist_clip)
            valid2 = valid2 & (dis2 <= dist_clip)
        epe1 = (pred_pts3d1 - gt_pts1).norm(dim=-1)
        epe2 = (pred_pts3d2 - gt_pts2).norm(dim=-1)
        pr_pts1, pr_pts2 = normalize_pointcloud(pred_pts3d1, pred_pts3d2, 'avg_dis', valid1, valid2)
        gt_pts1, gt_pts2 = normalize_pointcloud(gt_pts1, gt_pts2, 'avg_dis', valid1, valid2)
        l1 = self.criterion(pr_pts1[valid1], gt_pts1[valid1])
        l2 = self.criterion(pr_pts2[valid2], gt_pts2[valid2])
        
        if l1.numel() == 0:
            print('NO VALID POINTS in img1')
        if l2.numel() == 0:
            print('NO VALID POINTS in img2')
        # weight by confidence
        conf1, log_conf1 = self.get_conf_log(pred_conf1[valid1])
        conf2, log_conf2 = self.get_conf_log(pred_conf2[valid2])
        conf_loss1 = l1 * conf1 - self._alpha * log_conf1
        conf_loss2 = l2 * conf2 - self._alpha * log_conf2

        # average + nan protection (in case of no valid pixels at all)
        conf_loss1 = conf_loss1.mean() if conf_loss1.numel() > 0 else 0
        conf_loss2 = conf_loss2.mean() if conf_loss2.numel() > 0 else 0

        losses = {
            f"conf_loss1_{scale}": conf_loss1,
            f"conf_loss2_{scale}": conf_loss2,
        }
        wandb.log(losses, step = romatch.GLOBAL_STEP)
        return losses, epe1, epe2        

    def forward(self, corresps, batch):
        scales = list(corresps.keys())
        tot_loss = 0.0
        # scale_weights due to differences in scale for regression gradients and classification gradients
        scale_weights = {1:1, 2:1, 4:1, 8:1, 16:1}
        for scale in scales:
            scale_corresps = corresps[scale]
            scale_certainty, flow_pre_delta, delta_cls, offset_scale, scale_gm_cls, scale_gm_certainty, flow, scale_gm_flow, scale_pts3d1, scale_conf1, scale_pts3d2, scale_conf2 = (
                scale_corresps["certainty"],
                scale_corresps.get("flow_pre_delta"),
                scale_corresps.get("delta_cls"),
                scale_corresps.get("offset_scale"),
                scale_corresps.get("gm_cls"),
                scale_corresps.get("gm_certainty"),
                scale_corresps["flow"],
                scale_corresps.get("gm_flow"),
                scale_corresps.get('pts1'),
                scale_corresps.get('pts_conf1'),
                scale_corresps.get('pts2'),
                scale_corresps.get('pts_conf2'),
            )
            if flow_pre_delta is not None:
                flow_pre_delta = rearrange(flow_pre_delta, "b d h w -> b h w d")
                b, h, w, d = flow_pre_delta.shape
            else:
                # _ = 1
                b, _, h, w = scale_certainty.shape
            gt_warp, gt_prob = get_gt_warp(                
            batch["im_A_depth"],
            batch["im_B_depth"],
            batch["T_1to2"],
            batch["K1"],
            batch["K2"],
            H=h,
            W=w,
        )
            x2 = gt_warp.float()
            prob = gt_prob

            gt_pts1 = F.interpolate(batch['pts1'].permute(0,3,1,2), size=(h,w), mode='bilinear', align_corner=False).permute(0,2,3,1) #[B, H, W, 3]  
            gt_valid1 = F.interpolate(batch['mask1'].unsqueeze(1), size=(h,w), mode='nearest').squeeze(1) #[B, H, W]
            valid1 = gt_valid1

            gt_pts2 = F.interpolate(batch['pts2'].permute(0,3,1,2), size=(h,w), mode='bilinear', align_corner=False).permute(0,2,3,1) #[B, H, W, 3]  
            gt_valid2 = F.interpolate(batch['mask2'].unsqueeze(1), size=(h,w), mode='nearest').squeeze(1) #[B, H, W]
            valid2 = gt_valid2

            T1 = batch['T1']
            if self.local_largest_scale >= scale:
                prob = prob * (
                        F.interpolate(prev_epe[:, None], size=(h, w), mode="nearest-exact")[:, 0]
                        < (2 / 512) * (self.local_dist[scale] * scale))
                valid1 = valid1 * (
                     F.interpolate(prev_epe1[:, None], size=(h, w), mode="nearest-exact")[:, 0]
                        < (5 / 32) * (self.local_dist[scale] * scale)
                )
                valid2 = valid2 * (
                     F.interpolate(prev_epe2[:, None], size=(h, w), mode="nearest-exact")[:, 0]
                        < (1 / 2) * (self.local_dist[scale] * scale)
                )
            
            # if scale_gm_cls is not None:
            #     gm_cls_losses = self.gm_cls_loss(x2, prob, scale_gm_cls, scale_gm_certainty, scale)
            #     gm_loss = self.ce_weight * gm_cls_losses[f"gm_certainty_loss_{scale}"] + gm_cls_losses[f"gm_cls_loss_{scale}"]
            #     tot_loss = tot_loss + scale_weights[scale] * gm_loss
            # elif scale_gm_flow is not None:
            #     gm_flow_losses = self.regression_loss(x2, prob, scale_gm_flow, scale_gm_certainty, scale, mode = "gm")
            #     gm_loss = self.ce_weight * gm_flow_losses[f"gm_certainty_loss_{scale}"] + gm_flow_losses[f"gm_regression_loss_{scale}"]
            #     tot_loss = tot_loss + scale_weights[scale] * gm_loss
            
            # if delta_cls is not None:
            #     delta_cls_losses = self.delta_cls_loss(x2, prob, flow_pre_delta, delta_cls, scale_certainty, scale, offset_scale)
            #     delta_cls_loss = self.ce_weight * delta_cls_losses[f"delta_certainty_loss_{scale}"] + delta_cls_losses[f"delta_cls_loss_{scale}"]
            #     tot_loss = tot_loss + scale_weights[scale] * delta_cls_loss
            # else:
            #     delta_regression_losses = self.regression_loss(x2, prob, flow, scale_certainty, scale)
            #     reg_loss = self.ce_weight * delta_regression_losses[f"delta_certainty_loss_{scale}"] + delta_regression_losses[f"delta_regression_loss_{scale}"]
            #     tot_loss = tot_loss + scale_weights[scale] * reg_loss
            if scale_pts3d1 is not None and scale_pts3d2 is not None:
                conf_loss, epe1, epe2 = self.Confloss(scale_pts3d1, scale_conf1, scale_pts3d2, scale_conf2, gt_pts1, valid1, gt_pts2, valid2, T1, scale)
                tot_loss = tot_loss + scale_weights[scale] * (conf_loss[f"conf_loss1_{scale}"]+conf_loss[f"conf_loss2_{scale}"])
            prev_epe = (flow.permute(0,2,3,1) - x2).norm(dim=-1).detach()
            prev_epe1 = epe1.detach()
            prev_epe2 = epe2.detach()
        return tot_loss

class RobustLosses_only_dpt(nn.Module):
    def __init__(
        self,
        robust=False,
        center_coords=False,
        scale_normalize=False,
        ce_weight=0.01,
        local_loss=True,
        local_dist=4.0,
        local_largest_scale=8,
        smooth_mask = False,
        depth_interpolation_mode = "bilinear",
        mask_depth_loss = False,
        relative_depth_error_threshold = 0.05,
        alpha = 1.,
        c = 1e-3,
        _alpha = 0.2,
    ):
        super().__init__()
        self.robust = robust  # measured in pixels
        self.center_coords = center_coords
        self.scale_normalize = scale_normalize
        self.ce_weight = ce_weight
        self.local_loss = local_loss
        self.local_dist = local_dist
        self.local_largest_scale = local_largest_scale
        self.smooth_mask = smooth_mask
        self.depth_interpolation_mode = depth_interpolation_mode
        self.mask_depth_loss = mask_depth_loss
        self.relative_depth_error_threshold = relative_depth_error_threshold
        self.avg_overlap = dict()
        self.alpha = alpha
        self.c = c
        self.criterion = L21Loss()
        self._alpha = _alpha

    def get_conf_log(self, x):
        return x, torch.log(x)

    def Confloss(self, pred_pts3d1, pred_conf1, pred_pts3d2, pred_conf2, gt_pts3d1, gt_valid1, gt_pts3d2, gt_valid2, T1, scale, dist_clip = None, norm_mode = None, gt_scale = None):
        pred_pts3d1, pred_conf1 = reg(pred_pts3d1, pred_conf1)
        pred_pts3d2, pred_conf2 = reg(pred_pts3d2, pred_conf2)
        in_camera1 = inv(T1)
        gt_pts1 = geotrf(in_camera1, gt_pts3d1)  # B,H,W,3
        gt_pts2 = geotrf(in_camera1, gt_pts3d2)  # B,H,W,3
        valid1 = gt_valid1.clone() # B,H,W
        valid2 = gt_valid2.clone()
        if dist_clip is not None:
            dis1 = gt_pts1.norm(dim=-1)
            dis2 = gt_pts2.norm(dim=-1)
            valid1 = valid1 & (dis1 <= dist_clip)
            valid2 = valid2 & (dis2 <= dist_clip)
        epe1 = (pred_pts3d1 - gt_pts1).norm(dim=-1)
        epe2 = (pred_pts3d2 - gt_pts2).norm(dim=-1)
        pr_pts1, pr_pts2 = normalize_pointcloud(pred_pts3d1, pred_pts3d2, 'avg_dis', valid1, valid2)
        gt_pts1, gt_pts2 = normalize_pointcloud(gt_pts1, gt_pts2, 'avg_dis', valid1, valid2)
        l1 = self.criterion(pr_pts1[valid1], gt_pts1[valid1])
        l2 = self.criterion(pr_pts2[valid2], gt_pts2[valid2])
        
        if l1.numel() == 0:
            print('NO VALID POINTS in img1')
        if l2.numel() == 0:
            print('NO VALID POINTS in img2')
        # weight by confidence
        conf1, log_conf1 = self.get_conf_log(pred_conf1[valid1])
        conf2, log_conf2 = self.get_conf_log(pred_conf2[valid2])
        conf_loss1 = l1 * conf1 - self._alpha * log_conf1
        conf_loss2 = l2 * conf2 - self._alpha * log_conf2

        # average + nan protection (in case of no valid pixels at all)
        conf_loss1 = conf_loss1.mean() if conf_loss1.numel() > 0 else 0
        conf_loss2 = conf_loss2.mean() if conf_loss2.numel() > 0 else 0

        losses = {
            f"conf_loss1_{scale}": conf_loss1,
            f"conf_loss2_{scale}": conf_loss2,
        }
        wandb.log(losses, step = romatch.GLOBAL_STEP)
        return losses, epe1, epe2        

    def forward(self, corresps, batch):
        scales = list(corresps.keys())
        tot_loss = 0.0
        # scale_weights due to differences in scale for regression gradients and classification gradients
        scale_weights = {1:1, 2:1, 4:1, 8:1, 16:1}
        for scale in scales:
            scale_corresps = corresps[scale]
            scale_pts3d1, scale_conf1, scale_pts3d2, scale_conf2 = (
                scale_corresps.get('pts1'),
                scale_corresps.get('pts_conf1'),
                scale_corresps.get('pts2'),
                scale_corresps.get('pts_conf2'),
            )
            
            b, _, h, w = scale_pts3d1.shape
            
            gt_pts1, gt_valid1 = get_gt_pts(
                batch["im_A_depth"],
                batch["K1"],
                batch["T1"],
                H=h,
                W=w,
            )
            valid1 = gt_valid1

            gt_pts2, gt_valid2 = get_gt_pts(
                batch["im_B_depth"],
                batch["K2"],
                batch["T2"],
                H=h,
                W=w,
            )
            valid2 = gt_valid2

            T1 = batch['T1']
            if self.local_largest_scale >= scale:
                valid1 = valid1 * (
                     F.interpolate(prev_epe1[:, None], size=(h, w), mode="nearest-exact")[:, 0]
                        < (5 / 32) * (self.local_dist[scale] * scale)
                )
                valid2 = valid2 * (
                     F.interpolate(prev_epe2[:, None], size=(h, w), mode="nearest-exact")[:, 0]
                        < (1 / 2) * (self.local_dist[scale] * scale)
                )
            
            if scale_pts3d1 is not None and scale_pts3d2 is not None:
                conf_loss, epe1, epe2 = self.Confloss(scale_pts3d1, scale_conf1, scale_pts3d2, scale_conf2, gt_pts1, valid1, gt_pts2, valid2, T1, scale)
                tot_loss = tot_loss + scale_weights[scale] * (conf_loss[f"conf_loss1_{scale}"]+conf_loss[f"conf_loss2_{scale}"])
            prev_epe1 = epe1.detach()
            prev_epe2 = epe2.detach()
        return tot_loss


class RobustLosses_3D(nn.Module):
    def __init__(
        self,
        robust=False,
        center_coords=False,
        scale_normalize=False,
        ce_weight=0.01,
        local_loss=True,
        local_dist=4.0,
        local_largest_scale=8,
        smooth_mask = False,
        depth_interpolation_mode = "bilinear",
        mask_depth_loss = False,
        relative_depth_error_threshold = 0.05,
        alpha = 1.,
        c = 1e-3,
        _alpha = 0.2
    ):
        super().__init__()
        self.robust = robust  # measured in pixels
        self.center_coords = center_coords
        self.scale_normalize = scale_normalize
        self.ce_weight = ce_weight
        self.local_loss = local_loss
        self.local_dist = local_dist
        self.local_largest_scale = local_largest_scale
        self.smooth_mask = smooth_mask
        self.depth_interpolation_mode = depth_interpolation_mode
        self.mask_depth_loss = mask_depth_loss
        self.relative_depth_error_threshold = relative_depth_error_threshold
        self.avg_overlap = dict()
        self.alpha = alpha
        self.c = c
        self.criterion = L21Loss()
        self._alpha = _alpha

    def gm_cls_loss(self, x2, prob, scale_gm_cls, gm_certainty, scale):
        with torch.no_grad():
            B, C, H, W = scale_gm_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(C))
            G = torch.meshgrid(*[torch.linspace(-1+1/cls_res, 1 - 1/cls_res, steps = cls_res,device = device) for _ in range(2)], indexing='ij')
            G = torch.stack((G[1], G[0]), dim = -1).reshape(C,2)
            GT = (G[None,:,None,None,:]-x2[:,None]).norm(dim=-1).min(dim=1).indices ##ground coordinates in res scale [B,H,W,2]
        cls_loss = F.cross_entropy(scale_gm_cls, GT, reduction  = 'none')[prob > 0.99]
        certainty_loss = F.binary_cross_entropy_with_logits(gm_certainty[:,0], prob)
        if not torch.any(cls_loss):
            cls_loss = (certainty_loss * 0.0)  # Prevent issues where prob is 0 everywhere
            
        losses = {
            f"gm_certainty_loss_{scale}": certainty_loss.mean(),
            f"gm_cls_loss_{scale}": cls_loss.mean(),
        }
        wandb.log(losses, step = romatch.GLOBAL_STEP)
        return losses

    def delta_cls_loss(self, x2, prob, flow_pre_delta, delta_cls, certainty, scale, offset_scale):
        with torch.no_grad():
            B, C, H, W = delta_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(C))
            G = torch.meshgrid(*[torch.linspace(-1+1/cls_res, 1 - 1/cls_res, steps = cls_res,device = device) for _ in range(2)])
            G = torch.stack((G[1], G[0]), dim = -1).reshape(C,2) * offset_scale
            GT = (G[None,:,None,None,:] + flow_pre_delta[:,None] - x2[:,None]).norm(dim=-1).min(dim=1).indices
        cls_loss = F.cross_entropy(delta_cls, GT, reduction  = 'none')[prob > 0.99]
        certainty_loss = F.binary_cross_entropy_with_logits(certainty[:,0], prob)
        if not torch.any(cls_loss):
            cls_loss = (certainty_loss * 0.0)  # Prevent issues where prob is 0 everywhere
        losses = {
            f"delta_certainty_loss_{scale}": certainty_loss.mean(),
            f"delta_cls_loss_{scale}": cls_loss.mean(),
        }
        wandb.log(losses, step = romatch.GLOBAL_STEP)
        return losses

    def regression_loss(self, x2, prob, flow, certainty, scale, eps=1e-8, mode = "delta"):
        epe = (flow.permute(0,2,3,1) - x2).norm(dim=-1)
        if scale == 1:
            pck_05 = (epe[prob > 0.99] < 0.5 * (2/512)).float().mean()
            wandb.log({"train_pck_05": pck_05}, step = romatch.GLOBAL_STEP)

        ce_loss = F.binary_cross_entropy_with_logits(certainty[:, 0], prob)
        a = self.alpha[scale] if isinstance(self.alpha, dict) else self.alpha
        cs = self.c * scale
        x = epe[prob > 0.99]
        reg_loss = cs**a * ((x/(cs))**2 + 1**2)**(a/2)
        if not torch.any(reg_loss):
            reg_loss = (ce_loss * 0.0)  # Prevent issues where prob is 0 everywhere
        losses = {
            f"{mode}_certainty_loss_{scale}": ce_loss.mean(),
            f"{mode}_regression_loss_{scale}": reg_loss.mean(),
        }
        wandb.log(losses, step = romatch.GLOBAL_STEP)
        return losses
    
    def get_conf_log(self, x):
        return x, torch.log(x)

    def Confloss(self, gt_pts3d1, gt_valid1, gt_pts3d2, gt_valid2, T1, flow, scale_certainty, scale, dist_clip = None, norm_mode = None, gt_scale = None):
        certainty = scale_certainty.clone()
        B, _, H, W = flow.shape
        device = flow.device
        in_camera1 = inv(T1)
        gt_pts1 = geotrf(in_camera1, gt_pts3d1)  # B,H,W,3
        gt_pts2 = geotrf(in_camera1, gt_pts3d2)  # B,H,W,3
        valid1 = gt_valid1.clone() # B,H,W
        valid2 = gt_valid2.clone()
        if dist_clip is not None:
            dis1 = gt_pts1.norm(dim=-1)
            dis2 = gt_pts2.norm(dim=-1)
            valid1 = valid1 & (dis1 <= dist_clip)
            valid2 = valid2 & (dis2 <= dist_clip)
        gt_pts1, gt_pts2 = normalize_pointcloud(gt_pts1, gt_pts2, 'avg_dis', valid1, valid2)

        im_A_to_im_B = flow.permute(0, 2, 3, 1)
        # Create im_A meshgrid
        im_A_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=device),
                torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=device),
            ),
            indexing='ij'
        )
        im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
        im_A_coords = im_A_coords[None].expand(B, 2, H, W)
        grid1 = im_A_coords.permute(0, 2, 3, 1)

        grid2 = torch.clamp(im_A_to_im_B, -1, 1)
        pts1 = gt_pts3d1.permute(0, 3, 1, 2) #[B,3,H,W]
        pts2 = gt_pts3d2.permute(0, 3, 1, 2) 
        
        sampled_pts1 = F.grid_sample(pts1, grid1, align_corners=False, mode="bilinear")
        sampled_pts2 = F.grid_sample(pts2, grid2, align_corners=False, mode="bilinear")

        valid1 = valid1.float()
        valid2 = valid2.float()
        sampled_valid1 = F.grid_sample(valid1[:, None, :, :], grid1, align_corners=False, mode="nearest")  # [B,1,H,W]
        sampled_valid2 = F.grid_sample(valid2[:, None, :, :], grid2, align_corners=False, mode="nearest")
        mask = (sampled_valid1[:, 0, :, :] > 0.5) & (sampled_valid2[:, 0, :, :] > 0.5)

        sampled_pts1 = sampled_pts1.permute(0, 2, 3, 1) # B,H,W,3
        sampled_pts2 = sampled_pts2.permute(0, 2, 3, 1)
        sampled_pts1 = sampled_pts1[mask]
        sampled_pts2 = sampled_pts2[mask]

        l12 = self.criterion(sampled_pts1, sampled_pts2)
        
        certainty = 1 + torch.exp(certainty)  # logits -> probs
        if l12.numel() == 0:
            print('NO VALID POINTS in warp')
        conf, log_conf = self.get_conf_log(certainty[:,0,:,:][mask])
        conf_loss12 = l12 * conf - self._alpha * log_conf
        
        # average + nan protection (in case of no valid pixels at all)
        conf_loss12 = conf_loss12.mean() if conf_loss12.numel() > 0 else 0

        losses = {
            f"conf_loss12_{scale}":  conf_loss12,
        }
        wandb.log(losses, step = romatch.GLOBAL_STEP)
        return losses        

    def forward(self, corresps, batch):
        scales = list(corresps.keys())
        tot_loss = 0.0
        # scale_weights due to differences in scale for regression gradients and classification gradients
        scale_weights = {1:1, 2:1, 4:1, 8:1, 16:1}
        for scale in scales:
            scale_corresps = corresps[scale]
            scale_certainty, flow_pre_delta, delta_cls, offset_scale, scale_gm_cls, scale_gm_certainty, flow, scale_gm_flow, scale_pts3d1, scale_conf1, scale_pts3d2, scale_conf2 = (
                scale_corresps["certainty"],
                scale_corresps.get("flow_pre_delta"),
                scale_corresps.get("delta_cls"),
                scale_corresps.get("offset_scale"),
                scale_corresps.get("gm_cls"),
                scale_corresps.get("gm_certainty"),
                scale_corresps["flow"],
                scale_corresps.get("gm_flow"),
                scale_corresps.get('pts1'),
                scale_corresps.get('conf1'),
                scale_corresps.get('pts2'),
                scale_corresps.get('conf2'),
            )
            if flow_pre_delta is not None:
                flow_pre_delta = rearrange(flow_pre_delta, "b d h w -> b h w d")
                b, h, w, d = flow_pre_delta.shape
            else:
                # _ = 1
                b, _, h, w = scale_certainty.shape
            gt_warp, gt_prob = get_gt_warp(                
            batch["im_A_depth"],
            batch["im_B_depth"],
            batch["T_1to2"],
            batch["K1"],
            batch["K2"],
            H=h,
            W=w,
        )
            x2 = gt_warp.float()
            prob = gt_prob

            gt_pts1, gt_valid1 = get_gt_pts(
                batch["im_A_depth"],
                batch["K1"],
                batch["T1"],
                H=h,
                W=w,
            )
            gt_pts2, gt_valid2 = get_gt_pts(
                batch["im_B_depth"],
                batch["K2"],
                batch["T2"],
                H=h,
                W=w,
            )
            T1 = batch['T1']
            if self.local_largest_scale >= scale:
                prob = prob * (
                        F.interpolate(prev_epe[:, None], size=(h, w), mode="nearest-exact")[:, 0]
                        < (2 / 512) * (self.local_dist[scale] * scale))
            
            if scale_gm_cls is not None:
                gm_cls_losses = self.gm_cls_loss(x2, prob, scale_gm_cls, scale_gm_certainty, scale)
                gm_loss = self.ce_weight * gm_cls_losses[f"gm_certainty_loss_{scale}"] + gm_cls_losses[f"gm_cls_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * gm_loss
            elif scale_gm_flow is not None:
                gm_flow_losses = self.regression_loss(x2, prob, scale_gm_flow, scale_gm_certainty, scale, mode = "gm")
                gm_loss = self.ce_weight * gm_flow_losses[f"gm_certainty_loss_{scale}"] + gm_flow_losses[f"gm_regression_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * gm_loss
            
            if delta_cls is not None:
                delta_cls_losses = self.delta_cls_loss(x2, prob, flow_pre_delta, delta_cls, scale_certainty, scale, offset_scale)
                delta_cls_loss = self.ce_weight * delta_cls_losses[f"delta_certainty_loss_{scale}"] + delta_cls_losses[f"delta_cls_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * delta_cls_loss
            else:
                delta_regression_losses = self.regression_loss(x2, prob, flow, scale_certainty, scale)
                reg_loss = self.ce_weight * delta_regression_losses[f"delta_certainty_loss_{scale}"] + delta_regression_losses[f"delta_regression_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * reg_loss
            prev_epe = (flow.permute(0,2,3,1) - x2).norm(dim=-1).detach()
            conf_loss = self.Confloss(gt_pts1, gt_valid1, gt_pts2, gt_valid2, T1, flow, scale_certainty, scale)
            tot_loss = tot_loss + 0.1 * scale_weights[scale] * conf_loss[f"conf_loss12_{scale}"]
        return tot_loss

class L21Loss (nn.Module):
    """ L-norm loss
    """
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction
    def forward(self, a, b):
        assert a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3, f'Bad shape = {a.shape}'
        dist = self.distance(a, b)
        assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == 'none':
            return dist
        if self.reduction == 'sum':
            return dist.sum()
        if self.reduction == 'mean':
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f'bad {self.reduction=} mode')

    def distance(self, a, b):
        return torch.norm(a - b, dim=-1)  # normalized L2 distance