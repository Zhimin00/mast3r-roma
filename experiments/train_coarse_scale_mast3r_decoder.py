import os
import torch
from argparse import ArgumentParser

from torch import nn
from torch.utils.data import ConcatDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import wandb

from romatch.benchmarks import MegadepthDenseBenchmark
from romatch.datasets.megadepth import MegadepthBuilder
from romatch.losses.robust_loss import RobustLosses
from romatch.benchmarks import MegaDepthPoseEstimationBenchmark, MegadepthDenseBenchmark, HpatchesHomogBenchmark

from romatch.train.train import train_k_steps
from romatch.models.matcher import *
from romatch.models.transformer import Block, TransformerDecoder, MemEffAttention
from romatch.models.encoders import *
from romatch.checkpointing import CheckPoint, CheckPoint2
from romatch.utils.utils import get_gt_warp, cls_to_flow
import tqdm
from romatch.utils import warp_kpts

resolutions = {"low":(448, 448), "medium":(14*8*5, 14*8*5), "high":(14*8*6, 14*8*6)}

def reshape(x):
    return rearrange(x, "b d h w -> b (h w) d")


class MegadepthDenseBenchmark2:
    def __init__(self, data_root="data/megadepth", h = 384, w = 512, num_samples = 2000) -> None:
        mega = MegadepthBuilder(data_root=data_root)
        self.dataset = ConcatDataset(
            mega.build_scenes(split="test_loftr", ht=h, wt=w)
        )  # fixed resolution of 384,512
        self.num_samples = num_samples

    def geometric_dist(self, depth1, depth2, T_1to2, K1, K2, dense_matches):
        b, h1, w1, d = dense_matches.shape
        with torch.no_grad():
            x1 = dense_matches[..., :2].reshape(b, h1 * w1, 2)
            mask, x2 = warp_kpts(
                x1.double(),
                depth1.double(),
                depth2.double(),
                T_1to2.double(),
                K1.double(),
                K2.double(),
            )
            x2 = torch.stack(
                (w1 * (x2[..., 0] + 1) / 2, h1 * (x2[..., 1] + 1) / 2), dim=-1
            )
            prob = mask.float().reshape(b, h1, w1)
        x2_hat = dense_matches[..., 2:]
        x2_hat = torch.stack(
            (w1 * (x2_hat[..., 0] + 1) / 2, h1 * (x2_hat[..., 1] + 1) / 2), dim=-1
        )
        gd = (x2_hat - x2.reshape(b, h1, w1, 2)).norm(dim=-1)
        gd = gd[prob == 1]
        pck_1 = (gd < 1.0).float().mean()
        pck_3 = (gd < 3.0).float().mean()
        pck_5 = (gd < 5.0).float().mean()
        robust_scale = (gd < 2.0).float().mean()
        robustness = (gd < 32.0).float().mean()
        return gd, pck_1, pck_3, pck_5, prob, robust_scale, robustness

    def benchmark(self, model, batch_size=8):
        model.train(False)
        with torch.no_grad():
            gd_tot = 0.0
            pck_1_tot = 0.0
            pck_3_tot = 0.0
            pck_5_tot = 0.0
            robust_scale_tot = 0.0
            robustness_tot = 0.0
            sampler = torch.utils.data.WeightedRandomSampler(
                torch.ones(len(self.dataset)), replacement=False, num_samples=self.num_samples
            )
            B = batch_size
            dataloader = torch.utils.data.DataLoader(
                self.dataset, batch_size=B, num_workers=batch_size, sampler=sampler
            )
            for idx, data in tqdm.tqdm(enumerate(dataloader), disable = romatch.RANK > 0):
                im_A, im_B, depth1, depth2, T_1to2, K1, K2 = (
                    data["im_A"].cuda(),
                    data["im_B"].cuda(),
                    data["im_A_depth"].cuda(),
                    data["im_B_depth"].cuda(),
                    data["T_1to2"].cuda(),
                    data["K1"].cuda(),
                    data["K2"].cuda(),
                )
                matches, certainty = model.match(im_A, im_B, batched=True)
                gd, pck_1, pck_3, pck_5, prob, robust_scale, robustness = self.geometric_dist(
                    depth1, depth2, T_1to2, K1, K2, matches
                )
                if romatch.DEBUG_MODE:
                    from romatch.utils.utils import tensor_to_pil
                    import torch.nn.functional as F
                    path = "vis"
                    H, W = model.get_output_resolution()
                    white_im = torch.ones((B,1,H,W),device="cuda")
                    im_B_transfer_rgb = F.grid_sample(
                        im_B.cuda(), matches[:,:,:W, 2:], mode="bilinear", align_corners=False
                    )
                    warp_im = im_B_transfer_rgb
                    c_b = certainty[:,None]#(certainty*0.9 + 0.1*torch.ones_like(certainty))[:,None]
                    vis_im = c_b * warp_im + (1 - c_b) * white_im
                    for b in range(B):
                        import os
                        os.makedirs(f"{path}/{model.name}/{idx}_{b}_{H}_{W}",exist_ok=True)
                        tensor_to_pil(vis_im[b], unnormalize=True).save(
                            f"{path}/{model.name}/{idx}_{b}_{H}_{W}/warp.jpg")
                        tensor_to_pil(im_A[b].cuda(), unnormalize=True).save(
                            f"{path}/{model.name}/{idx}_{b}_{H}_{W}/im_A.jpg")
                        tensor_to_pil(im_B[b].cuda(), unnormalize=True).save(
                            f"{path}/{model.name}/{idx}_{b}_{H}_{W}/im_B.jpg")


                gd_tot, pck_1_tot, pck_3_tot, pck_5_tot, robust_scale_tot, robustness_tot = (
                    gd_tot + gd.mean(),
                    pck_1_tot + pck_1,
                    pck_3_tot + pck_3,
                    pck_5_tot + pck_5,
                    robust_scale_tot + robust_scale,
                    robustness_tot + robustness,
                )
        return {
            "epe": gd_tot.item() / len(dataloader),
            "mega_pck_1": pck_1_tot.item() / len(dataloader),
            "mega_pck_3": pck_3_tot.item() / len(dataloader),
            "mega_pck_5": pck_5_tot.item() / len(dataloader),
            "mega_robust_scale": robust_scale_tot.item() / len(dataloader),
            "mega_robust": robustness_tot.item() / len(dataloader)
        }


class mnn_matcher(nn.Module):
    def __init__(
        self,
        encoder,
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
        #gm_warp_or_cls = torch.where(distances == max_vals, torch.tensor(10.0, device=distances.device), torch.tensor(-10.0, device=distances.device))

        #output = torch.full_like(distances, -10.)
        #gm_warp_or_cls = torch.where(distances == max_vals, 10., output)
        #gm_warp_or_cls.requires_grad_(True)
        
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

            corresps = self.forward(batch, batched=True)
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

class RegByClsLoss(nn.Module):
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
            GT = (G[None,:,None,None,:]-x2[:,None]).norm(dim=-1).min(dim=1).indices
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


    def forward(self, corresps, batch):
        tot_loss = 0.0
        scale_corresps = corresps
        scale_gm_cls, scale_gm_certainty = (
            scale_corresps.get("gm_cls"),
            scale_corresps.get("gm_certainty"),
        )
        b, _, h, w = scale_gm_certainty.shape
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

        gm_cls_losses = self.gm_cls_loss(x2, prob, scale_gm_cls, scale_gm_certainty, 16)
        gm_loss = gm_cls_losses[f"gm_cls_loss_16"]#self.ce_weight * gm_cls_losses[f"gm_certainty_loss_16"] + gm_cls_losses[f"gm_cls_loss_16"]
        tot_loss = tot_loss + gm_loss
            
        return tot_loss
    


def get_model(pretrained_backbone = True, model_name = 'resnet50', resolution = "low", t = 0, alpha = 0.5,**kwargs):
    
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
    #encoder = FeatureExtractor_s16(model_name= model_name, pretrained=pretrained_backbone, t=t, alpha=alpha)#FeatureExtractor_woProj(model_name= model_name, pretrained=pretrained_backbone, t=t)
    h, w = resolutions[resolution]
    print(h, w)
    #matcher = mnn_matcher(encoder, h, w)

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
    proj = nn.ModuleDict({
        "16": proj16,
        })
    displacement_dropout_p = 0.0
    gm_warp_dropout_p = 0.0
    decoder = Decoder(coordinate_decoder, 
                      gps, 
                      proj, 
                      conv_refiner, 
                      detach=True, 
                      scales=["16"],#, "8", "4", "2", "1"], 
                      displacement_dropout_p = displacement_dropout_p,
                      gm_warp_dropout_p = gm_warp_dropout_p)
    h,w = resolutions[resolution]
    encoder = CNNandMast3r_16(
        cnn_kwargs = dict(
            pretrained=pretrained_backbone,
            amp = True),
        amp = True,
        use_vgg = True,
    )
    matcher = RegressionMatcher_mast3r_coarse(encoder, decoder, h=h, w=w,**kwargs)
    

    return matcher


def train(args):
    dist.init_process_group('nccl')
    #torch._dynamo.config.verbose=True
    gpus = int(os.environ['WORLD_SIZE'])
    # create model and move it to GPU with id rank
    rank = dist.get_rank()
    print(f"Start running DDP on rank {rank}")
    device_id = rank % torch.cuda.device_count()
    romatch.LOCAL_RANK = device_id
    torch.cuda.set_device(device_id)
    
    resolution = args.train_resolution
    wandb_log = not args.dont_log_wandb
    experiment_name = os.path.splitext(os.path.basename(__file__))[0]
    model_name = args.experiment
    print(model_name)

    wandb_mode = "online" if wandb_log and rank == 0 else "disabled"
    wandb.init(project="romatch", entity=args.wandb_entity, name=experiment_name + '_' + model_name, reinit=False, mode = wandb_mode)
    checkpoint_dir = "/cis/net/r24a/data/zshao/roma_checkpoints/coarse_refiner/"
    h,w = resolutions[resolution]
    model = get_model(pretrained_backbone=True, model_name=model_name, resolution=resolution, t=args.t, alpha = args.alpha, attenuate_cert = False).to(device_id)
    for name,module in model.named_children():
        print(f"Submodule: {name}")
        for param_name, param in module.named_parameters():
            if param.requires_grad:
                print(f" - {param_name} is trainable")
    # Num steps
    global_step = 0
    batch_size = args.gpu_batch_size
    step_size = gpus*batch_size
    romatch.STEP_SIZE = step_size
    
    N = (32 * 250000)  # 250k steps of batch size 32
    # checkpoint every
    k = 25000 // romatch.STEP_SIZE

    # Data "/export/r33/data/r24_data/megadepth"
    dataset_path = '/cis/net/io93/data/zshao/megadepth'
    mega = MegadepthBuilder(data_root=dataset_path, loftr_ignore=True, imc21_ignore = True)
    use_horizontal_flip_aug = True
    rot_prob = 0
    depth_interpolation_mode = "bilinear"
    megadepth_train1 = mega.build_scenes(
        split="train_loftr", min_overlap=0.01, shake_t=32, use_horizontal_flip_aug = use_horizontal_flip_aug, rot_prob = rot_prob,
        ht=h,wt=w,
    )
    megadepth_train2 = mega.build_scenes(
        split="train_loftr", min_overlap=0.35, shake_t=32, use_horizontal_flip_aug = use_horizontal_flip_aug, rot_prob = rot_prob,
        ht=h,wt=w,
    )
    megadepth_train = ConcatDataset(megadepth_train1 + megadepth_train2)
    mega_ws = mega.weight_scenes(megadepth_train, alpha=0.75)


    # Loss and optimizer
    depth_loss = RobustLosses(
        ce_weight=0.01, 
        local_dist={1:4, 2:4, 4:8, 8:8},
        local_largest_scale=8,
        depth_interpolation_mode=depth_interpolation_mode,
        alpha = 0.5,
        c = 1e-4,)
    parameters = [
        {"params": model.encoder.parameters(), "lr": romatch.STEP_SIZE * 5e-6 / 8},
        {"params": model.decoder.parameters(), "lr": romatch.STEP_SIZE * 1e-4 / 8},
    ]
    optimizer = torch.optim.AdamW(parameters, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[(9*N/romatch.STEP_SIZE)//20])
    megadense_benchmark = MegadepthDenseBenchmark2(dataset_path, num_samples = 1000, h=h,w=w)
    checkpointer = CheckPoint2(checkpoint_dir, experiment_name, model_name)
    model, optimizer, lr_scheduler, global_step = checkpointer.load(model, optimizer, lr_scheduler, global_step)
    romatch.GLOBAL_STEP = global_step
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters = False, gradient_as_bucket_view=True)
    grad_scaler = torch.cuda.amp.GradScaler(growth_interval=1_000_000)
    grad_clip_norm = 0.01
    for n in range(romatch.GLOBAL_STEP, N, k * romatch.STEP_SIZE):
        mega_sampler = torch.utils.data.WeightedRandomSampler(
            mega_ws, num_samples = batch_size * k, replacement=False
        )
        mega_dataloader = iter(
            torch.utils.data.DataLoader(
                megadepth_train,
                batch_size = batch_size,
                sampler = mega_sampler,
                num_workers = 8,
            )
        )
        train_k_steps(
            n, k, mega_dataloader, ddp_model, depth_loss, optimizer, lr_scheduler, grad_scaler, grad_clip_norm = grad_clip_norm,
        )
        checkpointer.save(model, optimizer, lr_scheduler, romatch.GLOBAL_STEP)
        wandb.log(megadense_benchmark.benchmark(model), step = romatch.GLOBAL_STEP)

def test_mega1500(model, name):
    mega1500_benchmark = MegaDepthPoseEstimationBenchmark("/cis/net/r24a/data/zshao/data/megadepth")
    mega1500_results = mega1500_benchmark.benchmark(model, model_name=name)
    json.dump(mega1500_results, open(f"./results/mega1500_{name}.json", "w"))

def test_mega_dense(model, name):
    megadense_benchmark = MegadepthDenseBenchmark("/cis/net/r24a/data/zshao/data/megadepth", num_samples = 1000)
    megadense_results = megadense_benchmark.benchmark(model)
    json.dump(megadense_results, open(f"./results/mega_dense_{name}.json", "w"))



if __name__ == "__main__":
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1" # For BF16 computations
    os.environ["OMP_NUM_THREADS"] = "16"
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    import romatch
    parser = ArgumentParser()
    parser.add_argument("--only_test", action='store_true')
    parser.add_argument("--debug_mode", action='store_true')
    parser.add_argument("--dont_log_wandb", action='store_true')
    parser.add_argument("--train_resolution", default='low')
    parser.add_argument("--gpu_batch_size", default=1, type=int)
    parser.add_argument("--wandb_entity", required = False)
    parser.add_argument("--experiment", default='diffusion')
    parser.add_argument("--t", type = int, default=0)
    parser.add_argument("--alpha", type = float, default=0.5)

    args, _ = parser.parse_known_args()
    romatch.DEBUG_MODE = args.debug_mode
    if not args.only_test:
        train(args)