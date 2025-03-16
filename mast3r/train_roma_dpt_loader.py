import os
import torch
from argparse import ArgumentParser

from torch import nn
from torch.utils.data import ConcatDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import SyncBatchNorm
import json
import wandb

from romatch.benchmarks import MegadepthDenseBenchmark
from romatch.datasets.megadepth import MegadepthBuilder
from romatch.losses.robust_loss import RobustLosses, RobustLosses_Depth
from romatch.benchmarks import MegaDepthPoseEstimationBenchmark, MegadepthDenseBenchmark, HpatchesHomogBenchmark, MegadepthDenseBenchmark_depth

from romatch.train.train import train_k_steps, train_k_steps_new, train_epoch_new
from romatch.models.matcher import *
from romatch.models.transformer import Block, TransformerDecoder, MemEffAttention
from romatch.models.encoders import *
from romatch.checkpointing import CheckPoint
import pdb
import torch.backends.cudnn as cudnn
import croco.utils.misc as misc
from dust3r.datasets import get_data_loader  


resolutions = {"low":(448, 448), "medium":(14*8*5, 14*8*5), "high":(14*8*6, 14*8*6)}
weight_urls = {
    "romatch": {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",
        "indoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_indoor.pth",
    },
    "tiny_roma_v1": {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/tiny_roma_v1_outdoor.pth",
    },
    "dinov2": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", #hopefully this doesnt change :D
}

def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))

    print(f"{split} dataset length: ", len(loader))
    return loader

def get_SD_model(pretrained_backbone=True, resolution = "medium", **kwargs):
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
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
    dpt1 = DPT(in_dim=512, hidden_dim=256, out_dim=4, kernel_size = kernel_size, amp=True)
    dpt2 = DPT(in_dim=512, hidden_dim=256, out_dim=4, kernel_size = kernel_size, amp=True)
    dpt_refiner1 = nn.ModuleDict(
        {
            "16": DPTRefiner(
                2 * 512+128,
                2 * 512+128,
                3 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=2,
                displacement_emb_dim=128,
                amp = True,
                bn_momentum = 0.01,
            ),
            "8": DPTRefiner(
                2 * 512+64,
                2 * 512+64,
                3 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=2,
                displacement_emb_dim=64,
                amp = True,
                bn_momentum = 0.01,
            ),
            "4": DPTRefiner(
                2 * 256+32,
                2 * 256+32,
                3 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=2,
                displacement_emb_dim=32,
                amp = True,
                bn_momentum = 0.01,
            ),
            "2": DPTRefiner(
                2 * 64+16,
                128+16,
                3 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=2,
                displacement_emb_dim=16,
                amp = True,
                bn_momentum = 0.01,
            ),
            "1": DPTRefiner(
                2 * 9 + 6,
                24,
                3 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks = 2,
                displacement_emb_dim = 6,
                amp = True,
                bn_momentum = 0.01,
            ),
        }
    )
    dpt_refiner2 = nn.ModuleDict(
        {
            "16": DPTRefiner(
                2 * 512+128,
                2 * 512+128,
                3 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=2,
                displacement_emb_dim=128,
                amp = True,
                bn_momentum = 0.01,
            ),
            "8": DPTRefiner(
                2 * 512+64,
                2 * 512+64,
                3 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=2,
                displacement_emb_dim=64,
                amp = True,
                bn_momentum = 0.01,
            ),
            "4": DPTRefiner(
                2 * 256+32,
                2 * 256+32,
                3 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=2,
                displacement_emb_dim=32,
                amp = True,
                bn_momentum = 0.01,
            ),
            "2": DPTRefiner(
                2 * 64+16,
                128+16,
                3 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=2,
                displacement_emb_dim=16,
                amp = True,
                bn_momentum = 0.01,
            ),
            "1": DPTRefiner(
                2 * 9 + 6,
                24,
                3 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks = 2,
                displacement_emb_dim = 6,
                amp = True,
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
    proj16 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1), nn.BatchNorm2d(512))
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
    decoder = Decoder_dpt(coordinate_decoder, dpt1, dpt2, dpt_refiner1, dpt_refiner2,
                      gps, 
                      proj, 
                      conv_refiner, 
                      detach=True, 
                      scales=["16", "8", "4", "2", "1"], 
                      displacement_dropout_p = displacement_dropout_p,
                      gm_warp_dropout_p = gm_warp_dropout_p)
    
    h,w = resolutions[resolution]
    dinov2_weights = torch.load('/home/cpeng26/scratchrchella4/checkpoints/dinov2_vitl14_pretrain.pth')
    encoder = CNNandDinov2(
        cnn_kwargs = dict(
            pretrained=pretrained_backbone,
            amp = True),
        dinov2_weights= dinov2_weights,
        amp = True,
        use_vgg = True,
    )
    matcher = RegressionMatcher_dpt(encoder, decoder, h=h, w=w,**kwargs)
    weights = torch.load('roma_outdoor.pth', map_location="cuda")
    # del weights["decoder.proj.16.0.weight"]
    # del weights["decoder.proj.16.0.bias"]
    matcher.load_state_dict(weights, strict = False)
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

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    resolution = args.train_resolution
    wandb_log = not args.dont_log_wandb
    experiment_name = os.path.splitext(os.path.basename(__file__))[0]
    wandb_mode = "online" if wandb_log and rank == 0 else "disabled"
    wandb.init(project="romatch", entity=args.wandb_entity, name=experiment_name, reinit=False, mode = wandb_mode)
    checkpoint_dir = "/cis/net/r24a/data/zshao/Mast3rRoMa_checkpoints/"
    h,w = resolutions[resolution]
    model = get_SD_model(pretrained_backbone=True, resolution=resolution, attenuate_cert = False).to(device_id)
    for name,module in model.named_children():
        print(f"Submodule: {name}")
        for param_name, param in module.named_parameters():
            if param.requires_grad:
                print(f" - {param_name} is trainable")
    #model = SyncBatchNorm.convert_sync_batchnorm(model)
    # Num steps
    global_step = 0
    batch_size = args.gpu_batch_size
    step_size = gpus*batch_size
    romatch.STEP_SIZE = step_size
    
    
    dataset_path = '/cis/net/io99/data/zshao/megadepth'
    # Data
    print('Building train dataset {:s}'.format(args.train_dataset))
    data_loader_train = build_dataset(args.train_dataset, args.gpu_batch_size, args.num_workers, test=False)
    N = int(args.epochs * len(data_loader_train) * romatch.STEP_SIZE) + 1  # 250k steps of batch size 32
    depth_interpolation_mode = "bilinear"
    # Loss and optimizer
    depth_loss = RobustLosses_Depth(
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
    megadense_benchmark = MegadepthDenseBenchmark_depth(dataset_path, num_samples = 1000, h=h,w=w)
    checkpointer = CheckPoint(checkpoint_dir, experiment_name)
    model, optimizer, lr_scheduler, global_step = checkpointer.load(model, optimizer, lr_scheduler, global_step)
    romatch.GLOBAL_STEP = global_step
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters = True, gradient_as_bucket_view=True)
    grad_scaler = torch.cuda.amp.GradScaler(growth_interval=1_000_000)
    grad_clip_norm = 0.01
    for epoch in range(args.epochs):
        if hasattr(data_loader_train, 'dataset') and hasattr(data_loader_train.dataset, 'set_epoch'):
            data_loader_train.dataset.set_epoch(epoch)
        if hasattr(data_loader_train, 'sampler') and hasattr(data_loader_train.sampler, 'set_epoch'):
            data_loader_train.sampler.set_epoch(epoch)
        train_epoch_new(data_loader_train, ddp_model, depth_loss, optimizer, lr_scheduler, epoch, grad_scaler, grad_clip_norm = grad_clip_norm,
        )
        checkpointer.save(model, optimizer, lr_scheduler, romatch.GLOBAL_STEP)
        wandb.log(megadense_benchmark.benchmark(model), step = romatch.GLOBAL_STEP)

def test_mega_8_scenes(model, name):
    mega_8_scenes_benchmark = MegaDepthPoseEstimationBenchmark("/export/r24a/data/zshao/data/megadepth",
                                                scene_names=['mega_8_scenes_0019_0.1_0.3.npz',
                                                    'mega_8_scenes_0025_0.1_0.3.npz',
                                                    'mega_8_scenes_0021_0.1_0.3.npz',
                                                    'mega_8_scenes_0008_0.1_0.3.npz',
                                                    'mega_8_scenes_0032_0.1_0.3.npz',
                                                    'mega_8_scenes_1589_0.1_0.3.npz',
                                                    'mega_8_scenes_0063_0.1_0.3.npz',
                                                    'mega_8_scenes_0024_0.1_0.3.npz',
                                                    'mega_8_scenes_0019_0.3_0.5.npz',
                                                    'mega_8_scenes_0025_0.3_0.5.npz',
                                                    'mega_8_scenes_0021_0.3_0.5.npz',
                                                    'mega_8_scenes_0008_0.3_0.5.npz',
                                                    'mega_8_scenes_0032_0.3_0.5.npz',
                                                    'mega_8_scenes_1589_0.3_0.5.npz',
                                                    'mega_8_scenes_0063_0.3_0.5.npz',
                                                    'mega_8_scenes_0024_0.3_0.5.npz'])
    mega_8_scenes_results = mega_8_scenes_benchmark.benchmark(model, model_name=name)
    print(mega_8_scenes_results)
    json.dump(mega_8_scenes_results, open(f"/.results/mega_8_scenes_{name}.json", "w"))

def test_mega1500(model, name):
    mega1500_benchmark = MegaDepthPoseEstimationBenchmark("/export/r24a/data/zshao/data/megadepth")
    mega1500_results = mega1500_benchmark.benchmark(model, model_name=name)
    json.dump(mega1500_results, open(f"/.results/mega1500_{name}.json", "w"))

def test_mega_dense(model, name):
    megadense_benchmark = MegadepthDenseBenchmark("/export/r24a/data/zshao/data/megadepth", num_samples = 1000)
    megadense_results = megadense_benchmark.benchmark(model)
    json.dump(megadense_results, open(f"./results/mega_dense_{name}.json", "w"))
    
def test_hpatches(model, name):
    hpatches_benchmark = HpatchesHomogBenchmark("/export/r24a/data/zshao/data/hpatches")
    hpatches_results = hpatches_benchmark.benchmark(model)
    json.dump(hpatches_results, open(f"./results/hpatches_{name}.json", "w"))


if __name__ == "__main__":
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1" # For BF16 computations
    os.environ["OMP_NUM_THREADS"] = "16"
    #os.environ['RANK'] = '0'  # Set the process rank (0 for single-node setup)
    #os.environ['WORLD_SIZE'] = '1'  # Set the total number of processes
    #os.environ['MASTER_ADDR'] = 'localhost'  # Use localhost for single-node
    #os.environ['MASTER_PORT'] = '12355'  # Any open port can be used
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    import romatch
    parser = ArgumentParser()
    parser.add_argument("--only_test", action='store_true')
    parser.add_argument("--debug_mode", action='store_true')
    parser.add_argument("--dont_log_wandb", action='store_true')
    parser.add_argument("--train_resolution", default='medium')
    parser.add_argument("--gpu_batch_size", default=4, type=int)
    parser.add_argument("--wandb_entity", required = False)
    parser.add_argument('--train_dataset', required=True, type=str, help="training set")
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--epochs', default=50, type=int)

    args, _ = parser.parse_known_args()
    romatch.DEBUG_MODE = args.debug_mode
    if not args.only_test:
        train(args)
