#!/bin/bash
#SBATCH --job-name=trainwarp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8             # 每节点 8 个进程，对应 8 张 GPU
#SBATCH --cpus-per-task=8               # 每个 GPU 分配的 CPU 数量
#SBATCH --gres=gpu:4                    # 每节点使用 4 张 GPU
#SBATCH --time=24:00:00
#SBATCH --partition=a100      
#SBATCH --output=/home/cpeng26/scratchrchella4/logs/%x_%j.out
#SBATCH --error=/home/cpeng26/scratchrchella4/logs/%x_%j.err

source ~/.bashrc
conda activate mast3r

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500
NODE_RANK=$SLURM_NODEID
NNODES=$SLURM_JOB_NUM_NODES

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=4 \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train_warp.py \
    --train_dataset "120_000 @ MegaDepth_all(split='train', ROOT='/home/cpeng26/scratchrchella4/data/megadepth', min_overlap=0.01, resolution=512, aug_crop=16, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 80_000 @ MegaDepth_all(split='train', ROOT='/home/cpeng26/scratchrchella4/data/megadepth', min_overlap=0.35, resolution=512, aug_crop=16, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5)" \
    --test_dataset "1000 @ MegaDepth_all(split='val',  ROOT='/home/cpeng26/scratchrchella4/data/megadepth', min_overlap=0.3, max_overlap=0.7, resolution=512, seed=777, n_corres=1024)" \
    --model "AsymmetricMASt3R_warp(freeze='backbone', pos_embed='RoPE100', patch_embed_cls='PatchEmbedDust3R_cnn', cnn_type='vgg', img_size=(512, 512), head_type='warp+dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='?avg_dis', loss_in_log=False), alpha=0.2)" \
    --test_criterion "Regr3D(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0)" \
    --pretrained "/home/cpeng26/scratchrchella4/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs 4 --epochs 20 --batch_size 4 --accum_iter 2 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 --print_freq=10 --disable_cudnn_benchmark \
    --output_dir "/home/cpeng26/scratchrchella4/checkpoints/MASt3R_freeze_trainwarp_onlymega512"
