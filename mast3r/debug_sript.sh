python -u train.py 
--train_dataset= + 10_000 @ Habitat(800_000, split='train', ROOT='data/habitat_processed', aug_crop=16, 
resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 
10_000 @ BlendedMVS(split='train', ROOT='data/blendedmvs_processed', aug_crop=16, 
resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 
20_000 @ MegaDepth(split='train', ROOT='data/megadepth_processed', aug_crop=16, 
resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 
10_000 @ ARKitScenes(split='train', aug_crop=256, ROOT='data/arkitscenes_processed', 
resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 
10_000 @ Co3d(split='train', ROOT='data/co3d_processed', aug_crop=16, mask_bg='rand', 
resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 
10_000 @ StaticThings3D(ROOT='data/static_3d_dataset_processed', aug_crop=256, 
mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 
10_000 @ ScanNetpp(split='train', ROOT='data/scannetpp_processed', 
aug_crop=256, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 
10_000 @ MegaDepth(split='train', ROOT='/cis/home/cpeng/dust3r/data/megadepth_dataset_processed', 
aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 
10_000 @ Waymo(split='train', ROOT='data/waymo_training_dataset_processed', 
aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) 
--test_dataset=Habitat(80_000, split='val', resolution=(512,384), seed=777, ROOT='data/habitat_processed') +
 1_000 @ BlendedMVS(split='val', ROOT='data/blendedmvs_processed', resolution=(512,384), seed=777) + 
1000 @ MegaDepth(split='val', ROOT='data/megadepth_processed', resolution=(512,336), seed=777) + 
1_000 @ Co3d(split='test', ROOT='data/co3d_processed', resolution=(512,384), seed=777)  
--train_criterion=ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2) 
--test_criterion=Regr3D_ScaleShiftInv(L21, gt_scale=True) --model=AsymmetricCroCo3DStereo(pos_embed='RoPE100', 
patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), 
conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12) 
--pretrained=checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth 
--lr=0.0001 --min_lr=1e-06 --warmup_epochs=15 --epochs=90 --batch_size=2 --accum_iter=2 --save_freq=5 
--keep_freq=10 --eval_freq=1 --print_freq=10 --disable_cudnn_benchmark --output_dir=checkpoints/dust3r_512dpt_fromlinear


torchrun --nproc_per_node=1 train.py \
    --train_dataset "57_000 @ Habitat(800_000, split='train',ROOT='/cis/net/io99/data/zshao/dust3r/data/habitat_processed', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=16, aug_monocular=0.005, n_corres=8192, nneg=0.5, transform=ColorJitter) + 68_400 @ BlendedMVS(split='train', ROOT='/cis/net/io99/data/zshao/dust3r/data/blendedmvs_processed', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop = 16, aug_monocular=0.005, n_corres=8192, nneg=0.5, transform=ColorJitter) + 68_400 @ MegaDepth(split='train', ROOT='/cis/net/io99/data/zshao/dust3r/data/megadepth_dataset_processed', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=16, aug_monocular=0.005, n_corres=8192, nneg=0.5, transform=ColorJitter) + 45_600 @ ARKitScenes(split='train',ROOT='/cis/net/io99/data/zshao/dust3r/data/arkitscenes_processed', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=16, aug_monocular=0.005, n_corres=8192, nneg=0.5, transform=ColorJitter) + 22_800 @ Co3d(split='train', ROOT='/cis/net/io99/data/zshao/dust3r/data/co3d_processed', mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=16, aug_monocular=0.005, n_corres=8192, nneg=0.5, transform=ColorJitter) + 22_800 @ StaticThings3D(ROOT='/cis/net/io99/data/zshao/dust3r/data/static_3d_dataset_processed', mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=16, aug_monocular=0.005, n_corres=8192, nneg=0.5, transform=ColorJitter) + 45_600 @ ScanNetpp(split='train', ROOT='/cis/net/io99/data/zshao/dust3r/data/scannetpp_processed', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=16, aug_monocular=0.005, n_corres=8192, nneg=0.5, transform=ColorJitter) + 22_800 @ WildRGBD(split='train', ROOT='/cis/net/io99/data/zshao/dust3r/data/wildrgb_processed', mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=16, aug_monocular=0.005, n_corres=8192, nneg=0.5, transform=ColorJitter)" \
    --test_dataset "Habitat(80_000, split='val', ROOT='/cis/net/io99/data/zshao/dust3r/data/habitat_processed', resolution=(512,384), seed=777, n_corres=1024) + 1_000 @ BlendedMVS(split='val', ROOT='/cis/net/io99/data/zshao/dust3r/data/blendedmvs_processed', resolution=(512,384), seed=777, n_corres=1024) + 1_000 @ MegaDepth(split='val', ROOT='/cis/net/io99/data/zshao/dust3r/data/megadepth_dataset_processed', resolution=(512,336), seed=777, n_corres=1024) + 1_000 @ Co3d(split='test', ROOT='/cis/net/io99/data/zshao/dust3r/data/co3d_processed', resolution=(512,384), mask_bg='rand', seed=777, n_corres=1024)" \
    --model "AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, inf))" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2) + 0.075*ConfMatchingLoss(MatchingLoss(InfoNCE(mode='proper', temperature=0.05), negatives_padding=0, blocksize=8192), alpha=10.0, confmode='mean')" \
    --test_criterion "Regr3D(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0) + -1.*MatchingLoss(APLoss(nq='torch', fp=torch.float16), negatives_padding=12288)" \
    --pretrained "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs 8 --epochs 50 --batch_size 4 --accum_iter 2 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 --print_freq=10 --disable_cudnn_benchmark \
    --output_dir "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"

torchrun --nproc_per_node=1 train_roma_dpt_loader.py --train_dataset "10_000 @ Habitat2(800_000, split='train', ROOT='/cis/net/io99/data/zshao/dust3r/data/habitat_processed', resolution=560) + 10_000 @ BlendedMVS2(split='train', ROOT='/cis/net/io99/data/zshao/dust3r/data/blendedmvs_processed', resolution=560) + 10_000 @ MegaDepth2(split='train',ROOT='/cis/net/io99/data/zshao/dust3r/data/megadepth_dataset_processed', resolution=560) + 10_000 @ ARKitScenes2(split='train', ROOT='/cis/net/io99/data/zshao/dust3r/data/arkitscenes_processed', resolution=560) + 10_000 @ Co3d2(split='train',mask_bg='rand',ROOT='/cis/net/io99/data/zshao/dust3r/data/co3d_processed', resolution=560) + 10_000 @ StaticThings3D2(ROOT='/cis/net/io99/data/zshao/dust3r/data/static_3d_dataset_processed', mask_bg='rand', resolution=560) + 10_000 @ ScanNetpp2(split='train', ROOT='/cis/net/io99/data/zshao/dust3r/data/scannetpp_processed', resolution=560)"  --gpu_batch_size 2


dust3r stage 3
10_000 @ Habitat(1_000_000, split='train', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 
10_000 @ BlendedMVS(split='train', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 
10_000 @ MegaDepth(split='train', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 
10_000 @ ARKitScenes(aug_crop=256, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 
10_000 @ Co3d(split='train', aug_crop=16, mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 
10_000 @ StaticThings3D(aug_crop=256, mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 
10_000 @ ScanNetpp(split='train', aug_crop=256, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 
10_000 @ InternalUnreleasedDataset(aug_crop=128, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) 