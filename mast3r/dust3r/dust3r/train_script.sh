torchrun --nproc_per_node=2 train.py \
    --train_dataset "1000 @ Co3d(split='train', ROOT='/export/r24a/data/zshao/data/co3d_subset_processed', aug_crop='auto', aug_monocular=0.005, aug_rot90='diff', mask_bg='rand', resolution=224, n_corres=8192, nneg=0.5, transform=ColorJitter)" \
    --test_dataset "100 @ Co3d(split='test', ROOT='/export/r24a/data/zshao/data/co3d_subset_processed', resolution=224, n_corres=1024, seed=777)" \
    --model "AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True)" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2) + 0.075*ConfMatchingLoss(MatchingLoss(InfoNCE(mode='proper', temperature=0.05), negatives_padding=0, blocksize=8192), alpha=10.0, confmode='mean')" \
    --test_criterion "Regr3D_ScaleShiftInv(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0) + -1.*MatchingLoss(APLoss(nq='torch', fp=torch.float16), negatives_padding=12288)" \
    --pretrained "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs 1 --epochs 10 --batch_size 2 --accum_iter 1 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 --disable_cudnn_benchmark \
    --output_dir "checkpoints/mast3r_demo"


# MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric - train mast3r with metric regression and matching loss
# we used cosxl to generate variations of DL3DV: "foggy", "night", "rainy", "snow", "sunny" but we were not convinced by it.


torchrun --nproc_per_node 4 train.py \
    --train_dataset " + 100_000 @ Habitat(800_000, split='train', ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ BlendedMVS(split='train', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ MegaDepth(split='train', ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ ARKitScenes(ROOT='/home/cpeng26/scratchrchella4/data/arkitscenes_processed', aug_crop=256, resolution=224, transform=ColorJitter) + 100_000 @ Co3d(split='train',ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter) + 100_000 @ StaticThings3D('/home/cpeng26/scratchrchella4/data/static_3d_dataset_processed', aug_crop=256, mask_bg='rand', resolution=224, transform=ColorJitter) + 100_000 @ ScanNetpp(split='train',ROOT='/home/cpeng26/scratchrchella4/data/scannetpp_processed', aug_crop=256, resolution=224, transform=ColorJitter)" \
    --test_dataset " Habitat(1_000, ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', split='val', resolution=224, seed=777) + 1_000 @ BlendedMVS(split='val', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', resolution=224, seed=777) + 1_000 @ MegaDepth(split='val', ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', resolution=224, seed=777) + 1_000 @ Co3d(split='test', ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed', mask_bg='rand', resolution=224, seed=777) " \
    --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --model="AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --pretrained="checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth" \
    --lr=0.0001 --min_lr=1e-06 --warmup_epochs=10 --epochs=100 --batch_size=16 --accum_iter=1 \
    --save_freq=5 --keep_freq=10 --eval_freq=1 \
    --output_dir "/home/cpeng26/scratchrchella4/checkpoints/dinov2_decoder_dpt_224"	

torchrun --nproc_per_node 4 train.py \
    --train_dataset " + 100_000 @ Habitat(800_000, split='train', ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ BlendedMVS(split='train', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ MegaDepth(split='train', ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ ARKitScenes(split='train', ROOT='/home/cpeng26/scratchrchella4/data/arkitscenes_processed', aug_crop=256, resolution=224, transform=ColorJitter) + 100_000 @ Co3d(split='train',ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter) + 100_000 @ StaticThings3D('/home/cpeng26/scratchrchella4/data/static_3d_dataset_processed', aug_crop=256, mask_bg='rand', resolution=224, transform=ColorJitter) + 100_000 @ ScanNetpp(split='train',ROOT='/home/cpeng26/scratchrchella4/data/scannetpp_processed', aug_crop=256, resolution=224, transform=ColorJitter)" \
    --test_dataset " Habitat(1_000, ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', split='val', resolution=224, seed=777) + 1_000 @ BlendedMVS(split='val', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', resolution=224, seed=777) + 1_000 @ MegaDepth(split='val', ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', resolution=224, seed=777) + 1_000 @ Co3d(split='test', ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed', mask_bg='rand', resolution=224, seed=777)" \
    --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --model="AsymmetricCroCo3DStereo_DINOv2(pos_embed='RoPE100', img_size=(224, 224), patch_size=14, head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), freeze='none', enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --pretrained="/home/cpeng26/scratchrchella4/checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth" \
    --lr=0.0001 --min_lr=1e-06 --warmup_epochs=10 --epochs=100 --batch_size=60 --accum_iter=1 \
    --save_freq=5 --keep_freq=10 --eval_freq=1 --num_workers=16\
    --output_dir "/home/cpeng26/scratchrchella4/checkpoints/dinov2train_decoder_224"	

torchrun --nproc_per_node 4 train.py \
    --train_dataset " + 100_000 @ Habitat(800_000, split='train', ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ BlendedMVS(split='train', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ MegaDepth(split='train', ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ ARKitScenes(split='train', ROOT='/home/cpeng26/scratchrchella4/data/arkitscenes_processed', aug_crop=256, resolution=224, transform=ColorJitter) + 100_000 @ Co3d(split='train',ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter) + 100_000 @ StaticThings3D('/home/cpeng26/scratchrchella4/data/static_3d_dataset_processed', aug_crop=256, mask_bg='rand', resolution=224, transform=ColorJitter) + 100_000 @ ScanNetpp(split='train',ROOT='/home/cpeng26/scratchrchella4/data/scannetpp_processed', aug_crop=256, resolution=224, transform=ColorJitter)" \
    --test_dataset " Habitat(1_000, ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', split='val', resolution=224, seed=777) + 1_000 @ BlendedMVS(split='val', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', resolution=224, seed=777) + 1_000 @ MegaDepth(split='val', ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', resolution=224, seed=777) + 1_000 @ Co3d(split='test', ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed', mask_bg='rand', resolution=224, seed=777)" \
    --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --model="AsymmetricCroCo3DStereo_DINOv2(pos_embed='RoPE100',img_size=(224, 224), patch_size=14, head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=1024, dec_depth=24, dec_num_heads=16)" \
    --lr=0.0001 --min_lr=1e-06 --warmup_epochs=10 --epochs=100 --batch_size=16 --accum_iter=1 \
    --save_freq=5 --keep_freq=10 --eval_freq=1 --num_workers=16 \
    --output_dir "/home/cpeng26/scratchrchella4/checkpoints/dinov2_vitl_224"


torchrun --nproc_per_node 4 train.py \
    --train_dataset " + 100_000 @ Habitat(800_000, split='train', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/habitat_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ BlendedMVS(split='train', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/blendedmvs_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ MegaDepth(split='train', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/megadepth_dataset_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ ARKitScenes(split='train', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/arkitscenes_processed', aug_crop=256, resolution=224, transform=ColorJitter) + 100_000 @ Co3d(split='train',ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/co3d_subset_processed', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter) + 100_000 @ StaticThings3D('/cis/net/r24a/data/zshao/data/dust3r/data/static_3d_dataset_processed', aug_crop=256, mask_bg='rand', resolution=224, transform=ColorJitter) + 100_000 @ ScanNetpp(split='train',ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/scannetpp_processed', aug_crop=256, resolution=224, transform=ColorJitter)" \
    --test_dataset "Habitat(80_000, ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/habitat_processed', split='val', resolution=224, seed=777) + 1_000 @ BlendedMVS(split='val', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/blendedmvs_processed', resolution=224, seed=777) + 1_000 @ MegaDepth(split='val', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/megadepth_dataset_processed', resolution=224, seed=777) + 1_000 @ Co3d(split='test', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/co3d_subset_processed', mask_bg='rand', resolution=224, seed=777)" \
    --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --model="AsymmetricCroCo3DStereo_DINOv2(pos_embed='RoPE100', img_size=(224, 224), patch_size=14, head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --pretrained="checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth" \
    --lr=0.0001 --min_lr=1e-06 --warmup_epochs=10 --epochs=100 --batch_size=16 --accum_iter=1 \
    --save_freq=5 --keep_freq=10 --eval_freq=1  --num_workers=16\
    --output_dir "/cis/net/r24a/data/zshao/checkpoints/dust3r/dinov2_decoder_dpt_224"	

torchrun --nproc_per_node=8 train.py \
    --train_dataset "57_000 @ Habitat512(1_000_000, split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 68_400 @ BlendedMVS(split='train', mask_sky=True, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 68_400 @ MegaDepth(split='train', mask_sky=True, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 45_600 @ ARKitScenes(split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 22_800 @ Co3d(split='train', mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 22_800 @ StaticThings3D(mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 45_600 @ ScanNetpp(split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 45_600 @ TartanAir(pairs_subset='', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 4_560 @ UnrealStereo4K(resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 1_140 @ VirtualKitti(optical_center_is_centered=True, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 22_800 @ WildRgbd(split='train', mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 145_920 @ NianticMapFree(split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 57_000 @ DL3DV(split='nlight', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 57_000 @ DL3DV(split='not-nlight', cosxl_augmentations=None, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 34_200 @ InternalUnreleasedDataset(resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5)" \
    --test_dataset "Habitat512(1_000, split='val', resolution=(512,384), seed=777, n_corres=1024) + 1_000 @ BlendedMVS(split='val', resolution=(512,384), mask_sky=True, seed=777, n_corres=1024) + 1_000 @ ARKitScenes(split='test', resolution=(512,384), seed=777, n_corres=1024) + 1_000 @ MegaDepth(split='val', mask_sky=True, resolution=(512,336), seed=777, n_corres=1024) + 1_000 @ Co3d(split='test', resolution=(512,384), mask_bg='rand', seed=777, n_corres=1024)" \
    --model "AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, inf))" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2, loss_in_log=False) + 0.075*ConfMatchingLoss(MatchingLoss(InfoNCE(mode='proper', temperature=0.05), negatives_padding=0, blocksize=8192), alpha=10.0, confmode='mean')" \
    --test_criterion "Regr3D(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0) + -1.*MatchingLoss(APLoss(nq='torch', fp=torch.float16), negatives_padding=12288)" \
    --pretrained "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs 8 --epochs 50 --batch_size 4 --accum_iter 2 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 --print_freq=10 --disable_cudnn_benchmark \
    --output_dir "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"



'/export/r24a/data/zshao/data/co3d_subset_processed'
"57_000 @ Habitat512(1_000_000, split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 
68_400 @ BlendedMVS(split='train', mask_sky=True, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 
68_400 @ MegaDepth(split='train', mask_sky=True, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 
45_600 @ ARKitScenes(split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 
22_800 @ Co3d(split='train', mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 
22_800 @ StaticThings3D(mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 
45_600 @ ScanNetpp(split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 
45_600 @ TartanAir(pairs_subset='', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 
4_560 @ UnrealStereo4K(resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 
1_140 @ VirtualKitti(optical_center_is_centered=True, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 
22_800 @ WildRgbd(split='train', mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 
145_920 @ NianticMapFree(split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 
57_000 @ DL3DV(split='nlight', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 
57_000 @ DL3DV(split='not-nlight', cosxl_augmentations=None, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 
34_200 @ InternalUnreleasedDataset(resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5)"

"57_000 @ Habitat2(800_000, split='train', ROOT='/cis/net/io96/data/zshao/dust3r/data/megadepth_dataset_processed', resolution=560) + 68_400 @ BlendedMVS2(split='train', ROOT='/cis/net/io96/data/zshao/dust3r/data/blendedmvs_processed', resolution=560) + 68_400 @ MegaDepth(split='train',ROOT='/cis/net/io96/data/zshao/dust3r/data/megadepth_dataset_processed', resolution=560) + 
45_600 @ ARKitScenes2(split='train', ROOT='/cis/net/io96/data/zshao/dust3r/data/arkitscenes_processed', resolution=560) + 22_800 @ Co3d2(split='train',mask_bg='rand',ROOT='/cis/net/io96/data/zshao/dust3r/data/co3d_processed', resolution=560) + 
22_800 @ StaticThings3D2(ROOT='/cis/net/io96/data/zshao/dust3r/data/static_3d_dataset_processed', mask_bg='rand', resolution=560) + 45_600 @ ScanNetpp2(split='train', ROOT='/cis/net/io96/data/zshao/dust3r/data/scannetpp_processed', resolution=560)
22_800 @ WildRgbd2(split='train', ROOT='/cis/net/io96/data/zshao/dust3r/data/wildrgbd_processed', mask_bg='rand', resolution=560)

AsymmetricMASt3R(
  (patch_embed): PatchEmbedDust3R(
    (proj): Conv2d(3, 1024, kernel_size=(16, 16), stride=(16, 16))
    (norm): Identity()
  )
  (mask_generator): RandomMask()
  (rope): cuRoPE2D()
  (enc_blocks): ModuleList(
    (0-23): 24 x Block(
      (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=1024, out_features=3072, bias=True)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=1024, out_features=1024, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
        (rope): cuRoPE2D()
      )
      (drop_path): Identity()
      (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=1024, out_features=4096, bias=True)
        (act): GELU(approximate='none')
        (drop1): Dropout(p=0.0, inplace=False)
        (fc2): Linear(in_features=4096, out_features=1024, bias=True)
        (drop2): Dropout(p=0.0, inplace=False)
      )
    )
  )
  (enc_norm): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
  (decoder_embed): Linear(in_features=1024, out_features=768, bias=True)
  (dec_blocks): ModuleList(
    (0-11): 12 x DecoderBlock(
      (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=768, out_features=2304, bias=True)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=768, out_features=768, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
        (rope): cuRoPE2D()
      )
      (cross_attn): CrossAttention(
        (projq): Linear(in_features=768, out_features=768, bias=True)
        (projk): Linear(in_features=768, out_features=768, bias=True)
        (projv): Linear(in_features=768, out_features=768, bias=True)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=768, out_features=768, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
        (rope): cuRoPE2D()
      )
      (drop_path): Identity()
      (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (norm3): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (drop1): Dropout(p=0.0, inplace=False)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (drop2): Dropout(p=0.0, inplace=False)
      )
      (norm_y): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    )
  )
  (dec_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
  (dec_blocks2): ModuleList(
    (0-11): 12 x DecoderBlock(
      (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=768, out_features=2304, bias=True)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=768, out_features=768, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
        (rope): cuRoPE2D()
      )
      (cross_attn): CrossAttention(
        (projq): Linear(in_features=768, out_features=768, bias=True)
        (projk): Linear(in_features=768, out_features=768, bias=True)
        (projv): Linear(in_features=768, out_features=768, bias=True)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=768, out_features=768, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
        (rope): cuRoPE2D()
      )
      (drop_path): Identity()
      (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (norm3): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (drop1): Dropout(p=0.0, inplace=False)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (drop2): Dropout(p=0.0, inplace=False)
      )
      (norm_y): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    )
  )
  (downstream_head1): Cat_MLP_LocalFeatures_DPT_Pts3d(
    (dpt): DPTOutputAdapter_fix(
      (scratch): Module(
        (layer1_rn): Conv2d(96, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (layer2_rn): Conv2d(192, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (layer3_rn): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (layer4_rn): Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (layer_rn): ModuleList(
          (0): Conv2d(96, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): Conv2d(192, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (2): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (3): Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        )
        (refinenet1): FeatureFusionBlock_custom(
          (out_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (resConfUnit1): ResidualConvUnit_custom(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (activation): ReLU()
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (resConfUnit2): ResidualConvUnit_custom(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (activation): ReLU()
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (refinenet2): FeatureFusionBlock_custom(
          (out_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (resConfUnit1): ResidualConvUnit_custom(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (activation): ReLU()
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (resConfUnit2): ResidualConvUnit_custom(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (activation): ReLU()
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (refinenet3): FeatureFusionBlock_custom(
          (out_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (resConfUnit1): ResidualConvUnit_custom(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (activation): ReLU()
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (resConfUnit2): ResidualConvUnit_custom(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (activation): ReLU()
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (refinenet4): FeatureFusionBlock_custom(
          (out_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (resConfUnit1): ResidualConvUnit_custom(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (activation): ReLU()
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (resConfUnit2): ResidualConvUnit_custom(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (activation): ReLU()
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (head): Sequential(
        (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): Interpolate()
        (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace=True)
        (4): Conv2d(128, 4, kernel_size=(1, 1), stride=(1, 1))
      )
      (act_postprocess): ModuleList(
        (0): Sequential(
          (0): Conv2d(1024, 96, kernel_size=(1, 1), stride=(1, 1))
          (1): ConvTranspose2d(96, 96, kernel_size=(4, 4), stride=(4, 4))
        )
        (1): Sequential(
          (0): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1))
          (1): ConvTranspose2d(192, 192, kernel_size=(2, 2), stride=(2, 2))
        )
        (2): Sequential(
          (0): Conv2d(768, 384, kernel_size=(1, 1), stride=(1, 1))
        )
        (3): Sequential(
          (0): Conv2d(768, 768, kernel_size=(1, 1), stride=(1, 1))
          (1): Conv2d(768, 768, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
      )
    )
    (head_local_features): Mlp(
      (fc1): Linear(in_features=1792, out_features=7168, bias=True)
      (act): GELU(approximate='none')
      (drop1): Dropout(p=0.0, inplace=False)
      (fc2): Linear(in_features=7168, out_features=6400, bias=True)
      (drop2): Dropout(p=0.0, inplace=False)
    )
  )
  (downstream_head2): Cat_MLP_LocalFeatures_DPT_Pts3d(
    (dpt): DPTOutputAdapter_fix(
      (scratch): Module(
        (layer1_rn): Conv2d(96, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (layer2_rn): Conv2d(192, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (layer3_rn): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (layer4_rn): Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (layer_rn): ModuleList(
          (0): Conv2d(96, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): Conv2d(192, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (2): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (3): Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        )
        (refinenet1): FeatureFusionBlock_custom(
          (out_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (resConfUnit1): ResidualConvUnit_custom(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (activation): ReLU()
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (resConfUnit2): ResidualConvUnit_custom(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (activation): ReLU()
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (refinenet2): FeatureFusionBlock_custom(
          (out_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (resConfUnit1): ResidualConvUnit_custom(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (activation): ReLU()
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (resConfUnit2): ResidualConvUnit_custom(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (activation): ReLU()
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (refinenet3): FeatureFusionBlock_custom(
          (out_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (resConfUnit1): ResidualConvUnit_custom(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (activation): ReLU()
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (resConfUnit2): ResidualConvUnit_custom(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (activation): ReLU()
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (refinenet4): FeatureFusionBlock_custom(
          (out_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (resConfUnit1): ResidualConvUnit_custom(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (activation): ReLU()
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (resConfUnit2): ResidualConvUnit_custom(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (activation): ReLU()
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
      )
      (head): Sequential(
        (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): Interpolate()
        (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace=True)
        (4): Conv2d(128, 4, kernel_size=(1, 1), stride=(1, 1))
      )
      (act_postprocess): ModuleList(
        (0): Sequential(
          (0): Conv2d(1024, 96, kernel_size=(1, 1), stride=(1, 1))
          (1): ConvTranspose2d(96, 96, kernel_size=(4, 4), stride=(4, 4))
        )
        (1): Sequential(
          (0): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1))
          (1): ConvTranspose2d(192, 192, kernel_size=(2, 2), stride=(2, 2))
        )
        (2): Sequential(
          (0): Conv2d(768, 384, kernel_size=(1, 1), stride=(1, 1))
        )
        (3): Sequential(
          (0): Conv2d(768, 768, kernel_size=(1, 1), stride=(1, 1))
          (1): Conv2d(768, 768, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
      )
    )
    (head_local_features): Mlp(
      (fc1): Linear(in_features=1792, out_features=7168, bias=True)
      (act): GELU(approximate='none')
      (drop1): Dropout(p=0.0, inplace=False)
      (fc2): Linear(in_features=7168, out_features=6400, bias=True)
      (drop2): Dropout(p=0.0, inplace=False)
    )
  )
)

--train_dataset " + 100_000 @ Habitat(800_000, split='train', ROOT='/cis/home/cpeng/dust3r/data/habitat_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ BlendedMVS(split='train', ROOT='/cis/home/cpeng/dust3r/data/blendedmvs_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ MegaDepth(split='train', ROOT='/cis/home/cpeng/dust3r/data/megadepth_dataset_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ ARKitScenes(split='train', ROOT='/cis/home/cpeng/dust3r/data/arkitscenes_processed', aug_crop=256, resolution=224, transform=ColorJitter) + 100_000 @ Co3d(split='train',ROOT='/cis/home/cpeng/dust3r/data/co3d_subset_processed', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter) + 100_000 @ StaticThings3D('/cis/home/cpeng/dust3r/data/static_3d_dataset_processed', aug_crop=256, mask_bg='rand', resolution=224, transform=ColorJitter) + 100_000 @ ScanNetpp(split='train',ROOT='/cis/home/cpeng/dust3r/data/scannetpp_processed', aug_crop=256, resolution=224, transform=ColorJitter)" \
--test_dataset "Habitat(80_000, ROOT='/cis/home/cpeng/dust3r/data/habitat_processed', split='val', resolution=224, seed=777) + 1_000 @ BlendedMVS(split='val', ROOT='/cis/home/cpeng/dust3r/data/blendedmvs_processed', resolution=224, seed=777) + 1_000 @ MegaDepth(split='val', ROOT='/cis/home/cpeng/dust3r/data/megadepth_dataset_processed', resolution=224, seed=777) + 1_000 @ Co3d(split='test', ROOT='/cis/home/cpeng/dust3r/data/co3d_subset_processed', mask_bg='rand', resolution=224, seed=777)" \
    