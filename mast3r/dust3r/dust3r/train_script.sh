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

io104
torchrun --nproc_per_node 4 train.py \
    --train_dataset " + 100_000 @ Habitat(800_000, split='train', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/habitat_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 
    100_000 @ BlendedMVS(split='train', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/blendedmvs_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 
    100_000 @ MegaDepth(split='train', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/megadepth_dataset_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 
    100_000 @ ARKitScenes(split='train', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/arkitscenes_processed', aug_crop=256, resolution=224, transform=ColorJitter) + 
    100_000 @ Co3d(split='train',ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/co3d_subset_processed', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter) + 
    100_000 @ StaticThings3D(ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/static_3d_dataset_processed', aug_crop=256, mask_bg='rand', resolution=224, transform=ColorJitter) + 
    100_000 @ ScanNetpp(split='train',ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/scannetpp_processed', aug_crop=256, resolution=224, transform=ColorJitter)" \
    --test_dataset "Habitat(80_000, ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/habitat_processed', split='val', resolution=224, seed=777) + 1_000 @ BlendedMVS(split='val', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/blendedmvs_processed', resolution=224, seed=777) + 1_000 @ MegaDepth(split='val', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/megadepth_dataset_processed', resolution=224, seed=777) + 1_000 @ Co3d(split='test', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/co3d_subset_processed', mask_bg='rand', resolution=224, seed=777)" \
    --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --model="AsymmetricCroCo3DStereo_DINOv2(pos_embed='RoPE100', img_size=(224, 224), patch_size=14, head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --pretrained="checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth" \
    --lr=0.0001 --min_lr=1e-06 --warmup_epochs=10 --epochs=100 --batch_size=64 --accum_iter=1 \
    --save_freq=5 --keep_freq=10 --eval_freq=1  --num_workers=16 \
    --output_dir="/cis/net/r24a/data/zshao/checkpoints/dust3r/dinov2_decoder_dpt_224"	

torchrun --nproc_per_node 4 train.py \
    --train_dataset " + 10_000 @ Habitat(800_000, split='train', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/habitat_processed', aug_crop=16, resolution=[(518, 378), (518, 336), (518, 294), (518, 252), (518, 168)], transform=ColorJitter) + 10_000 @ BlendedMVS(split='train', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/blendedmvs_processed', aug_crop=16, resolution=[(518, 378), (518, 336), (518, 294), (518, 252), (518, 168)], transform=ColorJitter) + 10_000 @ MegaDepth(split='train', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/megadepth_dataset_processed', aug_crop=16, resolution=[(518, 378), (518, 336), (518, 294), (518, 252), (518, 168)], transform=ColorJitter) + 10_000 @ ARKitScenes(split='train', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/arkitscenes_processed', aug_crop=256, resolution=[(518, 378), (518, 336), (518, 294), (518, 252), (518, 168)], transform=ColorJitter) + 10_000 @ Co3d(split='train', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/co3d_subset_processed', aug_crop=16, mask_bg='rand', resolution=[(518, 378), (518, 336), (518, 294), (518, 252), (518, 168)], transform=ColorJitter) + 10_000 @ StaticThings3D(ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/static_3d_dataset_processed', aug_crop=256, mask_bg='rand', resolution=[(518, 378), (518, 336), (518, 294), (518, 252), (518, 168)], transform=ColorJitter) + 10_000 @ ScanNetpp(split='train', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/scannetpp_processed', aug_crop=256, resolution=[(518, 378), (518, 336), (518, 294), (518, 252), (518, 168)], transform=ColorJitter)" \
    --test_dataset "1_000 @ BlendedMVS(split='val', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/blendedmvs_processed', resolution=(518,378), seed=777) + 1_000 @ MegaDepth(split='val', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/megadepth_dataset_processed', resolution=(518,336), seed=777) + 1_000 @ Co3d(split='test', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/co3d_subset_processed', resolution=(518,378), seed=777)" \
    --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --model="AsymmetricCroCo3DStereo_DINOv2(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(518, 518), patch_size=14, head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --pretrained="/cis/net/r24a/data/zshao/checkpoints/dust3r/dinov2_decoder_dpt_224/checkpoint-best.pth" \
    --lr=0.0001 --min_lr=1e-06 --warmup_epochs=20 --epochs=100 --batch_size=8 --accum_iter=2 \
    --save_freq=10 --keep_freq=10 --eval_freq=1 --print_freq=10 --num_workers=16 \
    --output_dir="/cis/net/r24a/data/zshao/checkpoints/dust3r/dinov2_decoder_dpt_518"
    
torchrun --nproc_per_node 4 train.py \
    --train_dataset=" + 10_000 @ Habitat(1_000_000, split='train', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ BlendedMVS(split='train', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ MegaDepth(split='train', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ ARKitScenes(aug_crop=256, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ Co3d(split='train', aug_crop=16, mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ StaticThings3D(aug_crop=256, mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ ScanNetpp(split='train', aug_crop=256, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ InternalUnreleasedDataset(aug_crop=128, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) " \
    --test_dataset=" Habitat(1_000, split='val', resolution=(512,384), seed=777) + 1_000 @ BlendedMVS(split='val', resolution=(512,384), seed=777) + 1_000 @ MegaDepth(split='val', resolution=(512,336), seed=777) + 1_000 @ Co3d(split='test', resolution=(512,384), seed=777) " \
    --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --model="AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --pretrained="checkpoints/dust3r_512/checkpoint-best.pth" \
    --lr=0.0001 --min_lr=1e-06 --warmup_epochs=15 --epochs=90 --batch_size=4 --accum_iter=2 \
    --save_freq=5 --keep_freq=10 --eval_freq=1 --print_freq=10 --disable_cudnn_benchmark \
    --output_dir="checkpoints/dust3r_512dpt"
    
torchrun --nproc_per_node 4 train.py \
    --train_dataset=" + 10_000 @ Habitat(800_000, split='train', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/habitat_processed', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ BlendedMVS(split='train', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/blendedmvs_processed', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ MegaDepth(split='train',  ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/megadepth_dataset_processed', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ ARKitScenes(split='train', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/arkitscenes_processed', aug_crop=256, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ Co3d(split='train', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/co3d_subset_processed', aug_crop=16, mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ StaticThings3D('/cis/net/r24a/data/zshao/data/dust3r/data/static_3d_dataset_processed', aug_crop=256, mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ ScanNetpp(split='train', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/scannetpp_processed', aug_crop=256, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter)" \
    --test_dataset="1_000 @ BlendedMVS(split='val', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/blendedmvs_processed', resolution=(512,384), seed=777) + 1_000 @ MegaDepth(split='val', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/megadepth_dataset_processed', resolution=(512,336), seed=777) + 1_000 @ Co3d(split='test', ROOT='/cis/net/r24a/data/zshao/data/dust3r/data/co3d_subset_processed', resolution=(512,384), seed=777) " \
    --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --model="AsymmetricCroCo3DStereo_ResNet(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed_ResNet', img_size=(512, 512), head_type='dpt_resnet', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --pretrained="/cis/net/r24a/data/zshao/checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth" \
    --lr=0.0001 --min_lr=1e-06 --warmup_epochs=15 --epochs=90 --batch_size=4 --accum_iter=2 \
    --save_freq=5 --keep_freq=10 --eval_freq=1 --print_freq=10 --disable_cudnn_benchmark \
    --output_dir="/cis/net/r24a/data/zshao/checkpoints/dust3r/dust3r_512dpt_resnet"


torchrun --nproc_per_node 4 train.py \
    --train_dataset " + 100_000 @ Habitat(800_000, split='train', ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ BlendedMVS(split='train', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ MegaDepth(split='train', ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ ARKitScenes(split='train', ROOT='/home/cpeng26/scratchrchella4/data/arkitscenes_processed', aug_crop=256, resolution=224, transform=ColorJitter) + 100_000 @ Co3d(split='train',ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter) + 100_000 @ StaticThings3D('/home/cpeng26/scratchrchella4/data/static_3d_dataset_processed', aug_crop=256, mask_bg='rand', resolution=224, transform=ColorJitter) + 100_000 @ ScanNetpp(split='train',ROOT='/home/cpeng26/scratchrchella4/data/scannetpp_processed', aug_crop=256, resolution=224, transform=ColorJitter)" \
    --test_dataset " Habitat(1_000, ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', split='val', resolution=224, seed=777) + 1_000 @ BlendedMVS(split='val', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', resolution=224, seed=777) + 1_000 @ MegaDepth(split='val', ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', resolution=224, seed=777) + 1_000 @ Co3d(split='test', ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed', mask_bg='rand', resolution=224, seed=777)" \
    --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --model="AsymmetricCroCo3DStereo_DINOv2(pos_embed='RoPE100', img_size=(224, 224), patch_size=14, head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), freeze='none', enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --pretrained="/home/cpeng26/scratchrchella4/checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth" \
    --lr=0.0001 --min_lr=1e-06 --warmup_epochs=10 --epochs=100 --batch_size=16 --accum_iter=1 \
    --save_freq=5 --keep_freq=10 --eval_freq=1 --num_workers=16\
    --output_dir "/home/cpeng26/scratchrchella4/checkpoints/dinov2train_decoder_224"	

tmux 2 c002 ./output 
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
    --train_dataset " + 100_000 @ Habitat(800_000, split='train', ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ BlendedMVS(split='train', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ ARKitScenes(split='train', ROOT='/home/cpeng26/scratchrchella4/data/arkitscenes_processed', aug_crop=256, resolution=224, transform=ColorJitter) + 100_000 @ Co3d(split='train',ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter) + 100_000 @ StaticThings3D('/home/cpeng26/scratchrchella4/data/static_3d_dataset_processed', aug_crop=256, mask_bg='rand', resolution=224, transform=ColorJitter) + 100_000 @ ScanNetpp(split='train',ROOT='/home/cpeng26/scratchrchella4/data/scannetpp_processed', aug_crop=256, resolution=224, transform=ColorJitter) + 200_000 @ MegaDepth_all(split='train', ROOT='/home/cpeng26/scratchrchella4/data/megadepth', min_overlap=0.01, aug_crop=16, resolution=224, transform=ColorJitter) + 200_000 @ MegaDepth_all(split='train', ROOT='/home/cpeng26/scratchrchella4/data/megadepth', min_overlap=0.35, aug_crop=16, resolution=224, transform=ColorJitter)" \
    --test_dataset " Habitat(1_000, ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', split='val', resolution=224, seed=777) + 1_000 @ BlendedMVS(split='val', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', resolution=224, seed=777) + 1_000 @ MegaDepth(split='val', ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', resolution=224, seed=777) + 1_000 @ Co3d(split='test', ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed', mask_bg='rand', resolution=224, seed=777)" \
    --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --model="AsymmetricCroCo3DStereo(pos_embed='RoPE100',img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --pretrained="/home/cpeng26/scratchrchella4/checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth" \
    --lr=0.0001 --min_lr=1e-06 --warmup_epochs=10 --epochs=100 --batch_size=16 --accum_iter=1 \
    --save_freq=5 --keep_freq=10 --eval_freq=1 --num_workers=16 \
    --output_dir "/home/cpeng26/scratchrchella4/checkpoints/dust3r_224_megadepth_all"

stopped 70
torchrun --nproc_per_node 4 train.py \
    --train_dataset " + 100_000 @ Habitat(800_000, split='train', ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ BlendedMVS(split='train', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ ARKitScenes(split='train', ROOT='/home/cpeng26/scratchrchella4/data/arkitscenes_processed', aug_crop=256, resolution=224, transform=ColorJitter) + 100_000 @ Co3d(split='train',ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter) + 100_000 @ StaticThings3D('/home/cpeng26/scratchrchella4/data/static_3d_dataset_processed', aug_crop=256, mask_bg='rand', resolution=224, transform=ColorJitter) + 100_000 @ ScanNetpp(split='train',ROOT='/home/cpeng26/scratchrchella4/data/scannetpp_processed', aug_crop=256, resolution=224, transform=ColorJitter) + 200_000 @ MegaDepth_all(split='train', ROOT='/home/cpeng26/scratchrchella4/data/megadepth', min_overlap=0.01, aug_crop=16, resolution=224, transform=ColorJitter) + 200_000 @ MegaDepth_all(split='train', ROOT='/home/cpeng26/scratchrchella4/data/megadepth', min_overlap=0.35, aug_crop=16, resolution=224, transform=ColorJitter)" \
    --test_dataset " Habitat(1_000, ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', split='val', resolution=224, seed=777) + 1_000 @ BlendedMVS(split='val', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', resolution=224, seed=777) + 1_000 @ MegaDepth(split='val', ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', resolution=224, seed=777) + 1_000 @ Co3d(split='test', ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed', mask_bg='rand', resolution=224, seed=777)" \
    --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --model="AsymmetricCroCo3DStereo(pos_embed='RoPE100',img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --pretrained="/home/cpeng26/scratchrchella4/checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth" \
    --lr=0.0001 --min_lr=1e-06 --warmup_epochs=10 --epochs=100 --batch_size=24 --accum_iter=1 \
    --save_freq=5 --keep_freq=10 --eval_freq=1 --num_workers=16 \
    --output_dir "/home/cpeng26/scratchrchella4/checkpoints/dust3r_224_megadepth_all"

stopped 83
torchrun --nproc_per_node 4 train.py \
    --train_dataset " + 100_000 @ Habitat(800_000, split='train', ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ BlendedMVS(split='train', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ MegaDepth(split='train', ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ ARKitScenes(split='train', ROOT='/home/cpeng26/scratchrchella4/data/arkitscenes_processed', aug_crop=256, resolution=224, transform=ColorJitter) + 100_000 @ Co3d(split='train',ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter) + 100_000 @ StaticThings3D('/home/cpeng26/scratchrchella4/data/static_3d_dataset_processed', aug_crop=256, mask_bg='rand', resolution=224, transform=ColorJitter) + 100_000 @ ScanNetpp(split='train',ROOT='/home/cpeng26/scratchrchella4/data/scannetpp_processed', aug_crop=256, resolution=224, transform=ColorJitter)" \
    --test_dataset " Habitat(1_000, ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', split='val', resolution=224, seed=777) + 1_000 @ BlendedMVS(split='val', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', resolution=224, seed=777) + 1_000 @ MegaDepth(split='val', ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', resolution=224, seed=777) + 1_000 @ Co3d(split='test', ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed', mask_bg='rand', resolution=224, seed=777)" \
    --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --model="AsymmetricCroCo3DStereo_DINOv2(pos_embed='RoPE100', img_size=(224, 224), patch_size=14, head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), freeze='none', enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --pretrained="/home/cpeng26/scratchrchella4/checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth" \
    --lr=0.0001 --min_lr=1e-06 --warmup_epochs=10 --epochs=100 --batch_size=16 --accum_iter=1 \
    --save_freq=5 --keep_freq=10 --eval_freq=1 --num_workers=16\
    --output_dir "/home/cpeng26/scratchrchella4/checkpoints/dinov2train_decoder_224"

check rope stopped 50
torchrun --nproc_per_node 4 train.py \
    --train_dataset " + 100_000 @ Habitat(800_000, split='train', ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ BlendedMVS(split='train', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ MegaDepth(split='train', ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', aug_crop=16, resolution=224, transform=ColorJitter) + 100_000 @ ARKitScenes(split='train', ROOT='/home/cpeng26/scratchrchella4/data/arkitscenes_processed', aug_crop=256, resolution=224, transform=ColorJitter) + 100_000 @ Co3d(split='train',ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter) + 100_000 @ StaticThings3D('/home/cpeng26/scratchrchella4/data/static_3d_dataset_processed', aug_crop=256, mask_bg='rand', resolution=224, transform=ColorJitter) + 100_000 @ ScanNetpp(split='train',ROOT='/home/cpeng26/scratchrchella4/data/scannetpp_processed', aug_crop=256, resolution=224, transform=ColorJitter)" \
    --test_dataset " Habitat(1_000, ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', split='val', resolution=224, seed=777) + 1_000 @ BlendedMVS(split='val', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', resolution=224, seed=777) + 1_000 @ MegaDepth(split='val', ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', resolution=224, seed=777) + 1_000 @ Co3d(split='test', ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed', mask_bg='rand', resolution=224, seed=777)" \
    --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --model="AsymmetricCroCo3DStereo_DINOv2_rope(pos_embed='RoPE100', img_size=(224, 224), patch_size=14, head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), freeze='none', enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --pretrained="/home/cpeng26/scratchrchella4/checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth" \
    --lr=0.0001 --min_lr=1e-06 --warmup_epochs=10 --epochs=100 --batch_size=24 --accum_iter=1 \
    --save_freq=5 --keep_freq=10 --eval_freq=1 --num_workers=16\
    --output_dir "/home/cpeng26/scratchrchella4/checkpoints/dinov2train_rope_decoder_224"


epoch 10+
torchrun --nproc_per_node 4 train.py \
    --train_dataset=" + 10_000 @ Habitat(800_000, split='train', ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ BlendedMVS(split='train', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ MegaDepth(split='train',  ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ ARKitScenes(split='train', ROOT='/home/cpeng26/scratchrchella4/data/arkitscenes_processed', aug_crop=256, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ Co3d(split='train', ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed', aug_crop=16, mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ StaticThings3D('/home/cpeng26/scratchrchella4/data/static_3d_dataset_processed', aug_crop=256, mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ ScanNetpp(split='train', ROOT='/home/cpeng26/scratchrchella4/data/scannetpp_processed', aug_crop=256, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter)" \
    --test_dataset="1_000 @ BlendedMVS(split='val', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', resolution=(512,384), seed=777) + 1_000 @ MegaDepth(split='val', ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', resolution=(512,336), seed=777) + 1_000 @ Co3d(split='test', ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed', resolution=(512,384), seed=777) " \
    --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --model="AsymmetricCroCo3DStereo_ResNet(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed_VGG', img_size=(512, 512), head_type='dpt_resnet', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --pretrained="/home/cpeng26/scratchrchella4/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth" \
    --lr=0.0001 --min_lr=1e-06 --warmup_epochs=15 --epochs=90 --batch_size=4 --accum_iter=2 \
    --save_freq=5 --keep_freq=10 --eval_freq=1 --print_freq=10 --disable_cudnn_benchmark \
    --output_dir="/home/cpeng26/scratchrchella4/checkpoints/dust3r_512dpt_vgg"

finished ##conv-add with resnet 8, 4 features
torchrun --nproc_per_node 4 train.py \
    --train_dataset=" + 10_000 @ Habitat(800_000, split='train', ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ BlendedMVS(split='train', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ MegaDepth(split='train',  ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ ARKitScenes(split='train', ROOT='/home/cpeng26/scratchrchella4/data/arkitscenes_processed', aug_crop=256, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ Co3d(split='train', ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed', aug_crop=16, mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ StaticThings3D('/home/cpeng26/scratchrchella4/data/static_3d_dataset_processed', aug_crop=256, mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ ScanNetpp(split='train', ROOT='/home/cpeng26/scratchrchella4/data/scannetpp_processed', aug_crop=256, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter)" \
    --test_dataset="1_000 @ BlendedMVS(split='val', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', resolution=(512,384), seed=777) + 1_000 @ MegaDepth(split='val', ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', resolution=(512,336), seed=777) + 1_000 @ Co3d(split='test', ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed', resolution=(512,384), seed=777) " \
    --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --model="AsymmetricCroCo3DStereo_cnn(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed_cnn', img_size=(512, 512), head_type='dpt_resnetrefine', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --pretrained="/home/cpeng26/scratchrchella4/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth" \
    --lr=0.0001 --min_lr=1e-06 --warmup_epochs=15 --epochs=90 --batch_size=4 --accum_iter=2 \
    --save_freq=5 --keep_freq=10 --eval_freq=1 --print_freq=10 --num_workers=16 --disable_cudnn_benchmark \
    --output_dir="/home/cpeng26/scratchrchella4/checkpoints/dust3r_512dpt_resnetrefine"


 ##train stage 3 from pretrained stage 2
torchrun --nproc_per_node 4 train.py \
    --train_dataset=" + 10_000 @ Habitat(800_000, split='train', ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ BlendedMVS(split='train', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ MegaDepth(split='train',  ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ ARKitScenes(split='train', ROOT='/home/cpeng26/scratchrchella4/data/arkitscenes_processed', aug_crop=256, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ Co3d(split='train', ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed', aug_crop=16, mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ StaticThings3D('/home/cpeng26/scratchrchella4/data/static_3d_dataset_processed', aug_crop=256, mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter) + 10_000 @ ScanNetpp(split='train', ROOT='/home/cpeng26/scratchrchella4/data/scannetpp_processed', aug_crop=256, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter)" \
    --test_dataset="1_000 @ BlendedMVS(split='val', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', resolution=(512,384), seed=777) + 1_000 @ MegaDepth(split='val', ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', resolution=(512,336), seed=777) + 1_000 @ Co3d(split='test', ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed', resolution=(512,384), seed=777) " \
    --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --model="AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --pretrained="/home/cpeng26/scratchrchella4/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth" \
    --lr=0.0001 --min_lr=1e-06 --warmup_epochs=15 --epochs=90 --batch_size=4 --accum_iter=2 \
    --save_freq=5 --keep_freq=10 --eval_freq=1 --print_freq=10 --disable_cudnn_benchmark \
    --output_dir="/home/cpeng26/scratchrchella4/checkpoints/dust3r_512dpt_pretrained"


visualize:
cd mast3r/dust3r
python demo_myresnet.py --weights /cis/net/r24a/data/zshao/checkpoints/dust3r/dust3r_512dpt_resnet/checkpoint-best.pth --local_network --server_port 7861



torchrun --nproc_per_node=8 train.py \
    --train_dataset "57_000 @ Habitat512(1_000_000, split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 68_400 @ BlendedMVS(split='train', mask_sky=True, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 68_400 @ MegaDepth(split='train', mask_sky=True, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 45_600 @ ARKitScenes(split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 22_800 @ Co3d(split='train', mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 22_800 @ StaticThings3D(mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 45_600 @ ScanNetpp(split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 45_600 @ TartanAir(pairs_subset='', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 4_560 @ UnrealStereo4K(resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 1_140 @ VirtualKitti(optical_center_is_centered=True, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 22_800 @ WildRGBD(split='train', mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 145_920 @ NianticMapFree(split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 57_000 @ DL3DV(split='nlight', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 57_000 @ DL3DV(split='not-nlight', cosxl_augmentations=None, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 34_200 @ InternalUnreleasedDataset(resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5)" \
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

tmux 6 c006 running
torchrun --nproc_per_node=4 train.py \
    --train_dataset "57_000 @ Habitat(800_000, split='train', ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=16, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 68_400 @ BlendedMVS(split='train', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=16, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 68_400 @ MegaDepth(split='train', ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=16, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 45_600 @ ARKitScenes(split='train', ROOT='/home/cpeng26/scratchrchella4/data/arkitscenes_processed', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=256, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 22_800 @ Co3d(split='train',mask_bg='rand',ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed',resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=256, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 22_800 @ StaticThings3D(ROOT='/home/cpeng26/scratchrchella4/data/static_3d_dataset_processed', mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=256, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 45_600 @ ScanNetpp(split='train', ROOT='/home/cpeng26/scratchrchella4/data/scannetpp_processed', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=256, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 22_800 @ WildRGBD(split='train', ROOT='/home/cpeng26/scratchrchella4/data/wildrgb_processed', mask_bg='rand',resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=16, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5)" \
    --test_dataset "Habitat(1_000, split='val', ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', resolution=(512,384), seed=777, n_corres=1024) + 1_000 @ BlendedMVS(split='val', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', resolution=(512,384), seed=777, n_corres=1024) + 1_000 @ MegaDepth(split='val',  ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', resolution=(512,336), seed=777, n_corres=1024) + 1_000 @ Co3d(split='test', mask_bg='rand', ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed', resolution=(512,384), seed=777, n_corres=1024)" \
    --model "AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, inf))" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='?avg_dis', loss_in_log=False), alpha=0.2) + 0.075*ConfMatchingLoss(MatchingLoss(InfoNCE(mode='proper', temperature=0.05), negatives_padding=0, blocksize=8192), alpha=10.0, confmode='mean')" \
    --test_criterion "Regr3D(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0) + -1.*MatchingLoss(APLoss(nq='torch', fp=torch.float16), negatives_padding=12288)" \
    --pretrained "/home/cpeng26/scratchrchella4/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs 8 --epochs 50 --batch_size 4 --accum_iter 2 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 --print_freq=10 --disable_cudnn_benchmark \
    --output_dir "/home/cpeng26/scratchrchella4/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"

debug
torchrun --nproc_per_node=4 train_warp.py \
    --train_dataset "57_000 @ Habitat(800_000, split='train', ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=16, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 68_400 @ BlendedMVS(split='train', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=16, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 68_400 @ MegaDepth(split='train', ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=16, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 45_600 @ ARKitScenes(split='train', ROOT='/home/cpeng26/scratchrchella4/data/arkitscenes_processed', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=256, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 22_800 @ Co3d(split='train',mask_bg='rand',ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed',resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=256, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 22_800 @ StaticThings3D(ROOT='/home/cpeng26/scratchrchella4/data/static_3d_dataset_processed', mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=256, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 45_600 @ ScanNetpp(split='train', ROOT='/home/cpeng26/scratchrchella4/data/scannetpp_processed', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=256, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 22_800 @ WildRGBD(split='train', ROOT='/home/cpeng26/scratchrchella4/data/wildrgb_processed', mask_bg='rand',resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=16, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5)" \
    --test_dataset "Habitat(1_000, split='val', ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', resolution=(512,384), seed=777, n_corres=1024) + 1_000 @ BlendedMVS(split='val', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', resolution=(512,384), seed=777, n_corres=1024) + 1_000 @ MegaDepth(split='val',  ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', resolution=(512,336), seed=777, n_corres=1024) + 1_000 @ Co3d(split='test', mask_bg='rand', ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed', resolution=(512,384), seed=777, n_corres=1024)" \
    --model "AsymmetricMASt3R_warp(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed_cnn', cnn_type='vgg', img_size=(512, 512), head_type='warp+dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='?avg_dis', loss_in_log=False), alpha=0.2)" \
    --test_criterion "Regr3D(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0)" \
    --pretrained "/home/cpeng26/scratchrchella4/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs 8 --epochs 50 --batch_size 2 --accum_iter 2 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 --print_freq=10 --disable_cudnn_benchmark \
    --output_dir "/home/cpeng26/scratchrchella4/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_warpdpt_metric"

tmux 7 running
torchrun --nproc_per_node=4 train_warp.py \
    --train_dataset "10_000 @ Habitat(800_000, split='train', ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=16, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 10_000 @ BlendedMVS(split='train', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=16, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 10_000 @ MegaDepth(split='train', ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=16, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 10_000 @ ARKitScenes(split='train', ROOT='/home/cpeng26/scratchrchella4/data/arkitscenes_processed', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=256, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 10_000 @ Co3d(split='train',mask_bg='rand',ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed',resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=256, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 10_000 @ StaticThings3D(ROOT='/home/cpeng26/scratchrchella4/data/static_3d_dataset_processed', mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=256, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 10_000 @ ScanNetpp(split='train', ROOT='/home/cpeng26/scratchrchella4/data/scannetpp_processed', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=256, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 10_000 @ WildRGBD(split='train', ROOT='/home/cpeng26/scratchrchella4/data/wildrgb_processed', mask_bg='rand',resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=16, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5)" \
    --test_dataset "Habitat(1_000, split='val', ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', resolution=(512,384), seed=777, n_corres=1024) + 1_000 @ BlendedMVS(split='val', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', resolution=(512,384), seed=777, n_corres=1024) + 1_000 @ MegaDepth(split='val',  ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', resolution=(512,336), seed=777, n_corres=1024) + 1_000 @ Co3d(split='test', mask_bg='rand', ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed', resolution=(512,384), seed=777, n_corres=1024)" \
    --model "AsymmetricMASt3R_cnn_warp(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed_cnn', cnn_type='vgg', img_size=(512, 512), head_type='warp+dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='?avg_dis', loss_in_log=False), alpha=0.2) + -1.*MatchingLoss(APLoss(nq='torch', fp=torch.float16), negatives_padding=12288)"  \
    --test_criterion "Regr3D(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0)" \
    --pretrained "/home/cpeng26/scratchrchella4/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
    --pretrained_warp "/home/cpeng26/scratchrchella4/checkpoints/train_Mast3r_warp_latest.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs 8 --epochs 20 --batch_size 2 --accum_iter 2 \
    --save_freq 1 --keep_freq 2 --eval_freq 1 --print_freq=10 --disable_cudnn_benchmark \
    --output_dir "/home/cpeng26/scratchrchella4/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_warpdpt_metric2" \
    --num_workers 2

tmux 3 c001 running train mast3r warp on dust3r datasets.
torchrun --nproc_per_node=4 train_onlywarp.py \
    --train_dataset "10_000 @ Habitat(800_000, split='train', ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=16, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 10_000 @ BlendedMVS(split='train', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=16, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 10_000 @ MegaDepth(split='train', ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=16, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 10_000 @ ARKitScenes(split='train', ROOT='/home/cpeng26/scratchrchella4/data/arkitscenes_processed', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=256, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 10_000 @ Co3d(split='train',mask_bg='rand',ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed',resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=256, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 10_000 @ StaticThings3D(ROOT='/home/cpeng26/scratchrchella4/data/static_3d_dataset_processed', mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=256, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 10_000 @ ScanNetpp(split='train', ROOT='/home/cpeng26/scratchrchella4/data/scannetpp_processed', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=256, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 10_000 @ WildRGBD(split='train', ROOT='/home/cpeng26/scratchrchella4/data/wildrgb_processed', mask_bg='rand',resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=16, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5)" \
    --test_dataset "Habitat(1_000, split='val', ROOT='/home/cpeng26/scratchrchella4/data/habitat_processed', resolution=(512,384), seed=777, n_corres=1024) + 1_000 @ BlendedMVS(split='val', ROOT='/home/cpeng26/scratchrchella4/data/blendedmvs_processed', resolution=(512,384), seed=777, n_corres=1024) + 1_000 @ MegaDepth(split='val',  ROOT='/home/cpeng26/scratchrchella4/data/megadepth_dataset_processed', resolution=(512,336), seed=777, n_corres=1024) + 1_000 @ Co3d(split='test', mask_bg='rand', ROOT='/home/cpeng26/scratchrchella4/data/co3d_subset_processed', resolution=(512,384), seed=777, n_corres=1024)" \
    --model "AsymmetricMASt3R_only_warp(freeze='backbone', pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed_cnn', cnn_type='vgg', img_size=(512, 512), head_type='warp+dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='?avg_dis', loss_in_log=False), alpha=0.2)" \
    --test_criterion "Regr3D(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0)" \
    --pretrained "/home/cpeng26/scratchrchella4/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
    --pretrained_warp "/home/cpeng26/scratchrchella4/checkpoints/train_Mast3r_warp_latest.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs 8 --epochs 20 --batch_size 8 --accum_iter 2 \
    --save_freq 1 --keep_freq 2 --eval_freq 1 --print_freq=10 --disable_cudnn_benchmark \
    --output_dir "/home/cpeng26/scratchrchella4/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_warp_metric" \
    --num_workers 8

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
    