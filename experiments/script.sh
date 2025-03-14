cd ~/Downloads/RoMa/LDFF
python BEV_KITTI_test.py
torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d BEV_KITTI_train.py --dpp 1 --batch_size 12 --n_gpus 4 --mixed_precision --lr 0.00016

torchrun --nproc_per_node=2 --nnodes=1 --rdzv_backend=c10d BEV_KITTI_train.py --dpp 1 --batch_size 6 --n_gpus 2 --mixed_precision --lr 0.00002

torchrun --nproc_per_node=2 --nnodes=1 --rdzv_backend=c10d BEV_KITTI_train.py --dpp 1 --batch_size 6 --n_gpus 2 --lr 0.00002





python train.py --train_resolution medium --gpu_batch_size 8 --wandb_entity <your_wandb_account>
python train.py --train_resolution medium --gpu_batch_size 8 --dont_log_wandb

torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d train_romaSD_outdoor.py --gpu_batch_size 1
torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d train_coarse_matcher.py --gpu_batch_size 4

## multiple machines
nohup torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="10.160.220.27" --master_port=29500 train_romaSD_outdoor.py --gpu_batch_size 2 > node0.out 2>&1 &
nohup torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr="10.160.220.27" --master_port=29500 train_romaSD_outdoor.py --gpu_batch_size 2 > node1.out 2>&1 &

torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d train_romaSD_outdoor.py --gpu_batch_size 1
torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d train_coarse_matcher.py --gpu_batch_size 8

##probe
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --nnodes=1 --rdzv_backend=c10d eval_coarse.py --gpu_batch_size 8 --experiment diffusion --t 0
torchrun --nproc_per_node=1 --nnodes=1 --rdzv_backend=c10d eval_coarse_depthpro.py --gpu_batch_size 1 --experiment depthpro

##linear projection
nohup torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d train_coarse_linear.py --gpu_batch_size 16 --experiment googlenet/dino_SD/dinov2/diffusion/vgg19/resnet50 --t 0 > node0.out 2>&1 &
nohup torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d train_s8_linear.py --gpu_batch_size 16 --experiment googlenet/diffusion/vgg19/resnet50 --t 0 > node0.out 2>&1 &



cd ~/Downloads/RoMa/experiments
conda activate mast3r
export DATASET_PATH=/export/io104/data/zshao14/megadepth
export DATASET_PATH=/cis/net/io96/data/zshao
nohup torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d train_BEVroma.py --gpu_batch_size 8 > BEVroma.out 2>&1 &
nohup torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d train_romaSD_outdoor.py --gpu_batch_size 2 > romaSD.out 2>&1 &
nohup torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d train_roma_outdoor.py --gpu_batch_size 2 > roma.out 2>&1 &
nohup torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d train_MyromaSD_outdoor.py --gpu_batch_size 8 > MyromaSD.out 2>&1 &

nohup torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d train_MyromaDinoSD_outdoor.py --gpu_batch_size 6 > MyromaDinoSD_con.out 2>&1 &

python eval_myromaSD_outdoor_model.py --exp_name MyromaSD_16_latest --model_path /cis/home/zshao14/Downloads/RoMa/experiments/workspace/checkpoints/train_MyromaSD_outdoor_latest.pth
python eval_myromaSD_outdoor_model.py --exp_name MyromaSD16_8_latest --model_path /cis/home/zshao14/Downloads/RoMa/experiments/workspace/checkpoints/train_MyromaSD16_8_outdoor_latest.pth
python eval_myromaSD_outdoor_model.py --exp_name MyromaSD8_latest --model_path /cis/home/zshao14/Downloads/RoMa/experiments/workspace/checkpoints/train_MyromaSD8_outdoor_latest.pth
python eval_myromaSD_outdoor_model.py --exp_name MyromaDinoSD_latest --model_path /cis/home/zshao14/Downloads/RoMa/experiments/workspace/checkpoints/train_MyromaDinoSD_outdoor_latest.pth
python eval_myromaSD_outdoor_model.py --exp_name MyromaDinoSD_linear_latest --model_path /cis/home/zshao14/Downloads/RoMa/experiments/workspace/checkpoints/train_MyromaDinoSD_outdoor2_latest.pth

python eval_mast3r_roma.py --exp_name mast3r-roma_scale8 --model_path /cis/home/zshao14/Downloads/RoMa_mast3r/experiments/workspace/mast3r/train_Mast3r_roma_latest2.pth
python eval_mast3r_roma.py --exp_name dinosd-roma_scale8 --model_path /cis/home/zshao14/Downloads/RoMa_mast3r/experiments/workspace/dinoSD_512/train_MyromaDinoSD_outdoor_from_scratch6_best.pth
python eval_roma_outdoor.py 
## change the learning rate encoder 不变 decoder 1e-5
nohup torchrun --nproc_per_node=8 --nnodes=1 --rdzv_backend=c10d train_MyromaDinoSD_512_512_finetune.py --gpu_batch_size 6 > RoMadinoSD_512_512_finetune.out 2>&1 &

nohup torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d train_MyromaDinoSD_512_512_scratch.py --gpu_batch_size 16 > RoMadinoSD_512_512_scratch.out 2>&1 &
pck1 still fluctuates
变成encoder 2304-512 ,decoder 512-512
nohup torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d train_MyromaDinoSD_512_512_scratch.py --gpu_batch_size 16 > RoMadinoSD_512_512_scratch.out 2>&1 &

export PYTHONPATH="$PYTHONPATH:/cis/home/zshao14/Downloads/RoMa/mast3r"
nohup torchrun --nproc_per_node=8 --nnodes=1 --rdzv_backend=c10d mast3r/train_Mast3r_roma_loader.py --train_dataset "MegaDepth2(split='train', ROOT='/cis/net/io96/data/zshao/dust3r/data/megadepth_dataset_processed', resolution=560)" --gpu_batch_size 8 > Mast3r_roma_3D.out 2>&1 &



## need cont. 
# finetune 512 with 8M steps
nohup torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d train_MyromaDinoSD_outdoor2.py --gpu_batch_size 16 > RoMadinoSD_512_finetune.out 2>&1 &



nohup torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d train_roma_dpt.py --gpu_batch_size 8 > roma_dpt.out 2>&1 &








