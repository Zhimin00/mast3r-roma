nohup torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d train_MyromaSD16_8_outdoor.py --gpu_batch_size 8 > MyromaSD16_8.out 2>&1 & 
wait
nohup torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d train_MyromaSD8_outdoor.py --gpu_batch_size 8 > MyromaSD8.out 2>&1 &