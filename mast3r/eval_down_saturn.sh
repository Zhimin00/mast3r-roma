CUDA_VISIBLE_DEVICES=2 python relpose_hpatches.py --weights /home/jovyan/workspace/checkpoints/finetune_aerial_mast3r_megadepth/checkpoint-best.pth  --model "AsymmetricMASt3R" --datapath /home/jovyan/workspace/data > /home/jovyan/workspace/checkpoints/mast3r_results/aerial_mast3r_onlymega_hpatches.txt & 
CUDA_VISIBLE_DEVICES=0 python relpose.py --weights /home/jovyan/workspace/checkpoints/finetune_aerial_mast3r_megadepth/checkpoint-best.pth --dataset "RelPoseScanNet1500('/home/jovyan/workspace/data/scannet1500', pairsfile='test', resolution=(512,384))" --pose_estimator cv2 > /home/jovyan/workspace/checkpoints/mast3r_results/aerial_mast3r_onlymega_scannet1500.txt & 
CUDA_VISIBLE_DEVICES=1 python relpose.py --weights /home/jovyan/workspace/checkpoints/finetune_aerial_mast3r_megadepth/checkpoint-best.pth --dataset "RelPoseMegaDepth1500('/home/jovyan/workspace/data/megadepth1500', pairsfile='megadepth_test_pairs', resolution=(512,384))" --pose_estimator cv2 > /home/jovyan/workspace/checkpoints/mast3r_results/aerial_mast3r_onlymega_megadepth1500.txt &
#CUDA_VISIBLE_DEVICES=2 python relpose_hpatches.py --weights /home/cpeng26/scratchrchella4/checkpoints/MASt3R_onlywarp_megadepth_0424/checkpoint-last.pth  --model "AsymmetricMASt3R_only_warp" --datapath /home/cpeng26/scratchrchella4/data > /home/cpeng26/scratchrchella4/checkpoints/mast3r_results/mast3r_onlywarp_hpatches.txt & 
#CUDA_VISIBLE_DEVICES=0 python relpose_warp.py --weights /home/cpeng26/scratchrchella4/checkpoints/MASt3R_onlywarp_megadepth_0424/checkpoint-last.pth --dataset "RelPoseScanNet1500('/home/cpeng26/scratchrchella4/data/scannet1500', pairsfile='test', resolution=(512,384))" --pose_estimator cv2 > /home/cpeng26/scratchrchella4/checkpoints/mast3r_results/mast3r_onlywarp_scannet1500.txt & 
#CUDA_VISIBLE_DEVICES=1 python relpose_warp.py --weights /home/cpeng26/scratchrchella4/checkpoints/MASt3R_onlywarp_megadepth_0424/checkpoint-last.pth --dataset "RelPoseMegaDepth1500('/home/cpeng26/scratchrchella4/data/megadepth1500', pairsfile='megadepth_test_pairs', resolution=(512,384))" --pose_estimator cv2 > /home/cpeng26/scratchrchella4/checkpoints/mast3r_results/mast3r_onlywarp_megadepth1500.txt &

wait

echo "All scripts have finished running."


