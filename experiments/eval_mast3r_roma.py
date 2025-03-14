from importlib import import_module
import json

from romatch.benchmarks import MegadepthDenseBenchmark
from romatch.benchmarks import MegaDepthPoseEstimationBenchmark, HpatchesHomogBenchmark
from romatch.benchmarks import Mega1500PoseLibBenchmark, SpairBenchmark
import torch
import os
from argparse import ArgumentParser

dataset_path = '/cis/net/r24a/data/zshao/data'
def test_mega_8_scenes(model, name):
    mega_8_scenes_benchmark = MegaDepthPoseEstimationBenchmark(dataset_path+"/megadepth/megadepth_test_8_scenes",
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
    json.dump(mega_8_scenes_results, open(f"./results/mega_8_scenes_{name}.json", "w"))

def test_mega1500(model, name):
    mega1500_benchmark = MegaDepthPoseEstimationBenchmark(dataset_path+"/megadepth/megadepth_test_1500")
    mega1500_results = mega1500_benchmark.benchmark(model, model_name=name)
    json.dump(mega1500_results, open(f"./results/mega1500_{name}.json", "w"))

def test_mega1500_poselib(model, name):
    mega1500_benchmark = Mega1500PoseLibBenchmark(dataset_path + "/megadepth/megadepth_test_1500")
    mega1500_results = mega1500_benchmark.benchmark(model, model_name=name)
    json.dump(mega1500_results, open(f"./results/mega1500_poselib_{name}.json", "w"))

def test_mega_dense(model, name):
    megadense_benchmark = MegadepthDenseBenchmark(dataset_path, num_samples = 1000, h=560,w=560)
    megadense_results = megadense_benchmark.benchmark(model)
    json.dump(megadense_results, open(f"./results/mega_dense_{name}.json", "w"))
    
def test_hpatches(model, name):
    hpatches_benchmark = HpatchesHomogBenchmark(dataset_path + "/hpatches-sequence-release")
    hpatches_results = hpatches_benchmark.benchmark(model)
    json.dump(hpatches_results, open(f"./results/hpatches_{name}.json", "w"))
    
    hpatches_benchmark = HpatchesHomogBenchmark(dataset_path + "/hpatches-sequence-release", seqs_dir="hpatches-sequences-v")
    hpatches_results = hpatches_benchmark.benchmark(model)
    json.dump(hpatches_results, open(f"./results/hpatches_view_{name}.json", "w"))

    hpatches_benchmark = HpatchesHomogBenchmark(dataset_path + "/hpatches-sequence-release", seqs_dir="hpatches-sequences-i")
    hpatches_results = hpatches_benchmark.benchmark(model)
    json.dump(hpatches_results, open(f"./results/hpatches_illumination_{name}.json", "w"))

def test_spair(model, name):
    spair_benchmark = SpairBenchmark()
    spair_result = spair_benchmark.benchmark(model)
    json.dump(spair_result, open(f"./results/spair_{name}.json", "w"))

if __name__ == "__main__":
    from romatch import romaSD_outdoor, romaSD16_8_outdoor, romaSD8_outdoor, roma_outdoor, romaDinoSD_outdoor, romaDinoSD_linear_outdoor, Mast3r_Roma_outdoor
    device = "cuda"
    parser = ArgumentParser()
    parser.add_argument("--model_path", default='/cis/home/zshao14/Downloads/RoMa/experiments/workspace/checkpoints/train_MyromaSD_outdoor_latest.pth', type = str)
    parser.add_argument("--exp_name", default='MyromaSD_outdoor', type=str)
    args, _ = parser.parse_known_args()
    weights = torch.load(args.model_path, map_location=device)['model']
    if 'romaSD_outdoor' in args.model_path:
        model = romaSD_outdoor(device = device, weights=weights, coarse_res = 560, upsample_res=14*16*6)
    elif 'romaSD16_8_outdoor' in args.model_path:
        model = romaSD16_8_outdoor(device = device, weights=weights, coarse_res = 560, upsample_res=14*16*6)
    elif 'romaSD8_outdoor' in args.model_path:
        model = romaSD8_outdoor(device = device, weights=weights, coarse_res = 560, upsample_res=14*16*6)
    elif 'romaDinoSD_512_512_finetune' in args.model_path:
        print('romaDinoSD_512_512_finetune')
        model = romaDinoSD_linear_outdoor(device = device, weights=weights, coarse_res = 560, upsample_res=14*16*6)
    elif 'romaDinoSD_outdoor' in args.model_path:
        model = romaDinoSD_outdoor(device = device, weights=weights, coarse_res = 560, upsample_res=14*16*6)
    elif 'Mast3r_roma' in args.model_path:
        model = Mast3r_Roma_outdoor(device = device, weights = weights, coarse_res = 560, upsample_res= 14*16*6)
    experiment_name = args.exp_name
    #test_spair(model, experiment_name)
    
    test_mega1500(model, experiment_name)
    #test_mega1500_poselib(model, experiment_name)
    test_mega_8_scenes(model, experiment_name)
    test_hpatches(model, experiment_name)
    #test_spair(model, experiment_name)
