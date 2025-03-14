import json

from romatch.benchmarks import MegadepthDenseBenchmark
from romatch.benchmarks import MegaDepthPoseEstimationBenchmark, HpatchesHomogBenchmark
from romatch.benchmarks import Mega1500PoseLibBenchmark, KittiBenchmark, SpairBenchmark
import os


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
    megadense_benchmark = MegadepthDenseBenchmark(dataset_path+'/megadepth', num_samples = 1000, h=560,w=560)
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

def test_kitti(model, name):
    kitti_benchmark = KittiBenchmark(batch_size=1)
    kitti_benchmark.benchmark1(model)
    kitti_benchmark.benchmark2(model)

def test_spair(model, name):
    spair_benchmark = SpairBenchmark()
    spair_result = spair_benchmark.benchmark(model)
    json.dump(spair_result, open(f"./results/spair_{name}.json", "w"))

if __name__ == "__main__":
    from romatch import roma_outdoor
    device = "cuda"
    model = roma_outdoor(device = device, coarse_res = 560, upsample_res=14*16*6)
    experiment_name = "RoMa_scale8"
    #test_spair(model, experiment_name)
    #test_kitti(model, experiment_name)
    
    #test_mega_dense(model, experiment_name)
    test_mega1500(model, experiment_name)
    #test_mega1500_poselib(model, experiment_name)
    test_mega_8_scenes(model, experiment_name)
    test_hpatches(model, experiment_name)
