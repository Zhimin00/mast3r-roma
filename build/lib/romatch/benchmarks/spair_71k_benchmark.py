import torch
import numpy as np
import tqdm
from romatch.datasets import MegadepthBuilder
from romatch.utils import warp_kpts
from torch.utils.data import ConcatDataset
import romatch
import pdb
import json
import os
from PIL import Image
import torch.nn.functional as F

class SpairBenchmark:
    def __init__(self, data_root='/cis/net/r24a/data/zshao/data/SPair-71k', h = 384, w = 512, num_samples = 2000) -> None:
        self.dataset_path = data_root
        self.test_path = 'PairAnnotation/test'
        json_list = os.listdir(os.path.join(self.dataset_path, self.test_path))
        all_cats = os.listdir(os.path.join(self.dataset_path, 'JPEGImages'))
        cat2json = {}
        for cat in all_cats:
            cat_list = []
            for i in json_list:
                if cat in i:
                    cat_list.append(i)
            cat2json[cat] = cat_list
        self.all_cats = all_cats
        self.cat2json = cat2json

    def benchmark(self, model):
        model.train(False)
        total_pck = []
        all_correct = 0
        all_total = 0
        cat_point_pcks = {}
        cat_image_pcks = {}
        for cat in self.all_cats:
            cat_list = self.cat2json[cat]
            
            cat_pck = []
            cat_correct = 0
            cat_total = 0

            for json_path in tqdm.tqdm(cat_list):
                with open(os.path.join(self.dataset_path, self.test_path, json_path)) as temp_f:
                    data = json.load(temp_f)
                im_A_path = os.path.join(self.dataset_path, 'JPEGImages', cat, data['src_imname'])
                im_B_path = os.path.join(self.dataset_path, 'JPEGImages', cat, data['trg_imname'])

                W_A, H_A = Image.open(im_A_path).size
                W_B, H_B = Image.open(im_B_path).size
                # Match
                warp, certainty = model.match(im_A_path, im_B_path)
    
                kptsA = torch.Tensor(data['src_kps']).float().cuda()
                kpts_A = torch.stack((2/W_A * kptsA[...,0] - 1, 2/H_A * kptsA[...,1] - 1),axis=-1)
                
                H,W2,_ = warp.shape
                W = W2//2
                kpts_B_matched = F.grid_sample(warp[:, :W, -2:].permute(2,0,1)[None], kpts_A[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
                kpts2 = model._to_pixel_coordinates(kpts_B_matched, H_B, W_B)

                trg_bndbox = data['trg_bndbox']
                threshold = max(trg_bndbox[3] - trg_bndbox[1], trg_bndbox[2] - trg_bndbox[0])

                total = 0
                correct = 0
                for idx in range(len(data['src_kps'])):
                    total += 1
                    cat_total += 1
                    all_total += 1
                    pred_point = kpts2[idx]
                    trg_point = data['trg_kps'][idx]
                    total += 1
                    dist = ((pred_point[1].data.cpu().numpy() - trg_point[1])**2 + (pred_point[0].data.cpu().numpy() - trg_point[0])**2)**0.5
                    if (dist / threshold) <= 0.1:
                        correct += 1
                        cat_correct += 1
                        all_correct += 1

                cat_pck.append(correct / total)
                
            total_pck.extend(cat_pck)
            cat_image_pcks[cat] = np.mean(cat_pck) * 100
            cat_point_pcks[cat] = cat_correct / cat_total * 100

            print(f'{cat} per image PCK@0.1: {np.mean(cat_pck) * 100:.2f}')
            print(f'{cat} per point PCK@0.1: {cat_correct / cat_total * 100:.2f}')
        print(f'All per image PCK@0.1: {np.mean(total_pck) * 100:.2f}')
        print(f'All per point PCK@0.1: {all_correct / all_total * 100:.2f}')
        cat_point_pcks['total'] = all_correct / all_total * 100
        return cat_point_pcks
        

