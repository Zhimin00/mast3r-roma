import argparse
import os
from collections import defaultdict
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
from PIL import Image
import torchvision.transforms as transforms

from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r_relpose.datasets import *
from tqdm import tqdm
import cv2
from romatch.utils import pose_auc
from dust3r.inference import inference, make_batch_symmetric
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.model import AsymmetricMASt3R, AsymmetricMASt3R_warp, AsymmetricMASt3R_only_warp
from dust3r.utils.device import collate_with_cat
import torch.nn.functional as F
import random

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, help="path to the model weights", default=None)
    parser.add_argument("--model_name", type=str, help="model name", default='AsymmetricMASt3R')
    parser.add_argument("--datapath", type=str, default='/home/jovyan/workspace/data', help="hpatches dir")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--output_dir", type=str, default=None, help="output path")
    parser.add_argument('--num_workers', type=int, default=10)
    return parser


def dense_match(corresps, symmetric = True):
    im_A_to_im_B = corresps[1]["flow"]
    im_A_to_im_B = im_A_to_im_B.permute(
                0, 2, 3, 1
            )
    _, h, w, _ = im_A_to_im_B.shape
    b = 1
    low_res_certainty = F.interpolate(
                    corresps[16]["certainty"], size=(h, w), align_corners=False, mode="bilinear"
                )
    cert_clamp = 0
    factor = 0.5
    low_res_certainty = factor*low_res_certainty*(low_res_certainty < cert_clamp)
    certainty = corresps[1]["certainty"] - low_res_certainty
    
    im_A_coords = torch.meshgrid(
        (
            torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=im_A_to_im_B.device),
            torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=im_A_to_im_B.device),
        ),
        indexing='ij'
    )
    im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
    im_A_coords = im_A_coords[None].expand(b, 2, h, w)
    certainty = certainty.sigmoid()  # logits -> probs
    
    im_A_coords = im_A_coords.permute(0, 2, 3, 1)
    if (im_A_to_im_B.abs() > 1).any() and True:
        wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
        certainty[wrong[:, None]] = 0
    im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
    if symmetric:
        A_to_B, B_to_A = im_A_to_im_B.chunk(2)
        q_warp = torch.cat((im_A_coords, A_to_B), dim=-1)
        im_B_coords = im_A_coords
        s_warp = torch.cat((B_to_A, im_B_coords), dim=-1)
        warp = torch.cat((q_warp, s_warp), dim=2)
        certainty = torch.cat(certainty.chunk(2), dim=3)
    else:
        warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)

    return (warp[0], certainty[0,0])

def kde(x, std = 0.1, down = None):
    # use a gaussian kernel to estimate density
    if down is not None:
        scores = (-torch.cdist(x,x[::down])**2/(2*std**2)).exp()
    else:
        scores = (-torch.cdist(x,x)**2/(2*std**2)).exp()
    density = scores.sum(dim=-1)
    return density

def sample_to_sparse(dense_matches,
            dense_certainty,
            num=10000,
            sample_mode = "threshold_balanced",
    ):
        if "threshold" in sample_mode:
            upper_thresh = 0.05
            dense_certainty = dense_certainty.clone()
            dense_certainty_ = dense_certainty.clone()
            dense_certainty[dense_certainty > upper_thresh] = 1
        matches, certainty = (
            dense_matches.reshape(-1, 4),
            dense_certainty.reshape(-1),
        )
        # noinspection PyUnboundLocalVariable
        certainty_ = dense_certainty_.reshape(-1)
        expansion_factor = 4 if "balanced" in sample_mode else 1
        if not certainty.sum(): certainty = certainty + 1e-8
        good_samples = torch.multinomial(certainty,
                                         num_samples=min(expansion_factor * num, len(certainty)),
                                         replacement=False)
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        good_certainty_ = certainty_[good_samples]
        good_certainty = good_certainty_
        if "balanced" not in sample_mode:
            return good_matches, good_certainty

        density = kde(good_matches, std=0.1)
        p = 1 / (density+1)
        p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        balanced_samples = torch.multinomial(p,
                                             num_samples = min(num,len(good_certainty)),
                                             replacement=False)
        return good_matches[balanced_samples], good_certainty[balanced_samples]


class HpatchesHomogBenchmark:
    """Hpatches grid goes from [0,n-1] instead of [0.5,n-0.5]"""

    def __init__(self, dataset_path, seqs_dir = "hpatches-sequences-release") -> None:
        #seqs_dir = "hpatches-sequences-release"#-v"#release"
        self.seqs_path = os.path.join(dataset_path, seqs_dir)
        self.seq_names = sorted(os.listdir(self.seqs_path))
        # Ignore seqs is same as LoFTR.
        self.ignore_seqs = set(
            [
                "i_contruction",
                "i_crownnight",
                "i_dc",
                "i_pencils",
                "i_whitebuilding",
                "v_artisans",
                "v_astronautis",
                "v_talent",
            ]
        )

    def convert_coordinates(self, im_A_coords, im_A_to_im_B, wq, hq, wsup, hsup):
        offset = 0.5  # Hpatches assumes that the center of the top-left pixel is at [0,0] (I think)
        im_A_coords = (
            np.stack(
                (
                    wq * (im_A_coords.cpu().numpy()[..., 0] + 1) / 2,
                    hq * (im_A_coords.cpu().numpy()[..., 1] + 1) / 2,
                ),
                axis=-1,
            )
            - offset
        )
        im_A_to_im_B = (
            np.stack(
                (
                    wsup * (im_A_to_im_B.cpu().numpy()[..., 0] + 1) / 2,
                    hsup * (im_A_to_im_B.cpu().numpy()[..., 1] + 1) / 2,
                ),
                axis=-1,
            )
            - offset
        )
        return im_A_coords, im_A_to_im_B
    
    @torch.no_grad()
    def benchmark_mast3r(self, model, device):
        n_matches = []
        homog_dists = []
        for seq_idx, seq_name in tqdm(
            enumerate(self.seq_names), total=len(self.seq_names)
        ):
            hs, ws = 512, 384
            transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize(size=(hs, ws)),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) ,
                            # Converts image to tensor (scales to [0,1])
                ])
            im_A_path = os.path.join(self.seqs_path, seq_name, "1.ppm")
            im_A = Image.open(im_A_path)
            w1, h1 = im_A.size
            im_A = transform(im_A)[None].to(device)
            
            for im_idx in range(2, 7):
                im_B_path = os.path.join(self.seqs_path, seq_name, f"{im_idx}.ppm")
                im_B = Image.open(im_B_path)
                w2, h2 = im_B.size
                H = np.loadtxt(
                    os.path.join(self.seqs_path, seq_name, "H_1_" + str(im_idx))
                )
                                
                im_B = Image.open(im_B_path)
                im_B = transform(im_B)[None].to(device)

                batch = [({"img": im_A, "idx": 0, "instance": 0},
                    {"img": im_B, "idx": 1, "instance": 1})]
                view1, view2 = collate_with_cat(batch[:1])
                pred1, pred2 = model(view1, view2)

                desc1, desc2 = (
                    pred1['desc'].squeeze(0).detach(),
                    pred2['desc'].squeeze(0).detach(),
                )

                matches_im0, matches_im1 = fast_reciprocal_NNs(
                    desc1,
                    desc2,
                    subsample_or_initxy1=2,
                    device=device,
                    dist="dot",
                    block_size=2**13,
                )

                kpts1 = matches_im0.copy()
                kpts2 = matches_im1.copy()
                
                offset = 0.5
               
                pos_a = (
                    np.stack(
                        (
                            w1 / ws * kpts1[...,0], 
                            h1 / hs * kpts1[...,1]
                        ), 
                        axis=-1,
                    )
                    - offset
                )
                pos_b = (
                    np.stack(
                        (
                            w2 / ws * kpts2[...,0], 
                            h2 / hs * kpts2[...,1]
                        ), 
                        axis=-1,
                    )
                    - offset
                )

                try:
                    H_pred, inliers = cv2.findHomography(
                        pos_a,
                        pos_b,
                        method = cv2.RANSAC,
                        confidence = 0.99999,
                        ransacReprojThreshold = 3 * min(w2, h2) / 480,
                    )
                except:
                    H_pred = None
                if H_pred is None:
                    H_pred = np.zeros((3, 3))
                    H_pred[2, 2] = 1.0
                corners = np.array(
                    [[0, 0, 1], [0, h1 - 1, 1], [w1 - 1, 0, 1], [w1 - 1, h1 - 1, 1]]
                )
                real_warped_corners = np.dot(corners, np.transpose(H))
                real_warped_corners = (
                    real_warped_corners[:, :2] / real_warped_corners[:, 2:]
                )
                warped_corners = np.dot(corners, np.transpose(H_pred))
                warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
                mean_dist = np.mean(
                    np.linalg.norm(real_warped_corners - warped_corners, axis=1)
                ) / (min(w2, h2) / 480.0)
                homog_dists.append(mean_dist)

        n_matches = np.array(n_matches)
        thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        #auc, accus = pose_auc(np.array(homog_dists), thresholds)
        auc = pose_auc(np.array(homog_dists), thresholds)
        return {
            "hpatches_homog_auc_3": auc[2],
            "hpatches_homog_auc_5": auc[4],
            "hpatches_homog_auc_10": auc[9],
        }
    @torch.no_grad()
    def benchmark_mast3r_warp(self, model, device):
        n_matches = []
        homog_dists = []
        for seq_idx, seq_name in tqdm(
            enumerate(self.seq_names), total=len(self.seq_names)
        ):
            hs, ws = 512, 384
            transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize(size=(hs, ws)),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) ,
                            # Converts image to tensor (scales to [0,1])
                ])
            im_A_path = os.path.join(self.seqs_path, seq_name, "1.ppm")
            im_A = Image.open(im_A_path)
            w1, h1 = im_A.size
            im_A = transform(im_A)[None].to(device)
            
            for im_idx in range(2, 7):
                im_B_path = os.path.join(self.seqs_path, seq_name, f"{im_idx}.ppm")
                im_B = Image.open(im_B_path)
                w2, h2 = im_B.size
                H = np.loadtxt(
                    os.path.join(self.seqs_path, seq_name, "H_1_" + str(im_idx))
                )
                
                
                im_B = Image.open(im_B_path)
                im_B = transform(im_B)[None].to(device)

                batch = [({"img": im_A, "idx": 0, "instance": 0},
                 {"img": im_B, "idx": 1, "instance": 1})]
                batch = collate_with_cat(batch[:1])
                view1, view2 = make_batch_symmetric(batch)
                _, _ , corresps = model(view1, view2)
                
                dense_matches, dense_certainty = dense_match(corresps)
                sparse_matches, sparse_certainty = sample_to_sparse(dense_matches, dense_certainty, 5000)
                pos_a, pos_b = self.convert_coordinates(
                    sparse_matches[:, :2], sparse_matches[:, 2:], w1, h1, w2, h2
                )

                try:
                    H_pred, inliers = cv2.findHomography(
                        pos_a,
                        pos_b,
                        method = cv2.RANSAC,
                        confidence = 0.99999,
                        ransacReprojThreshold = 3 * min(w2, h2) / 480,
                    )
                except:
                    H_pred = None
                if H_pred is None:
                    H_pred = np.zeros((3, 3))
                    H_pred[2, 2] = 1.0
                corners = np.array(
                    [[0, 0, 1], [0, h1 - 1, 1], [w1 - 1, 0, 1], [w1 - 1, h1 - 1, 1]]
                )
                real_warped_corners = np.dot(corners, np.transpose(H))
                real_warped_corners = (
                    real_warped_corners[:, :2] / real_warped_corners[:, 2:]
                )
                warped_corners = np.dot(corners, np.transpose(H_pred))
                warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
                mean_dist = np.mean(
                    np.linalg.norm(real_warped_corners - warped_corners, axis=1)
                ) / (min(w2, h2) / 480.0)
                homog_dists.append(mean_dist)

        n_matches = np.array(n_matches)
        thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        #auc, accus = pose_auc(np.array(homog_dists), thresholds)
        auc = pose_auc(np.array(homog_dists), thresholds)
        return {
            "hpatches_homog_auc_3": auc[2],
            "hpatches_homog_auc_5": auc[4],
            "hpatches_homog_auc_10": auc[9],
        }


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # If using CUDA
    torch.backends.cudnn.deterministic = True  # Makes CUDA deterministic
    torch.backends.cudnn.benchmark = False  

    device = args.device
    weights = args.weights
    model = eval(args.model_name).from_pretrained(weights).to(device)
    hpatches_benchmark = HpatchesHomogBenchmark(args.datapath + "/hpatches-sequence-release")
    
    if 'warp' in args.model_name:
        hpatches_results = hpatches_benchmark.benchmark_mast3r_warp(model, device)
    else:
        hpatches_results = hpatches_benchmark.benchmark_mast3r(model, device)
    print(args.model_name, args.weights)
    print(hpatches_results)