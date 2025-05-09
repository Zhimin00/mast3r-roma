import os
from PIL import Image
import h5py
import numpy as np
import torch
import torchvision.transforms.functional as tvf
import kornia.augmentation as K
from romatch.utils import get_depth_tuple_transform_ops, get_tuple_transform_ops
import romatch
from romatch.utils import *
import math
import pdb
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """ Open an image or a depthmap with opencv-python.
    """
    if path.endswith(('.exr', 'EXR')):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f'Could not load image={path} with {options=}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class MegadepthScene:
    def __init__(
        self,
        data_root,
        scene_info,
        ht=384,
        wt=512,
        min_overlap=0.0,
        max_overlap=1.0,
        shake_t=0,
        rot_prob=0.0,
        normalize=True,
        max_num_pairs = 100_000,
        scene_name = None,
        use_horizontal_flip_aug = False,
        use_single_horizontal_flip_aug = False,
        colorjiggle_params = None,
        random_eraser = None,
        use_randaug = False,
        randaug_params = None,
        randomize_size = False,
    ) -> None:
        self.data_root = data_root
        self.scene_name = os.path.splitext(scene_name)[0]+f"_{min_overlap}_{max_overlap}"
        self.image_paths = scene_info["image_paths"]
        self.depth_paths = scene_info["depth_paths"]
        self.intrinsics = scene_info["intrinsics"]
        self.poses = scene_info["poses"]
        self.pairs = scene_info["pairs"]
        self.overlaps = scene_info["overlaps"]
        threshold = (self.overlaps > min_overlap) & (self.overlaps < max_overlap)
        self.pairs = self.pairs[threshold]
        self.overlaps = self.overlaps[threshold]
        if len(self.pairs) > max_num_pairs:
            pairinds = np.random.choice(
                np.arange(0, len(self.pairs)), max_num_pairs, replace=False
            )
            self.pairs = self.pairs[pairinds]
            self.overlaps = self.overlaps[pairinds]
        if randomize_size:
            area = ht * wt
            s = int(16 * (math.sqrt(area)//16))
            sizes = ((ht,wt), (s,s), (wt,ht))
            choice = romatch.RANK % 3
            ht, wt = sizes[choice] 
        # counts, bins = np.histogram(self.overlaps,20)
        # print(counts)
        self.im_transform_ops = get_tuple_transform_ops(
            resize=(ht, wt), normalize=normalize, colorjiggle_params = colorjiggle_params,
        )
        self.depth_transform_ops = get_depth_tuple_transform_ops(
                resize=(ht, wt)
            )
        self.wt, self.ht = wt, ht
        self.shake_t = shake_t
        self.random_eraser = random_eraser
        if use_horizontal_flip_aug and use_single_horizontal_flip_aug:
            raise ValueError("Can't both flip both images and only flip one")
        self.use_horizontal_flip_aug = use_horizontal_flip_aug
        self.use_single_horizontal_flip_aug = use_single_horizontal_flip_aug
        self.use_randaug = use_randaug

    def load_im(self, im_path):
        im = Image.open(im_path)
        return im
    
    def horizontal_flip(self, im_A, im_B, depth_A, depth_B,  K_A, K_B):
        im_A = im_A.flip(-1)
        im_B = im_B.flip(-1)
        depth_A, depth_B = depth_A.flip(-1), depth_B.flip(-1) 
        flip_mat = torch.tensor([[-1, 0, self.wt],[0,1,0],[0,0,1.]]).to(K_A.device)
        K_A = flip_mat@K_A  
        K_B = flip_mat@K_B  
        
        return im_A, im_B, depth_A, depth_B, K_A, K_B
    
    def load_depth(self, depth_ref, crop=None):
        depth = np.array(h5py.File(depth_ref, "r")["depth"])
        return torch.from_numpy(depth)

    def __len__(self):
        return len(self.pairs)

    def scale_intrinsic(self, K, wi, hi):
        sx, sy = self.wt / wi, self.ht / hi
        sK = torch.tensor([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        return sK @ K

    def rand_shake(self, *things):
        t = np.random.choice(range(-self.shake_t, self.shake_t + 1), size=2)
        return [
            tvf.affine(thing, angle=0.0, translate=list(t), scale=1.0, shear=[0.0, 0.0])
            for thing in things
        ], t

    def __getitem__(self, pair_idx):
        # read intrinsics of original size
        idx1, idx2 = self.pairs[pair_idx]
        K1 = torch.tensor(self.intrinsics[idx1].copy(), dtype=torch.float).reshape(3, 3)
        K2 = torch.tensor(self.intrinsics[idx2].copy(), dtype=torch.float).reshape(3, 3)

        # read and compute relative poses
        T1 = self.poses[idx1]
        T2 = self.poses[idx2]
        T_1to2 = torch.tensor(np.matmul(T2, np.linalg.inv(T1)), dtype=torch.float)[
            :4, :4
        ]  # (4, 4)
        T1 = torch.tensor(T1.copy(), dtype=torch.float)
        T2 = torch.tensor(T2.copy(), dtype=torch.float)
        # Load positive pair data
        im_A, im_B = self.image_paths[idx1], self.image_paths[idx2]
        depth1, depth2 = self.depth_paths[idx1], self.depth_paths[idx2]
        im_A_ref = os.path.join(self.data_root, im_A)
        im_B_ref = os.path.join(self.data_root, im_B)
        depth_A_ref = os.path.join(self.data_root, depth1)
        depth_B_ref = os.path.join(self.data_root, depth2)
        im_A = self.load_im(im_A_ref)
        im_B = self.load_im(im_B_ref)
        K1 = self.scale_intrinsic(K1, im_A.width, im_A.height)
        K2 = self.scale_intrinsic(K2, im_B.width, im_B.height)

        if self.use_randaug:
            im_A, im_B = self.rand_augment(im_A, im_B)

        depth_A = self.load_depth(depth_A_ref)
        depth_B = self.load_depth(depth_B_ref)
        # Process images
        im_A, im_B = self.im_transform_ops((im_A, im_B))
        depth_A, depth_B = self.depth_transform_ops(
            (depth_A[None, None], depth_B[None, None])
        )
        
        [im_A, im_B, depth_A, depth_B], t = self.rand_shake(im_A, im_B, depth_A, depth_B)
        K1[:2, 2] += t
        K2[:2, 2] += t
        
        im_A, im_B = im_A[None], im_B[None]
        if self.random_eraser is not None:
            im_A, depth_A = self.random_eraser(im_A, depth_A)
            im_B, depth_B = self.random_eraser(im_B, depth_B)
                
        if self.use_horizontal_flip_aug:
            if np.random.rand() > 0.5:
                im_A, im_B, depth_A, depth_B, K1, K2 = self.horizontal_flip(im_A, im_B, depth_A, depth_B, K1, K2)
        if self.use_single_horizontal_flip_aug:
            if np.random.rand() > 0.5:
                im_B, depth_B, K2 = self.single_horizontal_flip(im_B, depth_B, K2)
        
        if romatch.DEBUG_MODE:
            tensor_to_pil(im_A[0], unnormalize=True).save(
                            f"vis/im_A.jpg")
            tensor_to_pil(im_B[0], unnormalize=True).save(
                            f"vis/im_B.jpg")
            
        data_dict = {
            "im_A": im_A[0],
            "im_A_identifier": self.image_paths[idx1].split("/")[-1].split(".jpg")[0],
            "im_B": im_B[0],
            "im_B_identifier": self.image_paths[idx2].split("/")[-1].split(".jpg")[0],
            "im_A_depth": depth_A[0, 0],
            "im_B_depth": depth_B[0, 0],
            "K1": K1,
            "K2": K2,
            "T1": T1,
            "T2": T2,
            "T_1to2": T_1to2,
            "im_A_path": im_A_ref,
            "im_B_path": im_B_ref,
            "im_A_depth_path": depth_A_ref,
            "im_B_depth_path": depth_B_ref,  
            "domainid": 0,          
        }
        return data_dict


class Aerial_MegaDepth:
    def __init__(
        self,
        data_root,
        split,
        ht=384,
        wt=512,
        shake_t=0,
        rot_prob=0.0,
        normalize=True,
        use_horizontal_flip_aug = False,
        use_single_horizontal_flip_aug = False,
        colorjiggle_params = None,
        random_eraser = None,
        use_randaug = False,
        randaug_params = None,
        randomize_size = False
    ) -> None:
        self.split = split
        self.data_root = data_root
        self.loaded_data = self._load_data(self.split)
        

        if randomize_size:
            area = ht * wt
            s = int(16 * (math.sqrt(area)//16))
            sizes = ((ht,wt), (s,s), (wt,ht))
            choice = romatch.RANK % 3
            ht, wt = sizes[choice] 
        # counts, bins = np.histogram(self.overlaps,20)
        # print(counts)
        self.im_transform_ops = get_tuple_transform_ops(
            resize=(ht, wt), normalize=normalize, colorjiggle_params = colorjiggle_params,
        )
        self.depth_transform_ops = get_depth_tuple_transform_ops(
                resize=(ht, wt)
            )
        self.wt, self.ht = wt, ht
        self.shake_t = shake_t
        self.random_eraser = random_eraser
        if use_horizontal_flip_aug and use_single_horizontal_flip_aug:
            raise ValueError("Can't both flip both images and only flip one")
        self.use_horizontal_flip_aug = use_horizontal_flip_aug
        self.use_single_horizontal_flip_aug = use_single_horizontal_flip_aug
        self.use_randaug = use_randaug


    def _load_data(self, split):
        if split == 'train1':
            with np.load(os.path.join(self.data_root, 'aerial_megadepth_train_part1.npz'), allow_pickle=True) as data:
                self.all_scenes = data['scenes']
                self.all_images = data['images']
                self.pairs = data['pairs']
        elif split == 'train2':
            with np.load(os.path.join(self.data_root, 'aerial_megadepth_train_part2.npz'), allow_pickle=True) as data:
                self.all_scenes = data['scenes']
                self.all_images = data['images']
                self.pairs = data['pairs']
        elif split == 'val':
            with np.load(os.path.join(self.data_root, 'aerial_megadepth_val.npz'), allow_pickle=True) as data:
                self.all_scenes = data['scenes']
                self.all_images = data['images']
                self.pairs = data['pairs']
        elif split == 'test':
            with np.load(os.path.join(self.data_root, 'aerial_megadepth_test_scenes0015_0022.npz'), allow_pickle=True) as data:
                self.all_scenes = data['scenes']
                self.all_images = data['images']
                self.pairs = data['pairs']

    def __len__(self):
        return len(self.pairs)

    def get_stats(self):
        return f'{len(self)} pairs from {len(self.all_scenes)} scenes'
    
    def load_image(self, image_path):
        image_np = imread_cv2(image_path)
        return Image.fromarray(image_np)

    def load_depth(self, depth_path):
        depth_np = imread_cv2(depth_path, options=cv2.IMREAD_ANYDEPTH)
        return torch.from_numpy(depth_np.astype(np.float32))

    def load_intrinsics_and_pose(self, npz_path):
        camera_params = np.load(npz_path)
        K = camera_params["intrinsics"].astype(np.float32)
        T = camera_params["cam2world"].astype(np.float32)
        T_inv = np.linalg.inv(T)  
        return torch.from_numpy(K), torch.from_numpy(T_inv)
    
    def scale_intrinsic(self, K, wi, hi):
        sx, sy = self.wt / wi, self.ht / hi
        sK = torch.tensor([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        return sK @ K
    
    def horizontal_flip(self, im_A, im_B, depth_A, depth_B,  K_A, K_B):
        im_A = im_A.flip(-1)
        im_B = im_B.flip(-1)
        depth_A, depth_B = depth_A.flip(-1), depth_B.flip(-1) 
        flip_mat = torch.tensor([[-1, 0, self.wt],[0,1,0],[0,0,1.]]).to(K_A.device)
        K_A = flip_mat@K_A  
        K_B = flip_mat@K_B  
        
        return im_A, im_B, depth_A, depth_B, K_A, K_B
    
    def rand_shake(self, *things):
        t = np.random.choice(range(-self.shake_t, self.shake_t + 1), size=2)
        return [
            tvf.affine(thing, angle=0.0, translate=list(t), scale=1.0, shear=[0.0, 0.0])
            for thing in things
        ], t
    
    
    def __getitem__(self, pair_idx):
        # read intrinsics of original size
        scene_id, idx1, idx2, score = self.pairs[pair_idx]

        scene = self.all_scenes[scene_id]
        seq_path = os.path.join(self.data_root, scene)


        im_A_name, im_B_name = self.all_images[idx1], self.all_images[idx2]
        
        ## load camera parameters
        K1, T1 = self.load_intrinsics_and_pose(os.path.join(seq_path, im_A_name + ".npz"))
        K2, T2 = self.load_intrinsics_and_pose(os.path.join(seq_path, im_B_name + ".npz"))
        K1 = K1.reshape(3, 3)
        K2 = K2.reshape(3, 3)
        
        T_1to2 = torch.matmul(T2, torch.linalg.inv(T1)).to(dtype=torch.float32)[:4, :4]


        im_A_ref = os.path.join(seq_path, im_A_name + '.jpg')
        im_B_ref = os.path.join(seq_path, im_B_name + '.jpg')
        depth_A_ref = os.path.join(seq_path, im_A_name + ".exr")
        depth_B_ref = os.path.join(seq_path, im_A_name + ".exr")

        im_A = self.load_image(im_A_ref)
        im_B = self.load_image(im_B_ref)

        K1 = self.scale_intrinsic(K1, im_A.width, im_A.height)
        K2 = self.scale_intrinsic(K2, im_B.width, im_B.height)
        
        if self.use_randaug:
            im_A, im_B = self.rand_augment(im_A, im_B)

        depth_A = self.load_depth(depth_A_ref)
        depth_B = self.load_depth(depth_B_ref)
        # Process images
        im_A, im_B = self.im_transform_ops((im_A, im_B))
        depth_A, depth_B = self.depth_transform_ops(
            (depth_A[None, None], depth_B[None, None])
        )
        
        [im_A, im_B, depth_A, depth_B], t = self.rand_shake(im_A, im_B, depth_A, depth_B)
        K1[:2, 2] += t
        K2[:2, 2] += t
        
        im_A, im_B = im_A[None], im_B[None]
        if self.random_eraser is not None:
            im_A, depth_A = self.random_eraser(im_A, depth_A)
            im_B, depth_B = self.random_eraser(im_B, depth_B)
                
        if self.use_horizontal_flip_aug:
            if np.random.rand() > 0.5:
                im_A, im_B, depth_A, depth_B, K1, K2 = self.horizontal_flip(im_A, im_B, depth_A, depth_B, K1, K2)
        if self.use_single_horizontal_flip_aug:
            if np.random.rand() > 0.5:
                im_B, depth_B, K2 = self.single_horizontal_flip(im_B, depth_B, K2)
        
        if romatch.DEBUG_MODE:
            tensor_to_pil(im_A[0], unnormalize=True).save(
                            f"vis/im_A.jpg")
            tensor_to_pil(im_B[0], unnormalize=True).save(
                            f"vis/im_B.jpg")
            
        data_dict = {
            "im_A": im_A[0],
            "im_A_identifier": im_A_name,
            "im_B": im_B[0],
            "im_B_identifier": im_B_name,
            "im_A_depth": depth_A[0, 0],
            "im_B_depth": depth_B[0, 0],
            "K1": K1,
            "K2": K2,
            "T1": T1,
            "T2": T2,
            "T_1to2": T_1to2,
            "im_A_path": im_A_ref,
            "im_B_path": im_B_ref,
            "im_A_depth_path": depth_A_ref,
            "im_B_depth_path": depth_B_ref,    
            "domainid": 1,        
        }
        return data_dict
    
class MegadepthBuilder:
    def __init__(self, data_root="data/megadepth", loftr_ignore=True, imc21_ignore = True) -> None:
        self.data_root = data_root
        self.scene_info_root = os.path.join(data_root, "prep_scene_info")
        self.all_scenes = os.listdir(self.scene_info_root)
        self.test_scenes = ["0017.npy", "0004.npy", "0048.npy", "0013.npy"]
        # LoFTR did the D2-net preprocessing differently than we did and got more ignore scenes, can optionially ignore those
        self.loftr_ignore_scenes = set(['0121.npy', '0133.npy', '0168.npy', '0178.npy', '0229.npy', '0349.npy', '0412.npy', '0430.npy', '0443.npy', '1001.npy', '5014.npy', '5015.npy', '5016.npy'])
        self.imc21_scenes = set(['0008.npy', '0019.npy', '0021.npy', '0024.npy', '0025.npy', '0032.npy', '0063.npy', '1589.npy'])
        self.test_scenes_loftr = ["0015.npy", "0022.npy"]
        self.loftr_ignore = loftr_ignore
        self.imc21_ignore = imc21_ignore

    def build_scenes(self, split="train", min_overlap=0.0, scene_names = None, **kwargs):
        if split == "train":
            scene_names = set(self.all_scenes) - set(self.test_scenes)
        elif split == "train_loftr":
            scene_names = set(self.all_scenes) - set(self.test_scenes_loftr)
        elif split == "test":
            scene_names = self.test_scenes
        elif split == "test_loftr":
            scene_names = self.test_scenes_loftr
        elif split == "custom":
            scene_names = scene_names
        else:
            raise ValueError(f"Split {split} not available")
        scenes = []
        for scene_name in scene_names:
            if self.loftr_ignore and scene_name in self.loftr_ignore_scenes:
                continue
            if self.imc21_ignore and scene_name in self.imc21_scenes:
                continue
            if ".npy" not in scene_name:
                continue
            scene_info = np.load(
                os.path.join(self.scene_info_root, scene_name), allow_pickle=True
            ).item()
            scenes.append(
                MegadepthScene(
                    self.data_root, scene_info, min_overlap=min_overlap,scene_name = scene_name, **kwargs
                )
            )
        return scenes

    def weight_scenes(self, concat_dataset, alpha=0.5):
        ns = []
        for d in concat_dataset.datasets:
            ns.append(len(d))
        ws = torch.cat([torch.ones(n) / n**alpha for n in ns])
        return ws
