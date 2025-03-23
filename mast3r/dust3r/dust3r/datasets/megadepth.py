# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed MegaDepth
# dataset at https://www.cs.cornell.edu/projects/megadepth/
# See datasets_preprocess/preprocess_megadepth.py
# --------------------------------------------------------
import os.path as osp
import numpy as np
import os
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset, BaseStereoViewDataset2
from dust3r.utils.image import imread_cv2, load_images
import torch

class MegaDepth(BaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.loaded_data = self._load_data(self.split)

        if self.split is None:
            pass
        elif self.split == 'train':
            self.select_scene(('0015', '0022'), opposite=True)
        elif self.split == 'val':
            self.select_scene(('0015', '0022'))
        else:
            raise ValueError(f'bad {self.split=}')

    def _load_data(self, split):
        with np.load(osp.join(self.ROOT, 'all_metadata.npz')) as data:
            self.all_scenes = data['scenes']
            self.all_images = data['images']
            self.pairs = data['pairs']

    def __len__(self):
        return len(self.pairs)

    def get_stats(self):
        return f'{len(self)} pairs from {len(self.all_scenes)} scenes'

    def select_scene(self, scene, *instances, opposite=False):
        scenes = (scene,) if isinstance(scene, str) else tuple(scene)
        scene_id = [s.startswith(scenes) for s in self.all_scenes]
        assert any(scene_id), 'no scene found'

        valid = np.in1d(self.pairs['scene_id'], np.nonzero(scene_id)[0])
        if instances:
            image_id = [i.startswith(instances) for i in self.all_images]
            image_id = np.nonzero(image_id)[0]
            assert len(image_id), 'no instance found'
            # both together?
            if len(instances) == 2:
                valid &= np.in1d(self.pairs['im1_id'], image_id) & np.in1d(self.pairs['im2_id'], image_id)
            else:
                valid &= np.in1d(self.pairs['im1_id'], image_id) | np.in1d(self.pairs['im2_id'], image_id)

        if opposite:
            valid = ~valid
        assert valid.any()
        self.pairs = self.pairs[valid]

    def _get_views(self, pair_idx, resolution, rng):
        scene_id, im1_id, im2_id, score = self.pairs[pair_idx]

        scene, subscene = self.all_scenes[scene_id].split()
        seq_path = osp.join(self.ROOT, scene, subscene)

        views = []

        for im_id in [im1_id, im2_id]:
            img = self.all_images[im_id]
            try:
                image = imread_cv2(osp.join(seq_path, img + '.jpg'))
                depthmap = imread_cv2(osp.join(seq_path, img + ".exr"))
                camera_params = np.load(osp.join(seq_path, img + ".npz"))
            except Exception as e:
                raise OSError(f'cannot load {img}, got exception {e}')

            intrinsics = np.float32(camera_params['intrinsics'])
            camera_pose = np.float32(camera_params['cam2world'])

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(seq_path, img))

            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset='MegaDepth',
                label=osp.relpath(seq_path, self.ROOT),
                instance=img))

        return views

class MegaDepth_all(BaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, min_overlap=0.0, max_overlap=1.0, max_num_pairs = 100_000, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)

        self.scene_info_root = osp.join(ROOT, "prep_scene_info")
        self.all_scenes = os.listdir(self.scene_info_root)
        self.test_scenes_loftr = ['0015.npy', '0022.npy']
        if self.split is None:
            pass
        elif self.split == 'train':
            scene_names = set(self.all_scenes) - set(self.test_scenes_loftr)
        elif self.split == 'val':
            scene_names = self.test_scenes_loftr
        else:
            raise ValueError(f'bad {self.split=}')
        self.scene_names = scene_names
        self.image_paths = {}
        self.depth_paths = {}
        self.intrinsics = {}
        self.poses = {}
        self.pairs = []
        
        self.overlaps = []
        for scene_name in self.scene_names:
            scene_info = np.load(
                osp.join(self.scene_info_root, scene_name), allow_pickle=True
            ).item()
            self.image_paths[scene_name] = scene_info['image_paths']
            self.depth_paths[scene_name] = scene_info['depth_paths']
            self.intrinsics[scene_name] = scene_info['intrinsics']
            self.poses[scene_name] = scene_info['poses']
            pairs = scene_info['pairs']
            overlaps = scene_info['overlaps']
            threshold = (overlaps > min_overlap) & (overlaps < max_overlap)
            pairs = pairs[threshold]
            overlaps = overlaps[threshold]
            if len(pairs) > max_num_pairs:
                pairinds = np.random.choice(
                    np.arange(0, len(pairs), max_num_pairs, replace=False)
                )
                pairs = pairs[pairinds]
                overlaps = overlaps[pairinds]
            for pair in pairs:
                self.pairs.append((scene_name, pair[0], pair[1]))

    def __len__(self):
        return len(self.pairs)

    def get_stats(self):
        return f'{len(self)} pairs from {len(self.all_scenes)} scenes'

    def _get_views(self, pair_idx, resolution, rng):
        scene_name, im1_id, im2_id = self.pairs[pair_idx]

        views = []

        for im_id in [im1_id, im2_id]:
            img_path = self.image_paths[scene_name][im_id]
            depth_path = self.depth_paths[scene_name][im_id]
            try:
                image = imread_cv2(osp.join(self.ROOT, img_path))
                depthmap = imread_cv2(osp.join(self.root, depth_path))
                intrinsics = self.intrinsics[scene_name][im_id] 
                camera_pose = self.poses[scene_name][im_id]

            except Exception as e:
                raise OSError(f'cannot load {img_path}, got exception {e}')

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=osp.join(self.ROOT, img_path))

            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset='MegaDepth',
                label=img_path,
                instance=img_path.split('/')[-1]))

        return views
    
class MegaDepth2(BaseStereoViewDataset2):
    def __init__(self, *args, split, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.loaded_data = self._load_data(self.split)

        if self.split is None:
            pass
        elif self.split == 'train':
            self.select_scene(('0015', '0022'), opposite=True)
        elif self.split == 'val':
            self.select_scene(('0015', '0022'))
        else:
            raise ValueError(f'bad {self.split=}')

    def _load_data(self, split):
        with np.load(osp.join(self.ROOT, 'all_metadata.npz')) as data:
            self.all_scenes = data['scenes']
            self.all_images = data['images']
            self.pairs = data['pairs']

    def __len__(self):
        return len(self.pairs)

    def get_stats(self):
        return f'{len(self)} pairs from {len(self.all_scenes)} scenes'

    def select_scene(self, scene, *instances, opposite=False):
        scenes = (scene,) if isinstance(scene, str) else tuple(scene)
        scene_id = [s.startswith(scenes) for s in self.all_scenes]
        assert any(scene_id), 'no scene found'

        valid = np.in1d(self.pairs['scene_id'], np.nonzero(scene_id)[0])
        if instances:
            image_id = [i.startswith(instances) for i in self.all_images]
            image_id = np.nonzero(image_id)[0]
            assert len(image_id), 'no instance found'
            # both together?
            if len(instances) == 2:
                valid &= np.in1d(self.pairs['im1_id'], image_id) & np.in1d(self.pairs['im2_id'], image_id)
            else:
                valid &= np.in1d(self.pairs['im1_id'], image_id) | np.in1d(self.pairs['im2_id'], image_id)

        if opposite:
            valid = ~valid
        assert valid.any()
        self.pairs = self.pairs[valid]

    def _get_views(self, pair_idx, resolution, rng):
        scene_id, im1_id, im2_id, score = self.pairs[pair_idx]

        scene, subscene = self.all_scenes[scene_id].split()
        seq_path = osp.join(self.ROOT, scene, subscene)

        views = []

        for im_id in [im1_id, im2_id]:
            img = self.all_images[im_id]
            try:
                image = imread_cv2(osp.join(seq_path, img + '.jpg'))
                depthmap = imread_cv2(osp.join(seq_path, img + ".exr"))
                camera_params = np.load(osp.join(seq_path, img + ".npz"))
            except Exception as e:
                raise OSError(f'cannot load {img}, got exception {e}')

            intrinsics = np.float32(camera_params['intrinsics'])
            camera_pose = np.float32(camera_params['cam2world'])

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(seq_path, img))

            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset='MegaDepth',
                label=osp.join(seq_path, img + '.jpg'),
                instance=img))

        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    from mast3r.model import AsymmetricMASt3R
    import pdb
    device = 'cuda'
    dataset = MegaDepth(split='train', ROOT="/cis/home/cpeng/dust3r/data/megadepth_dataset_processed", resolution=560, aug_crop='auto')

    sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=8,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    mast3r_model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model = AsymmetricMASt3R.from_pretrained(mast3r_model_name).to(device)

    for data_iter_step, batch in enumerate(data_loader,0):
        view1, view2 = batch
        ignore_keys = set(['depthmap', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng'])
        for view in batch:
            for name in view.keys():  # pseudo_focal
                if name in ignore_keys:
                    continue
                view[name] = view[name].to(device, non_blocking=True)
        pred1, pred2 = model(view1, view2)

        pdb.set_trace()

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == 2
        print(idx, view_name(views[0]), view_name(views[1]))
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in [0, 1]:
            pts3d = views[view_idx]['pts3d']
            pdb.set_trace()
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=(idx * 255, (1 - idx) * 255, 0),
                           image=colors,
                           cam_size=cam_size)
        viz.show()
