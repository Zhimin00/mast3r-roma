import argparse
import os
from collections import defaultdict
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.geometry import opencv_to_colmap_intrinsics
from dust3r.datasets import get_data_loader
from dust3r.utils.device import collate_with_cat
from dust3r.inference import make_batch_symmetric
from mast3r.model import AsymmetricMASt3R, AsymmetricMASt3R_warp, AsymmetricMASt3R_only_warp
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r_relpose.datasets import *
from tqdm import tqdm
import poselib, cv2, pycolmap
import pdb
import torch.nn.functional as F
import random

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="visloc dataset to eval")
    parser_weights = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("--weights", type=str, help="path to the model weights", default='naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')
    parser.add_argument("--model_name", type=str, help="model name", default='AsymmetricMASt3R_only_warp')

    parser.add_argument("--confidence_threshold", type=float, default=1.001,
                        help="confidence values higher than threshold are invalid")
    parser.add_argument('--pixel_tol', default=5, type=int)

    parser.add_argument("--coarse_to_fine", action='store_true', default=False,
                        help="do the matching from coarse to fine")
    parser.add_argument("--max_image_size", type=int, default=None,
                        help="max image size for the fine resolution")
    parser.add_argument("--c2f_crop_with_homography", action='store_true', default=False,
                        help="when using coarse to fine, crop with homographies to keep cx, cy centered")

    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--pose_estimator", type=str, default="cv2", choices=['cv2', 'poselib', 'pycolmap'],
                        help="Pose lib to use")
    parser_reproj = parser.add_mutually_exclusive_group()
    parser_reproj.add_argument("--reprojection_error", type=float, default=5.0, help="pnp reprojection error")
    parser_reproj.add_argument("--reprojection_error_diag_ratio", type=float, default=None,
                               help="pnp reprojection error as a ratio of the diagonal of the image")

    parser.add_argument("--max_batch_size", type=int, default=48,
                        help="max batch size for inference on crops when using coarse to fine")
    parser.add_argument("--pnp_max_points", type=int, default=100_000, help="pnp maximum number of points kept")
    parser.add_argument("--viz_matches", type=int, default=0, help="debug matches")

    parser.add_argument("--output_dir", type=str, default=None, help="output path")
    parser.add_argument("--output_label", type=str, default='', help="prefix for results files")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument("--use_amp", action='store_true', default=False, help="Use amp in fastnn")
    parser.add_argument("--single_loop", action='store_true', default=False, help="Use single-loop matmul")
    parser.add_argument("--use_tensorrt", action='store_true', default=False, help="Apply tensorrt to the model")
    return parser


class TensorWrapper:
    _data = None

    def __init__(self, data: torch.Tensor):
        self._data = data

    @property
    def shape(self):
        return self._data.shape[:-1]

    @property
    def device(self):
        return self._data.device

    @property
    def dtype(self):
        return self._data.dtype

    def __getitem__(self, index):
        return self.__class__(self._data[index])

    def __setitem__(self, index, item):
        self._data[index] = item.data

    def to(self, *args, **kwargs):
        return self.__class__(self._data.to(*args, **kwargs))

    def cpu(self):
        return self.__class__(self._data.cpu())

    def cuda(self):
        return self.__class__(self._data.cuda())

    def pin_memory(self):
        return self.__class__(self._data.pin_memory())

    def float(self):
        return self.__class__(self._data.float())

    def double(self):
        return self.__class__(self._data.double())

    def detach(self):
        return self.__class__(self._data.detach())

    @classmethod
    def stack(cls, objects, dim=0, *, out=None):
        data = torch.stack([obj._data for obj in objects], dim=dim, out=out)
        return cls(data)

    @classmethod
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func is torch.stack:
            return self.stack(*args, **kwargs)
        else:
            return NotImplemented


class Pose(TensorWrapper):
    def __init__(self, data: torch.Tensor):
        assert data.shape[-1] == 12
        super().__init__(data)

    @classmethod
    def from_Rt(cls, R: torch.Tensor, t: torch.Tensor):
        """Pose from a rotation matrix and translation vector.
        Accepts numpy arrays or PyTorch tensors.

        Args:
            R: rotation matrix with shape (..., 3, 3).
            t: translation vector with shape (..., 3).
        """
        assert R.shape[-2:] == (3, 3)
        assert t.shape[-1] == 3
        assert R.shape[:-2] == t.shape[:-1]
        data = torch.cat([R.flatten(start_dim=-2), t], -1)
        return cls(data)

    # @classmethod
    # def from_aa(cls, aa: torch.Tensor, t: torch.Tensor):
    #     """Pose from an axis-angle rotation vector and translation vector.
    #     Accepts numpy arrays or PyTorch tensors.

    #     Args:
    #         aa: axis-angle rotation vector with shape (..., 3).
    #         t: translation vector with shape (..., 3).
    #     """
    #     assert aa.shape[-1] == 3
    #     assert t.shape[-1] == 3
    #     assert aa.shape[:-1] == t.shape[:-1]
    #     return cls.from_Rt(so3exp_map(aa), t)

    @classmethod
    def from_4x4mat(cls, T: torch.Tensor):
        """Pose from an SE(3) transformation matrix.
        Args:
            T: transformation matrix with shape (..., 4, 4).
        """
        assert T.shape[-2:] == (4, 4)
        R, t = T[..., :3, :3], T[..., :3, 3]
        return cls.from_Rt(R, t)

    # @classmethod
    # def from_colmap(cls, image: NamedTuple):
    #     """Pose from a COLMAP Image."""
    #     return cls.from_Rt(image.qvec2rotmat(), image.tvec)

    @property
    def R(self) -> torch.Tensor:
        """Underlying rotation matrix with shape (..., 3, 3)."""
        rvec = self._data[..., :9]
        return rvec.reshape(rvec.shape[:-1] + (3, 3))

    @property
    def t(self) -> torch.Tensor:
        """Underlying translation vector with shape (..., 3)."""
        return self._data[..., -3:]

    def __repr__(self):
        return f"Pose: {self.shape} {self.dtype} {self.device}"


class Camera(TensorWrapper):
    eps = 1e-4

    def __init__(self, data: torch.Tensor):
        assert data.shape[-1] in {6, 8, 10}
        super().__init__(data)

    @classmethod
    def from_calibration_matrix(cls, K: torch.Tensor):
        cx, cy = K[..., 0, 2], K[..., 1, 2]
        fx, fy = K[..., 0, 0], K[..., 1, 1]
        data = torch.stack([2 * cx, 2 * cy, fx, fy, cx, cy], -1)
        return cls(data)

    @property
    def f(self) -> torch.Tensor:
        """Focal lengths (fx, fy) with shape (..., 2)."""
        return self._data[..., 2:4]

    @property
    def c(self) -> torch.Tensor:
        """Principal points (cx, cy) with shape (..., 2)."""
        return self._data[..., 4:6]

    def normalize(self, p2d: torch.Tensor) -> torch.Tensor:
        """Convert normalized 2D coordinates into pixel coordinates."""
        return (p2d - self.c.unsqueeze(-2)) / self.f.unsqueeze(-2)


    def image2cam(self, p2d: torch.Tensor) -> torch.Tensor:
        """Convert 2D pixel corrdinates to 3D points with z=1"""
        assert self._data.shape
        p2d = self.normalize(p2d)
        # iterative undistortion
        return to_homogeneous(p2d)

    def to_cameradict(self, camera_model=None):
        data = self._data.clone()
        if data.dim() == 1:
            data = data.unsqueeze(0)
        assert data.dim() == 2
        b, d = data.shape
        if camera_model is None:
            camera_model = {6: "PINHOLE", 8: "RADIAL", 10: "OPENCV"}[d]
        cameras = []
        for i in range(b):
            if camera_model.startswith("SIMPLE_"):
                params = [x.item() for x in data[i, 3 : min(d, 7)]]
            else:
                params = [x.item() for x in data[i, 2:]]
            cameras.append(
                {
                    "model": camera_model,
                    "width": int(data[i, 0].item()),
                    "height": int(data[i, 1].item()),
                    "params": params,
                }
            )
        return cameras if self._data.dim() == 2 else cameras[0]

    def __repr__(self):
        return f"Camera {self.shape} {self.dtype} {self.device}"

@torch.no_grad()
def inference(batch, model, device, use_amp=False): 
    # to device. 
    for view in batch:
        for name in 'img camera_intrinsics camera_pose'.split():  
            if name not in view:
                continue
            view[name] = view[name].to(device, non_blocking=True)
    # forward. 
    view1, view2 = batch
    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        return model(view1, view2)

def angle_error_vec(v1, v2, eps=1e-10):
    n = torch.clip(v1.norm(dim=-1) * v2.norm(dim=-1), min=eps)
    v1v2 = (v1 * v2).sum(dim=-1)  # dot product in the last dimension
    return torch.rad2deg(torch.arccos(torch.clip(v1v2 / n, -1.0, 1.0)))

def angle_error_mat(R1, R2):
    cos = (torch.trace(torch.einsum("...ij, ...jk -> ...ik", R1.T, R2)) - 1) / 2
    cos = torch.clip(cos, -1.0, 1.0)  # numerical errors can make it out of bounds
    return torch.rad2deg(torch.abs(torch.arccos(cos)))

def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0, eps=1e-10):
    if isinstance(T_0to1, torch.Tensor):
        R_gt, t_gt = T_0to1[:3, :3], T_0to1[:3, 3]
    else:
        R_gt, t_gt = T_0to1.R, T_0to1.t
    R_gt, t_gt = torch.squeeze(R_gt), torch.squeeze(t_gt)

    # angle error between 2 vectors
    t_err = angle_error_vec(t, t_gt, eps)
    t_err = torch.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if t_gt.norm() < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    r_err = angle_error_mat(R, R_gt)

    return t_err, r_err


def get_matches_scores(kpts0, kpts1, matches0, mscores0):
    m0 = matches0 > -1
    m1 = matches0[m0]
    pts0 = kpts0[m0]
    pts1 = kpts1[m1]
    scores = mscores0[m0]
    return pts0, pts1, scores

def eval_relative_pose_robust(data, pred, estimator='cv2', **kw):
    T_gt = data["T_0to1"][0]
    kp0, kp1 = pred["keypoints0"].squeeze(0), pred["keypoints1"].squeeze(0)
    m0, scores0 = pred["matches0"].squeeze(0), pred["matching_scores0"].squeeze(0)
    pts0, pts1, scores = get_matches_scores(kp0, kp1, m0, scores0)

    results = {}
    
    data_ = {
        "m_kpts0": pts0,
        "m_kpts1": pts1,
        "camera0": Camera.from_calibration_matrix(data["view0"]["camera_intrinsics"][0]).float(),
        "camera1": Camera.from_calibration_matrix(data["view1"]["camera_intrinsics"][0]).float(),
        "img_size0": list(data["view0"]['img'].shape[2:])[::-1],
        "img_size1": list(data["view1"]['img'].shape[2:])[::-1],
    }

    assert estimator in ['cv2', 'poselib', 'pycolmap'], f"Invalid estimator: {estimator}"
    if estimator == 'cv2':
        est = OpenCVRelativePoseEstimator(data_, **kw)
    elif estimator == 'poselib':
        est = PoseLibRelativePoseEstimator(data_, **kw)
    elif estimator == 'pycolmap':
        # est = PycolmapTwoViewEstimator(data_, **kw)
        raise Exception("Currently unsupported.")

    if not est["success"]:
        results["rel_pose_error"] = float("inf")
        results["ransac_inl"] = 0
        results["ransac_inl%"] = 0
    else:
        # R, t, inl = ret
        M = est["M_0to1"]
        inl = est["inliers"].cpu().detach().numpy()
        t_error, r_error = relative_pose_error(T_gt, M.R, M.t)
        results["rel_pose_error"] = max(r_error, t_error).cpu().detach()
        results["ransac_inl"] = np.sum(inl)
        results["ransac_inl%"] = np.mean(inl)

    return results

def from_homogeneous(points, eps=0.0):
    return points[..., :-1] / (points[..., -1:] + eps)

def to_homogeneous(points):
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1] + (1,))
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError

def OpenCVRelativePoseEstimator(data, ransac_th=0.5, confidence=0.99999):
    solver = cv2.RANSAC
    kpts0, kpts1 = data["m_kpts0"], data["m_kpts1"]
    camera0 = data["camera0"]
    camera1 = data["camera1"]
    M, inl = None, torch.zeros_like(kpts0[:, 0]).bool()

    if len(kpts0) >= 5:
        f_mean = torch.cat([camera0.f, camera1.f]).mean().item()
        norm_thresh = ransac_th / f_mean

        pts0 = from_homogeneous(camera0.image2cam(kpts0)).cpu().detach().numpy()
        pts1 = from_homogeneous(camera1.image2cam(kpts1)).cpu().detach().numpy()
        E, mask = cv2.findEssentialMat(
            pts0,
            pts1,
            np.eye(3),
            threshold=norm_thresh,
            prob=confidence,
            method=solver,
        )

        if E is not None:
            best_num_inliers = 0
            for _E in np.split(E, len(E) / 3):
                n, R, t, _ = cv2.recoverPose(
                    _E, pts0, pts1, np.eye(3), 1e9, mask=mask
                )
                if n > best_num_inliers:
                    best_num_inliers = n
                    inl = torch.tensor(mask.ravel() > 0)
                    M = Pose.from_Rt(
                        torch.tensor(R).to(kpts0), torch.tensor(t[:, 0]).to(kpts0)
                    )

    estimation = {
        "success": M is not None,
        "M_0to1": M if M is not None else Pose.from_4x4mat(torch.eye(4).to(kpts0)),
        "inliers": inl.to(device=kpts0.device),
    }

    return estimation
        
def PoseLibRelativePoseEstimator(data, ransac_th=2.0):
    pts0, pts1 = data["m_kpts0"], data["m_kpts1"]
    camera0 = data["camera0"]
    camera1 = data["camera1"]
    img_size0 = data["img_size0"]
    img_size1 = data["img_size1"]
    
    M, info = poselib.estimate_relative_pose(
        pts0.cpu().detach().numpy(),
        pts1.cpu().detach().numpy(),
        camera0.to_cameradict(),
        camera1.to_cameradict(),
        {
            "max_epipolar_error": ransac_th,
        },
    )
    success = M is not None
    if success:
        M = Pose.from_Rt(torch.tensor(M.R), torch.tensor(M.t)).to(pts0)
    else:
        M = Pose.from_4x4mat(torch.eye(4)).to(pts0)

    estimation = {
        "success": success,
        "M_0to1": M,
        "inliers": torch.tensor(info.pop("inliers")).to(pts0),
        **info,
    }

    return estimation


def PycolmapTwoViewEstimator(data, ransac_th=4.0):
    options = {**pycolmap.TwoViewGeometryOptions().todict()}
    pts0, pts1 = data["m_kpts0"], data["m_kpts1"]
    camera0 = data["camera0"]
    camera1 = data["camera1"]

    info = pycolmap.two_view_geometry_estimation(
        pts0.cpu().detach().numpy(),
        pts1.cpu().detach().numpy(),
        camera0.to_cameradict(),
        camera1.to_cameradict(),
        options,
    )
    success = info["success"]
    if success:
        R = pycolmap.qvec_to_rotmat(info["qvec"])
        t = info["tvec"]
        M = Pose.from_Rt(torch.tensor(R), torch.tensor(t)).to(pts0)
        inl = torch.tensor(info.pop("inliers")).to(pts0)
    else:
        M = Pose.from_4x4mat(torch.eye(4)).to(pts0)
        inl = torch.zeros_like(pts0[:, 0]).bool()

    estimation = {
        "success": success,
        "M_0to1": M,
        "inliers": inl,
        "type": str(
            info.get("configuration_type", pycolmap.TwoViewGeometry.UNDEFINED)
        ),
    }

    return estimation

class AUCMetric:
    def __init__(self, thresholds, elements=None):
        self._elements = elements
        self.thresholds = thresholds
        if not isinstance(thresholds, list):
            self.thresholds = [thresholds]

    def update(self, tensor):
        assert tensor.dim() == 1
        self._elements += tensor.cpu().numpy().tolist()

    def compute(self):
        if len(self._elements) == 0:
            return np.nan
        else:
            return cal_error_auc(self._elements, self.thresholds)

def cal_error_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.round((np.trapz(r, x=e) / t), 4))
    return aucs

def eval_poses(pose_results, auc_ths, key, unit="°"):
    pose_aucs = {}
    best_th = -1
    for th, results_i in pose_results.items():
        pose_aucs[th] = AUCMetric(auc_ths, results_i[key]).compute()
    mAAs = {k: np.mean(v) for k, v in pose_aucs.items()}
    best_th = max(mAAs, key=mAAs.get)

    if len(pose_aucs) > -1:
        print("Tested ransac setup with following results:")
        print("AUC", pose_aucs)
        print("mAA", mAAs)
        print("best threshold =", best_th)

    summaries = {}

    for i, ath in enumerate(auc_ths):
        summaries[f"{key}@{ath}{unit}"] = pose_aucs[best_th][i]
    summaries[f"{key}_mAA"] = mAAs[best_th]

    for k, v in pose_results[best_th].items():
        arr = np.array(v)
        if not np.issubdtype(np.array(v).dtype, np.number):
            continue
        summaries[f"m{k}"] = round(np.median(arr), 3)
    return summaries, best_th

def correct_intrinsic(b_view):
    for i in range(len(b_view['true_shape'])):
        height, width = b_view['true_shape'][i]
        if width < height:
            b_view['camera_intrinsics'][i] = b_view['camera_intrinsics'][i][[1, 0, 2]]

            K = b_view['camera_intrinsics'][i]
            temp = K[0,0]
            K[0,0] = K[1,1]
            K[1,1] = temp

            temp = K[0,2]
            K[0,2] = K[1,2]
            K[1,2] = temp
            b_view['camera_intrinsics'][i] = K

class InferenceWrapper(torch.nn.Module):
    def __init__(self, engine, device):
        super().__init__()
        self.engine = engine
        self.context = engine.create_execution_context()
        self.device = device

    def forward(self, *args):
        for i in range(len(args)):
            self.context.set_input_shape(f"input{i+1}", args[i].shape)
        output = torch.empty(*self.context.get_tensor_shape("output"), device=self.device, dtype=args[0].dtype)

        bindings = [int(input.data_ptr()) for input in args] + [int(output.data_ptr())]
        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), bindings[i])
        self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)

        return output


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



def to_pixel_coordinates(coords, H_A, W_A, H_B = None, W_B = None):
    
    if coords.shape[-1] == 2:
        return _to_pixel_coordinates(coords, H_A, W_A) 
    
    if isinstance(coords, (list, tuple)):
        kpts_A, kpts_B = coords[0], coords[1]
    else:
        kpts_A, kpts_B = coords[...,:2], coords[...,2:]
    return _to_pixel_coordinates(kpts_A, H_A, W_A), _to_pixel_coordinates(kpts_B, H_B, W_B)

def _to_pixel_coordinates(coords, H, W):
    offset = 0.5
    kpts = torch.stack((W/2 * (coords[...,0]+1), H/2 * (coords[...,1]+1)),axis=-1) - offset
    return kpts

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


    conf_thr = args.confidence_threshold
    device = args.device
    weights = args.weights
    pose_estimator = args.pose_estimator
    assert args.pixel_tol > 0
    reprojection_error = args.reprojection_error
    reprojection_error_diag_ratio = args.reprojection_error_diag_ratio
    pnp_max_points = args.pnp_max_points
    viz_matches = args.viz_matches

    model = eval(args.model_name).from_pretrained(weights).to(device)

    fast_nn_params = dict(device=device, dist='dot', block_size=2**13)
    dataset = eval(args.dataset)
    data_loader_test = get_data_loader(dataset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             pin_mem=True,
                             shuffle=False,
                             drop_last=False)
    print('Dataset length: ', len(data_loader_test))

    # start evaluation
    rerrs, terrs = [], []
    print(f'Testing {args.dataset}')

    results = defaultdict(list)
    test_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    pose_results = defaultdict(lambda: defaultdict(list))
    if 'scannet' in args.dataset:
        print('scannet')
    else:
        print('megadepth')

    with torch.no_grad():
        for batch in tqdm(data_loader_test):

            if 'scannet' in args.dataset:
                view1, view2 = make_batch_symmetric(batch)
                batch = (view1, view2)
                _, _ , corresps = inference(batch, model, device, use_amp=False)
                dense_matches, dense_certainty = dense_match(corresps)
            else:
                _, _ , corresps = inference(batch, model, device, use_amp=False)
                view1, view2 = batch
                dense_matches, dense_certainty = dense_match(corresps, symmetric = False)   

            sparse_matches, sparse_certainty = sample_to_sparse(dense_matches, dense_certainty, 5000)

            h1, w1 = view1['true_shape'][0]
            h2, w2 = view2['true_shape'][0]
            h1, w1, h2, w2 = int(h1), int(w1), int(h2), int(w2)
            kp0, kp1 = to_pixel_coordinates(sparse_matches, h1, w1, h2, w2)
            kp0, kp1 = kp0.clone(), kp1.clone()

            k0, c0 = kp0.shape
            k1, c1 = kp1.shape

            matches0 = torch.arange(k0, device=device).unsqueeze(0)
            matches1 = matches0.clone()

            mscores0 = sparse_certainty[None].clone().float()
            mscores1 = sparse_certainty[None].clone().float()

            pred = {
                "keypoints0": kp0[None],
                "keypoints1": kp1[None],
                "matches0": matches0,
                "matches1": matches1,
                "matching_scores0": mscores0,
                "matching_scores1": mscores1,
            }

            correct_intrinsic(view1)
            correct_intrinsic(view2)
            data = {
                # "T_0to1": view1['camera_pose'] @ torch.inverse(view2['camera_pose']),
                "T_0to1": torch.inverse(view2['camera_pose']) @ view1['camera_pose'],
                "view0": view1,
                "view1": view2,
                "scene": view1['dataset'],
                "name": f"{view1['instance']}_{view2['instance']}"
            }

            results_i = {}
            for th in test_thresholds:
                pose_results_i = eval_relative_pose_robust(
                    data,
                    pred,
                    ransac_th=th,
                    estimator=pose_estimator,
                )
                [pose_results[th][k].append(v) for k, v in pose_results_i.items()]

            # we also store the names for later reference
            results_i["names"] = data["name"]
            if "scene" in data.keys():
                results_i["scenes"] = data["scene"]

            for k, v in results_i.items():
                results[k].append(v)
            
            torch.cuda.synchronize()
        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(np.array(v).dtype, np.number):
                continue
            summaries[f"m{k}"] = round(np.mean(arr), 3)

        best_pose_results, best_th = eval_poses(
            pose_results, auc_ths=[5, 10, 20], key="rel_pose_error"
        )
        results = {**results, **pose_results[best_th]}
        summaries = {
            **summaries,
            **best_pose_results,
        }

        print(summaries)