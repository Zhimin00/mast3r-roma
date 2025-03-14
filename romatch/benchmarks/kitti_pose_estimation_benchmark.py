import numpy as np
import torch
from romatch.utils import *
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import romatch
import kornia.geometry.epipolar as kepi
import time
# wrap cause pyposelib is still in dev
# will add in deps later
import poselib
from LDFF.dataLoader.KITTI_dataset_BEV import load_train_data, load_test1_data, load_test2_data
from LDFF.RANSAC_lib.euclidean_trans import Least_Squares_weight, rt2edu_matrix, Least_Squares
import cv2
import os
import pdb
import LDFF.gen_BEV.utils as LDFF_utils
#from hloc import extract_features, extractors
#from hloc.utils.base_model import dynamic_load
import math

from romatch.utils.utils import to_cuda

def coords_grid(batch, ht, wd, device):#[B,2,H, W]
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)
def convert_to_cv_style(pil_array):
    # Ensure pil_array has shape [B, 3, H, W]
    assert pil_array.ndim == 4 and pil_array.shape[1] == 3, "Input should have shape [B, 3, H, W]"

    # Convert RGB to Grayscale
    grayscale = (
        0.2989 * pil_array[:, 0, :, :] +  # R channel
        0.5870 * pil_array[:, 1, :, :] +  # G channel
        0.1140 * pil_array[:, 2, :, :]    # B channel
    )  # Shape: [B, H, W]

    return grayscale

def matches(inds1, inds2, matchcerts, kpts1, kpts2):
    inds1 = inds1.detach().cpu().numpy()
    inds2 = inds2.detach().cpu().numpy()
    matchcerts = matchcerts.detach().cpu().numpy()
    # convert to hloc match format, and threshold confidence by requested match threshold
    val_inds1 = []
    val_inds2 = []
    val_certs = []
    for idx, i1 in enumerate(inds1.tolist()):
        certainty = float(matchcerts[idx])
        if certainty >= 0.1:
            val_inds1.append(i1)
            val_inds2.append(inds2[idx])
            val_certs.append(certainty)

    return val_inds1, val_inds2, val_certs

def match(mask, rot, tran_x, tran_y, coords0):
    #coords0: N, 2
    B,C,H,W = mask.size()
    coords1 = []

    for i in range(B):
        coords1_i = (coords0.clone())[None,:]#[1,N,2]
        N = coords0.shape[0]
        ones = torch.ones((1,N,1)).to(coords0.device)
        coords1_i = torch.cat((coords1_i, ones), dim=-1)
        coords1_i = coords1_i.view(N, 3, 1)

        rot1 = rot[i][None,:]/180*math.pi
        cos = torch.cos(rot1)
        cos = cos[:,None]
        sin = torch.sin(rot1)
        sin = sin[:,None]
        zero = torch.zeros_like(sin)
        ones = torch.ones_like(sin)
        tran_x1 = tran_x[i][None,:][:,None]
        tran_y1 = tran_y[i][None,:][:,None]

        rol_tra0 = torch.cat((cos,sin,zero),dim=-1)
        rol_tra1 = torch.cat((-sin,cos,zero),dim=-1)
        rol_tra2 = torch.cat((zero,zero,ones),dim=-1)
        rol_tra = torch.cat((rol_tra0,rol_tra1,rol_tra2),dim = 1)
        rol_tra = rol_tra.repeat(N, 1, 1)

        rol_center0 = torch.cat((ones,zero,ones*(-H/2)),dim=-1)
        rol_center1 = torch.cat((zero,ones,ones*(-W/2)),dim=-1)
        rol_center2 = torch.cat((zero,zero,ones),dim=-1)
        rol_center = torch.cat((rol_center0,rol_center1,rol_center2),dim = 1)
        rol_center = rol_center.repeat(N, 1, 1)

        tra0 = torch.cat((ones,zero,(-tran_x1)),dim=-1)
        tra1 = torch.cat((zero,ones,(tran_y1)),dim=-1)
        tra2 = torch.cat((zero,zero,ones),dim=-1)
        tra = torch.cat((tra0, tra1, tra2),dim = 1)
        tra = tra.repeat(N, 1, 1)

        points = torch.rand((B,3)).to(mask.device)
        points_tran = ((torch.inverse(rol_center))@rol_tra@rol_center@tra@coords1_i)
        #points_tran = (tra@coords1)

        coords1_i = (points_tran[:,:2,:]).view(1, N, 2)
        coords1.append(coords1_i)
    
    coords1 = torch.cat(coords1,dim=0)
    return coords1

class KittiBenchmark:
    def __init__(self, batch_size=1, shift_range_lat=20, shift_range_lon=20, rotation_range=10, dpp=1, save_path = "./results/kitti") -> None:
        self.batch_size = batch_size
        self.shift_range_lat = shift_range_lat
        self.shift_range_lon = shift_range_lon
        self.rotation_range = rotation_range
        self.dpp = dpp
        self.save_path = save_path
        self.dataloader = load_test1_data(self.batch_size, self.shift_range_lat, self.shift_range_lon, self.rotation_range)
        self.dataloader2 = load_test2_data(self.batch_size, self.shift_range_lat, self.shift_range_lon, self.rotation_range)
    def Least_Squares_weight(self, pstA, pstB, weight):
        B, num, _ = pstA.size()

        ws = weight.sum()
        G_A = (pstA * weight).sum(axis=1)/ws
        G_B = (pstB * weight).sum(axis=1)/ws

        Am = (pstA - G_A[:,None,:])*weight
        Bm = (pstB - G_B[:,None,:])*weight

        H = Am.permute(0,2,1) @ Bm
        U, S, V = torch.svd(H)
        # print(U@torch.diag(S[0])@V.permute(0,2,1))
        R = V @ U.permute(0,2,1)
        theta = torch.zeros((B,1),device = pstA.device)
        for i in range(B):
            if torch.det(R[i]) < 0:
                # print("det(R) < R, reflection detected!, correcting for it ...")
                V[i,:,1] *= -1
                R[i] = V[i] @ U[i].T 
            theta[i,0] = torch.arccos((torch.trace(R[i]))/2)*R[i,1,0]/torch.abs(R[i,0,1])

        G_A = G_A.unsqueeze(-1)
        G_B = G_B.unsqueeze(-1)
        t = -R @ G_A + G_B
        return theta, t[:,0], t[:,1]

    def benchmark1(self, model, model_name = None):
        with torch.no_grad():
            pred_shifts = []
            pred_headings = []
            
            gt_shifts = []
            gt_headings = []
            
            # RANSAC_E = RANSAC(0.5)
            # LS_weight = Least_Squares_weight(50)
            #'epe', '5px', '15px', '25px', '50px'
            test_met = [0,0,0,0,0]

            start_time = time.time()
            for i, data in enumerate(self.dataloader, 0):
                sat_map_gt, sat_map, left_camera_k, grd_left_img, gt_shift_u, gt_shift_v, gt_heading = (
                    data["sat_map_gt"].cuda(),
                    data["sat_map"].cuda(),
                    data["left_camera_k"].cuda(),
                    data["grd_left_img"].cuda(),
                    data["gt_u"].cuda(),
                    data["gt_v"].cuda(),
                    data["gt_heading"].cuda(),
                )

                s_gt_u = gt_shift_u * self.shift_range_lon
                s_gt_v = gt_shift_v * self.shift_range_lat
                s_gt_heading = gt_heading * self.rotation_range
                
                warp, certainty, mask = model.BEVmatch(grd_left_img, sat_map, left_camera_k, batched=False)#, left_camera_k, gt_shift_u, gt_shift_v, gt_heading, end2end=0)
                # kpts = torch.stack((W/2 * (coords[...,0]+1), H/2 * (coords[...,1]+1)),axis=-1)
                #warp [A, A, 4] between [-1,1], 
                #certainty: [A, A]
                #mask: [B, 1, A, A], between [0, H-1] and [0, W-1]
                           
                B, _, ori_grdH, ori_grdW = grd_left_img.shape
                _,_,sat_H,sat_W = sat_map_gt.size()
                
                #grid
                #coords0 = coords_grid(mask.size()[0], mask.size()[2], mask.size()[3], device=mask.device)
                coords0 = warp[...,:2][None] #[1, A, A, 2]
                coords1 = F.grid_sample(warp[...,-2:].permute(2,0,1)[None], warp[...,:2].view(-1, 2)[None][None], align_corners = False, mode = "bilinear")[0,:,0].mT.view(mask.size()[2], mask.size()[3], 2)[None] #[1, A, A,2]
                cert_A_to_B = certainty[None,:,:,None] #[1,A,A,1]
                
                ptsA = coords0[mask.permute(0,2,3,1).repeat(1,1,1,2)].view(1, -1, 2).detach() #[1,N,2]
                ptsB = coords1[mask.permute(0,2,3,1).repeat(1,1,1,2)].view(1, -1, 2).detach() #[1,N,2]
                ptsA = torch.stack((sat_W/2 * (ptsA[...,0]+1), sat_H/2 * (ptsA[...,1]+1)),axis=-1)
                ptsB = torch.stack((sat_W/2 * (ptsB[...,0]+1), sat_H/2 * (ptsB[...,1]+1)),axis=-1)
                cert_A_to_B = cert_A_to_B[mask.permute(0,2,3,1)].view(1, -1, 1).detach() #[1,N,1]

                #sample
                roma_matches, certainty = model.sample(warp, certainty, num=10000)
                #ptsA, ptsB = model.to_pixel_coordinates(roma_matches, sat_H, sat_W, sat_H, sat_W) 
                #ptsA = ptsA[None]#[1, Num, 2]
                #ptsB = ptsB[None]#[1, Num, 2]
                #cert_A_to_B = certainty[None,:,None]#[1, Num, 1]
                
                #superpoint
                #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                #feature_conf = extract_features.confs['superpoint_max']
                #Model = dynamic_load(extractors, feature_conf["model"]["name"])
                #sp_model = Model(feature_conf["model"]).eval().to(device)
                #pred1 = sp_model({"image": convert_to_cv_style(grd_left_img)[0].float().unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)})
                #sp_kpts1 = pred1['keypoints'][0] #[Num, 2]
                #pred2 = sp_model({"image": convert_to_cv_style(sat_map)[0].float().unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)})
                #sp_kpts2 = pred2['keypoints'][0] #[Num, 2]
                #sp_kpts1[:,0] /= float(ori_grdW)
                #sp_kpts1[:,1] /= float(ori_grdH)
                #sp_kpts2[:,0] /= float(sat_W)
                #sp_kpts2[:,1] /= float(sat_H)
                # finally, convert from range (0,1) to (-1,1)
                #sp_kpts1 = (sp_kpts1 * 2.) - 1.
                #sp_kpts2 = (sp_kpts2 * 2.) - 1.
                #sp_kpts1_proj = match(mask, gt_heading, gt_shift_u, gt_shift_v, sp_kpts1)
                #sp_kpts1_proj = sp_kpts1_proj[0]

                #sp_inds1, sp_inds2, sp_matchcerts = model.match_keypoints2(sp_kpts1_proj, sp_kpts2, warp, certainty)
                #val_inds1, val_inds2, val_certs = matches(sp_inds1, sp_inds2, sp_matchcerts, sp_kpts1, sp_kpts2)
                #val_inds1 = torch.Tensor(val_inds1).long().to(sp_kpts1_proj.device)
                #val_inds2 = torch.Tensor(val_inds2).long().to(sp_kpts2.device)

                #ptsA = model._to_pixel_coordinates(sp_kpts1_proj[val_inds1], sat_H, sat_W) 
                #ptsB = model._to_pixel_coordinates(sp_kpts2[val_inds2], sat_H, sat_W)
                #ptsA = ptsA[None]
                #ptsB = ptsB[None]
                #cert_A_to_B = torch.tensor(val_certs).unsqueeze(0).unsqueeze(-1).to(ptsA.device)

                pre_theta1, pre_u1, pre_v1 = self.Least_Squares_weight(ptsA, ptsB, cert_A_to_B)

                edu_matrix = rt2edu_matrix(pre_theta1, pre_u1, pre_v1)
                R = edu_matrix * torch.tensor([[[1,1,0],[1,1,0],[0,0,1]]], device=mask.device)
                try:
                    rol_center = torch.tensor([[[1,0,-sat_H/2],[0,1,-sat_W/2],[0,0,1]]], device=mask.device).repeat(B,1,1)
                    T1= torch.inverse(rol_center)@torch.inverse(R)@rol_center@edu_matrix
                    pre_theta = -pre_theta1/3.14*180
                    pre_u = T1[:,0,2][:,None]*LDFF_utils.get_meter_per_pixel()
                    pre_v = -T1[:,1,2][:,None]*LDFF_utils.get_meter_per_pixel()
                except:
                    pre_theta = torch.Tensor([0.]).unsqueeze(0).to(ptsA.device)
                    pre_u = torch.Tensor([0.]).unsqueeze(0).to(ptsA.device)
                    pre_v = torch.Tensor([0.]).unsqueeze(0).to(ptsA.device)
                #print(pre_theta, s_gt_heading, pre_u, pre_v, s_gt_u, s_gt_v)
                shifts = torch.cat([pre_v, pre_u], dim=-1)
                gt_shift = torch.cat([s_gt_v, s_gt_u], dim=-1)
                pred_shifts.append(shifts.data.cpu().numpy())
                gt_shifts.append(gt_shift.data.cpu().numpy())

                pred_headings.append(pre_theta.data.cpu().numpy())
                gt_headings.append(s_gt_heading.data.cpu().numpy())
        end_time = time.time()
        duration = (end_time - start_time)/len(self.dataloader)
        pred_shifts = np.concatenate(pred_shifts, axis=0)
        pred_headings = np.concatenate(pred_headings, axis=0)
        gt_shifts = np.concatenate(gt_shifts, axis=0)
        gt_headings = np.concatenate(gt_headings, axis=0)

        distance = np.sqrt(np.sum((pred_shifts - gt_shifts) ** 2, axis=1))
        angle_diff = np.remainder(np.abs(pred_headings - gt_headings), 360)
        idx0 = angle_diff > 180
        angle_diff[idx0] = 360 - angle_diff[idx0]

        init_dis = np.sqrt(np.sum(gt_shifts ** 2, axis=1))
        init_angle = np.abs(gt_headings)

        metrics = [1, 3, 5]
        angles = [1, 3, 5]
    
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        file_name = self.save_path+"/Test1_results.txt"
        f = open(os.path.join(file_name), 'a')
        f.write('====================================\n')
        f.write('Time per image (second): ' + str(duration) + '\n')
        f.write('Validation results:' + '\n')
        f.write('Init distance average: ' + str(np.mean(init_dis)) + '\n')
        f.write('Pred distance average: ' + str(np.mean(distance)) + '\n')
        f.write('Pred distance median: ' + str(np.median(distance)) + '\n')
        f.write('Init angle average: ' + str(np.mean(init_angle)) + '\n')
        f.write('Pred angle average: ' + str(np.mean(angle_diff)) + '\n')
        f.write('Pred angle median: ' + str(np.median(angle_diff)) + '\n')
        print('====================================')
        print('Time per image (second): ' + str(duration) + '\n')
        print('Validation results:')
        print('Init distance average: ', np.mean(init_dis))
        print('Pred distance average: ', np.mean(distance))
        print('Pred distance median: ', np.median(distance))
        print('Init angle average: ', np.mean(init_angle))
        print('Pred angle average: ', np.mean(angle_diff))
        print('Pred angle median: ', np.median(angle_diff))

    
        for idx in range(len(metrics)):
            pred = np.sum(distance < metrics[idx]) / distance.shape[0] * 100
            init = np.sum(init_dis < metrics[idx]) / init_dis.shape[0] * 100

            line = 'distance within ' + str(metrics[idx]) + ' meters (pred, init): ' + str(pred) + ' ' + str(init)
            print(line)
            f.write(line + '\n')
        print('-------------------------')
        f.write('------------------------\n')

        diff_shifts = np.abs(pred_shifts - gt_shifts)
        for idx in range(len(metrics)):
            pred = np.sum(diff_shifts[:, 0] < metrics[idx]) / diff_shifts.shape[0] * 100
            init = np.sum(np.abs(gt_shifts[:, 0]) < metrics[idx]) / init_dis.shape[0] * 100

            line = 'lateral      within ' + str(metrics[idx]) + ' meters (pred, init): ' + str(pred) + ' ' + str(init)
            print(line)
            f.write(line + '\n')

            pred = np.sum(diff_shifts[:, 1] < metrics[idx]) / diff_shifts.shape[0] * 100
            init = np.sum(np.abs(gt_shifts[:, 1]) < metrics[idx]) / diff_shifts.shape[0] * 100

            line = 'longitudinal within ' + str(metrics[idx]) + ' meters (pred, init): ' + str(pred) + ' ' + str(init)
            print(line)
            f.write(line + '\n')

        print('-------------------------')
        f.write('------------------------\n')
        for idx in range(len(angles)):
            pred = np.sum(angle_diff < angles[idx]) / angle_diff.shape[0] * 100
            init = np.sum(init_angle < angles[idx]) / angle_diff.shape[0] * 100
            line = 'angle within ' + str(angles[idx]) + ' degrees (pred, init): ' + str(pred) + ' ' + str(init)
            print(line)
            f.write(line + '\n')

        print('-------------------------')
        f.write('------------------------\n')

        for idx in range(len(angles)):
            pred = np.sum((angle_diff[:, 0] < angles[idx]) & (diff_shifts[:, 0] < metrics[idx])) / angle_diff.shape[0] * 100
            init = np.sum((init_angle[:, 0] < angles[idx]) & (np.abs(gt_shifts[:, 0]) < metrics[idx])) / angle_diff.shape[0] * 100
            line = 'lat within ' + str(metrics[idx]) + ' & angle within ' + str(angles[idx]) + \
                ' (pred, init): ' + str(pred) + ' ' + str(init)
            print(line)
            f.write(line + '\n')

        print('====================================')
        f.write('====================================\n')
        f.close()
        
    def benchmark2(self, model, model_name = None):
        with torch.no_grad():
            pred_shifts = []
            pred_headings = []
            
            gt_shifts = []
            gt_headings = []
            
            # RANSAC_E = RANSAC(0.5)
            # LS_weight = Least_Squares_weight(50)
            #'epe', '5px', '15px', '25px', '50px'
            test_met = [0,0,0,0,0]

            start_time = time.time()
            for i, data in enumerate(self.dataloader2, 0):
                sat_map_gt, sat_map, left_camera_k, grd_left_img, gt_shift_u, gt_shift_v, gt_heading = (
                    data["sat_map_gt"].cuda(),
                    data["sat_map"].cuda(),
                    data["left_camera_k"].cuda(),
                    data["grd_left_img"].cuda(),
                    data["gt_u"].cuda(),
                    data["gt_v"].cuda(),
                    data["gt_heading"].cuda(),
                )
                #sat_map_gt, sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading, gt_depth = [item.cuda() for item in data[:-1]]

                s_gt_u = gt_shift_u * self.shift_range_lon
                s_gt_v = gt_shift_v * self.shift_range_lat
                s_gt_heading = gt_heading * self.rotation_range
                
                warp, certainty, mask = model.BEVmatch(grd_left_img, sat_map, left_camera_k, batched=False)#, left_camera_k, gt_shift_u, gt_shift_v, gt_heading, end2end=0)
                # kpts = torch.stack((W/2 * (coords[...,0]+1), H/2 * (coords[...,1]+1)),axis=-1)
                #warp [A, A, 4] between [-1,1], 
                #certainty: [A, A]
                #mask: [B, 1, A, A], between [0, H-1] and [0, W-1]
                           
                B, _, ori_grdH, ori_grdW = grd_left_img.shape
                _,_,sat_H,sat_W = sat_map_gt.size()
                
                #grid
                #coords0 = coords_grid(mask.size()[0], mask.size()[2], mask.size()[3], device=mask.device)
                coords0 = warp[...,:2][None] #[1, A, A, 2]
                coords1 = F.grid_sample(warp[...,-2:].permute(2,0,1)[None], warp[...,:2].view(-1, 2)[None][None], align_corners = False, mode = "bilinear")[0,:,0].mT.view(mask.size()[2], mask.size()[3], 2)[None] #[1, A, A,2]
                cert_A_to_B = certainty[None,:,:,None] #[1,A,A,1]
                
                ptsA = coords0[mask.permute(0,2,3,1).repeat(1,1,1,2)].view(1, -1, 2).detach() #[1,N,2]
                ptsB = coords1[mask.permute(0,2,3,1).repeat(1,1,1,2)].view(1, -1, 2).detach() #[1,N,2]
                ptsA = torch.stack((sat_W/2 * (ptsA[...,0]+1), sat_H/2 * (ptsA[...,1]+1)),axis=-1)
                ptsB = torch.stack((sat_W/2 * (ptsB[...,0]+1), sat_H/2 * (ptsB[...,1]+1)),axis=-1)
                cert_A_to_B = cert_A_to_B[mask.permute(0,2,3,1)].view(1, -1, 1).detach() #[1,N,1]

                #sample
                #roma_matches, certainty = model.sample(warp, certainty, num=10000)
                #ptsA, ptsB = model.to_pixel_coordinates(roma_matches, sat_H,sat_W, sat_H,sat_W)   
                #ptsA = ptsA[None]#[1, Num, 2]
                #ptsB = ptsB[None]#[1, Num, 2]
                #cert_A_to_B = certainty[None,:,None]#[1, Num, 1]
                #superpoint
                #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                #feature_conf = extract_features.confs['superpoint_max']
                #Model = dynamic_load(extractors, feature_conf["model"]["name"])
                #sp_model = Model(feature_conf["model"]).eval().to(device)
                #pred1 = sp_model({"image": convert_to_cv_style(grd_left_img)[0].float().unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)})
                #sp_kpts1 = pred1['keypoints'][0] #[Num, 2]
                #pred2 = sp_model({"image": convert_to_cv_style(sat_map)[0].float().unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)})
                #sp_kpts2 = pred2['keypoints'][0] #[Num, 2]
                #sp_kpts1[:,0] /= float(ori_grdW)
                #sp_kpts1[:,1] /= float(ori_grdH)
                #sp_kpts2[:,0] /= float(sat_W)
                #sp_kpts2[:,1] /= float(sat_H)
                # finally, convert from range (0,1) to (-1,1)
                #sp_kpts1 = (sp_kpts1 * 2.) - 1.
                #sp_kpts2 = (sp_kpts2 * 2.) - 1.
                #sp_kpts1_proj = match(mask, gt_heading, gt_shift_u, gt_shift_v, sp_kpts1)
                #sp_kpts1_proj = sp_kpts1_proj[0]
                #sp_inds1, sp_inds2, sp_matchcerts = model.match_keypoints2(sp_kpts1_proj, sp_kpts2, warp, certainty)
                #val_inds1, val_inds2, val_certs = matches(sp_inds1, sp_inds2, sp_matchcerts, sp_kpts1, sp_kpts2)
                #val_inds1 = torch.Tensor(val_inds1).long().to(sp_kpts1_proj.device)
                #val_inds2 = torch.Tensor(val_inds2).long().to(sp_kpts2.device)

                #ptsA = model._to_pixel_coordinates(sp_kpts1_proj[val_inds1], sat_H, sat_W) 
                #ptsB = model._to_pixel_coordinates(sp_kpts2[val_inds2], sat_H, sat_W)
                #ptsA = ptsA[None]
                #ptsB = ptsB[None]
                #cert_A_to_B = torch.tensor(val_certs).unsqueeze(0).unsqueeze(-1).to(ptsA.device)

                pre_theta1, pre_u1, pre_v1 = self.Least_Squares_weight(ptsA, ptsB, cert_A_to_B)

                edu_matrix = rt2edu_matrix(pre_theta1, pre_u1, pre_v1)
                R = edu_matrix * torch.tensor([[[1,1,0],[1,1,0],[0,0,1]]], device=mask.device)
                try:
                    rol_center = torch.tensor([[[1,0,-sat_H/2],[0,1,-sat_W/2],[0,0,1]]], device=mask.device).repeat(B,1,1)
                    T1= torch.inverse(rol_center)@torch.inverse(R)@rol_center@edu_matrix
                    pre_theta = -pre_theta1/3.14*180
                    pre_u = T1[:,0,2][:,None]*LDFF_utils.get_meter_per_pixel()
                    pre_v = -T1[:,1,2][:,None]*LDFF_utils.get_meter_per_pixel()
                except:
                    pre_theta = torch.Tensor([0.]).unsqueeze(0).to(ptsA.device)
                    pre_u = torch.Tensor([0.]).unsqueeze(0).to(ptsA.device)
                    pre_v = torch.Tensor([0.]).unsqueeze(0).to(ptsA.device)

                #print(pre_theta, s_gt_heading, pre_u, pre_v, s_gt_u, s_gt_v)
                shifts = torch.cat([pre_v, pre_u], dim=-1)
                gt_shift = torch.cat([s_gt_v, s_gt_u], dim=-1)
                pred_shifts.append(shifts.data.cpu().numpy())
                gt_shifts.append(gt_shift.data.cpu().numpy())

                pred_headings.append(pre_theta.data.cpu().numpy())
                gt_headings.append(s_gt_heading.data.cpu().numpy())
        end_time = time.time()
        duration = (end_time - start_time)/len(self.dataloader)
        pred_shifts = np.concatenate(pred_shifts, axis=0)
        pred_headings = np.concatenate(pred_headings, axis=0)
        gt_shifts = np.concatenate(gt_shifts, axis=0)
        gt_headings = np.concatenate(gt_headings, axis=0)

        distance = np.sqrt(np.sum((pred_shifts - gt_shifts) ** 2, axis=1))
        angle_diff = np.remainder(np.abs(pred_headings - gt_headings), 360)
        idx0 = angle_diff > 180
        angle_diff[idx0] = 360 - angle_diff[idx0]

        init_dis = np.sqrt(np.sum(gt_shifts ** 2, axis=1))
        init_angle = np.abs(gt_headings)

        metrics = [1, 3, 5]
        angles = [1, 3, 5]
    
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        file_name = self.save_path+"/Test2_results.txt"
        f = open(os.path.join(file_name), 'a')
        f.write('====================================\n')
        f.write('Time per image (second): ' + str(duration) + '\n')
        f.write('Validation results:' + '\n')
        f.write('Init distance average: ' + str(np.mean(init_dis)) + '\n')
        f.write('Pred distance average: ' + str(np.mean(distance)) + '\n')
        f.write('Pred distance median: ' + str(np.median(distance)) + '\n')
        f.write('Init angle average: ' + str(np.mean(init_angle)) + '\n')
        f.write('Pred angle average: ' + str(np.mean(angle_diff)) + '\n')
        f.write('Pred angle median: ' + str(np.median(angle_diff)) + '\n')
        print('====================================')
        print('Time per image (second): ' + str(duration) + '\n')
        print('Validation results:')
        print('Init distance average: ', np.mean(init_dis))
        print('Pred distance average: ', np.mean(distance))
        print('Pred distance median: ', np.median(distance))
        print('Init angle average: ', np.mean(init_angle))
        print('Pred angle average: ', np.mean(angle_diff))
        print('Pred angle median: ', np.median(angle_diff))

    
        for idx in range(len(metrics)):
            pred = np.sum(distance < metrics[idx]) / distance.shape[0] * 100
            init = np.sum(init_dis < metrics[idx]) / init_dis.shape[0] * 100

            line = 'distance within ' + str(metrics[idx]) + ' meters (pred, init): ' + str(pred) + ' ' + str(init)
            print(line)
            f.write(line + '\n')
        print('-------------------------')
        f.write('------------------------\n')

        diff_shifts = np.abs(pred_shifts - gt_shifts)
        for idx in range(len(metrics)):
            pred = np.sum(diff_shifts[:, 0] < metrics[idx]) / diff_shifts.shape[0] * 100
            init = np.sum(np.abs(gt_shifts[:, 0]) < metrics[idx]) / init_dis.shape[0] * 100

            line = 'lateral      within ' + str(metrics[idx]) + ' meters (pred, init): ' + str(pred) + ' ' + str(init)
            print(line)
            f.write(line + '\n')

            pred = np.sum(diff_shifts[:, 1] < metrics[idx]) / diff_shifts.shape[0] * 100
            init = np.sum(np.abs(gt_shifts[:, 1]) < metrics[idx]) / diff_shifts.shape[0] * 100

            line = 'longitudinal within ' + str(metrics[idx]) + ' meters (pred, init): ' + str(pred) + ' ' + str(init)
            print(line)
            f.write(line + '\n')

        print('-------------------------')
        f.write('------------------------\n')
        for idx in range(len(angles)):
            pred = np.sum(angle_diff < angles[idx]) / angle_diff.shape[0] * 100
            init = np.sum(init_angle < angles[idx]) / angle_diff.shape[0] * 100
            line = 'angle within ' + str(angles[idx]) + ' degrees (pred, init): ' + str(pred) + ' ' + str(init)
            print(line)
            f.write(line + '\n')

        print('-------------------------')
        f.write('------------------------\n')

        for idx in range(len(angles)):
            pred = np.sum((angle_diff[:, 0] < angles[idx]) & (diff_shifts[:, 0] < metrics[idx])) / angle_diff.shape[0] * 100
            init = np.sum((init_angle[:, 0] < angles[idx]) & (np.abs(gt_shifts[:, 0]) < metrics[idx])) / angle_diff.shape[0] * 100
            line = 'lat within ' + str(metrics[idx]) + ' & angle within ' + str(angles[idx]) + \
                ' (pred, init): ' + str(pred) + ' ' + str(init)
            print(line)
            f.write(line + '\n')

        print('====================================')
        f.write('====================================\n')
        f.close()

class KittiBenchmark_new:
    def __init__(self, shift_range_lat=20, shift_range_lon=20, rotation_range=10, dpp=1, save_path = "./results/kitti") -> None:
        self.shift_range_lat = shift_range_lat
        self.shift_range_lon = shift_range_lon
        self.rotation_range = rotation_range
        self.dpp = dpp
        self.save_path = save_path
        self.dataloader = load_test1_data(1, self.shift_range_lat, self.shift_range_lon, self.rotation_range)
        self.dataloader2 = load_test2_data(1, self.shift_range_lat, self.shift_range_lon, self.rotation_range)
    def Least_Squares_weight(self, pstA, pstB, weight):
        B, num, _ = pstA.size()

        ws = weight.sum()
        G_A = (pstA * weight).sum(axis=1)/ws
        G_B = (pstB * weight).sum(axis=1)/ws

        Am = (pstA - G_A[:,None,:])*weight
        Bm = (pstB - G_B[:,None,:])*weight

        H = Am.permute(0,2,1) @ Bm

        U, S, V = torch.svd(H)
        # print(U@torch.diag(S[0])@V.permute(0,2,1))
        R = V @ U.permute(0,2,1)
        theta = torch.zeros((B,1),device = pstA.device)
        for i in range(B):
            if torch.det(R[i]) < 0:
                # print("det(R) < R, reflection detected!, correcting for it ...")
                V[i,:,1] *= -1
                R[i] = V[i] @ U[i].T 
            #pdb.set_trace()
            theta[i,0] = torch.arccos(torch.clamp(torch.trace(R[i]) / 2, min=-1.0, max=1.0)) * R[i, 1, 0] / (torch.abs(R[i, 0, 1]) + 1e-6)
        
        G_A = G_A.unsqueeze(-1)
        G_B = G_B.unsqueeze(-1)
        t = -R @ G_A + G_B
        return theta, t[:,0], t[:,1]

    def benchmark1(self, model, model_name = None):
        with torch.no_grad():
            pred_shifts = []
            pred_headings = []
            
            gt_shifts = []
            gt_headings = []
            
            # RANSAC_E = RANSAC(0.5)
            # LS_weight = Least_Squares_weight(50)
            #'epe', '5px', '15px', '25px', '50px'
            test_met = [0,0,0,0,0]

            start_time = time.time()
            for i, data in enumerate(self.dataloader, 0):
                #sat_map_gt, sat_map, left_camera_k, grd_left_img, gt_shift_u, gt_shift_v, gt_heading = (
                 #   data["sat_map_gt"].cuda(),
                  #  data["sat_map"].cuda(),
                   # data["left_camera_k"].cuda(),
                   # data["grd_left_img"].cuda(),
                   # data["gt_u"].cuda(),
                   # data["gt_v"].cuda(),
                   # data["gt_heading"].cuda(),
                #)
                data = to_cuda(data)
                s_gt_u = data['gt_u'] * self.shift_range_lon
                s_gt_v = data['gt_v'] * self.shift_range_lat
                s_gt_heading = data["gt_heading"] * self.rotation_range
                
                corresps, mask_pyramid,pre_theta,pre_u, pre_v = model.BEVforward(data, batched=False, end2end=True)#, left_camera_k, gt_shift_u, gt_shift_v, gt_heading, end2end=0)
                # kpts = torch.stack((W/2 * (coords[...,0]+1), H/2 * (coords[...,1]+1)),axis=-1)
                #warp [A, A, 4] between [-1,1], 
                #certainty: [A, A]
                #mask: [B, 1, A, A], between [0, H-1] and [0, W-1]
          
            
                #print(pre_theta, s_gt_heading, pre_u, pre_v, s_gt_u, s_gt_v)
                shifts = torch.cat([pre_v, pre_u], dim=-1)
                gt_shift = torch.cat([s_gt_v, s_gt_u], dim=-1)
                pred_shifts.append(shifts.data.cpu().numpy())
                gt_shifts.append(gt_shift.data.cpu().numpy())

                pred_headings.append(pre_theta.data.cpu().numpy())
                gt_headings.append(s_gt_heading.data.cpu().numpy())
        end_time = time.time()
        duration = (end_time - start_time)/len(self.dataloader)
        pred_shifts = np.concatenate(pred_shifts, axis=0)
        pred_headings = np.concatenate(pred_headings, axis=0)
        gt_shifts = np.concatenate(gt_shifts, axis=0)
        gt_headings = np.concatenate(gt_headings, axis=0)

        distance = np.sqrt(np.sum((pred_shifts - gt_shifts) ** 2, axis=1))
        angle_diff = np.remainder(np.abs(pred_headings - gt_headings), 360)
        idx0 = angle_diff > 180
        angle_diff[idx0] = 360 - angle_diff[idx0]

        metrics = [1, 3, 5]
    
        mean_distance = np.mean(distance)
        median_distance = np.median(distance)
        mean_angle = np.mean(angle_diff)
        median_angle = np.median(angle_diff)
        lats = []
        lons = []
        angles = []
        diff_shifts = np.abs(pred_shifts - gt_shifts)
        for idx in range(len(metrics)):
            lat_pred = np.sum(diff_shifts[:, 0] < metrics[idx]) / diff_shifts.shape[0] * 100
            lats.append(lat_pred)            

            lon_pred = np.sum(diff_shifts[:, 1] < metrics[idx]) / diff_shifts.shape[0] * 100
            lons.append(lon_pred)
        
        for idx in range(len(metrics)):
            angle_pred = np.sum(angle_diff < metrics[idx]) / angle_diff.shape[0] * 100
            angles.append(angle_pred)
        return{
             "mean_distance": mean_distance,
             "median_distance": median_distance,
             "mean_angle": mean_angle,
             "median_angle": median_angle,
             "lat_1": lats[0],
             "lat_3": lats[1],
             "lat_5": lats[2],
             "lon_1": lons[0],
             "lon_3": lons[1],
             "lon_5": lons[2],
             "angle_1": angles[0],
             "angle_3": angles[1],
             "angle_5": angles[2],
        }

    def benchmark2(self, model, model_name = None):
        with torch.no_grad():
            pred_shifts = []
            pred_headings = []
            
            gt_shifts = []
            gt_headings = []
            
            # RANSAC_E = RANSAC(0.5)
            # LS_weight = Least_Squares_weight(50)
            #'epe', '5px', '15px', '25px', '50px'
            test_met = [0,0,0,0,0]

            start_time = time.time()
            for i, data in enumerate(self.dataloader2, 0):
                sat_map_gt, sat_map, left_camera_k, grd_left_img, gt_shift_u, gt_shift_v, gt_heading = (
                    data["sat_map_gt"].cuda(),
                    data["sat_map"].cuda(),
                    data["left_camera_k"].cuda(),
                    data["grd_left_img"].cuda(),
                    data["gt_u"].cuda(),
                    data["gt_v"].cuda(),
                    data["gt_heading"].cuda(),
                )
                #sat_map_gt, sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading, gt_depth = [item.cuda() for item in data[:-1]]

                s_gt_u = gt_shift_u * self.shift_range_lon
                s_gt_v = gt_shift_v * self.shift_range_lat
                s_gt_heading = gt_heading * self.rotation_range
                
                warp, certainty, mask = model.BEVmatch(grd_left_img, sat_map, left_camera_k, batched=False)#, left_camera_k, gt_shift_u, gt_shift_v, gt_heading, end2end=0)
                # kpts = torch.stack((W/2 * (coords[...,0]+1), H/2 * (coords[...,1]+1)),axis=-1)
                #warp [A, A, 4] between [-1,1], 
                #certainty: [A, A]
                #mask: [B, 1, A, A], between [0, H-1] and [0, W-1]
                           
                B, _, ori_grdH, ori_grdW = grd_left_img.shape
                _,_,sat_H,sat_W = sat_map_gt.size()
                
                coords0 = warp[...,:2][None] #[1, A, A, 2]
                coords1 = F.grid_sample(warp[...,-2:].permute(2,0,1)[None], warp[...,:2].view(-1, 2)[None][None], align_corners = False, mode = "bilinear")[0,:,0].mT.view(mask.size()[2], mask.size()[3], 2)[None] #[1, A, A,2]
                cert_A_to_B = certainty[None,:,:,None] #[1,A,A,1]
                
                ptsA = coords0[mask.permute(0,2,3,1).repeat(1,1,1,2)].view(1, -1, 2).detach() #[1,N,2]
                ptsB = coords1[mask.permute(0,2,3,1).repeat(1,1,1,2)].view(1, -1, 2).detach() #[1,N,2]
                ptsA = torch.stack((sat_W/2 * (ptsA[...,0]+1), sat_H/2 * (ptsA[...,1]+1)),axis=-1)
                ptsB = torch.stack((sat_W/2 * (ptsB[...,0]+1), sat_H/2 * (ptsB[...,1]+1)),axis=-1)
                cert_A_to_B = cert_A_to_B[mask.permute(0,2,3,1)].view(1, -1, 1).detach() #[1,N,1]
                
                pre_theta1, pre_u1, pre_v1 = self.Least_Squares_weight(ptsA, ptsB, cert_A_to_B)
                
                edu_matrix = rt2edu_matrix(pre_theta1, pre_u1, pre_v1)
                R = edu_matrix * torch.tensor([[[1,1,0],[1,1,0],[0,0,1]]], device=mask.device)
                rol_center = torch.tensor([[[1,0,-sat_H/2],[0,1,-sat_W/2],[0,0,1]]], device=mask.device).repeat(B,1,1)
                T1= torch.inverse(rol_center)@torch.inverse(R)@rol_center@edu_matrix
                pre_theta = -pre_theta1/3.14*180
                pre_u = T1[:,0,2][:,None]*LDFF_utils.get_meter_per_pixel()
                pre_v = -T1[:,1,2][:,None]*LDFF_utils.get_meter_per_pixel()
                #print(pre_theta, s_gt_heading, pre_u, pre_v, s_gt_u, s_gt_v)
                shifts = torch.cat([pre_v, pre_u], dim=-1)
                gt_shift = torch.cat([s_gt_v, s_gt_u], dim=-1)
                pred_shifts.append(shifts.data.cpu().numpy())
                gt_shifts.append(gt_shift.data.cpu().numpy())

                pred_headings.append(pre_theta.data.cpu().numpy())
                gt_headings.append(s_gt_heading.data.cpu().numpy())
        end_time = time.time()
        duration = (end_time - start_time)/len(self.dataloader)
        pred_shifts = np.concatenate(pred_shifts, axis=0)
        pred_headings = np.concatenate(pred_headings, axis=0)
        gt_shifts = np.concatenate(gt_shifts, axis=0)
        gt_headings = np.concatenate(gt_headings, axis=0)

        distance = np.sqrt(np.sum((pred_shifts - gt_shifts) ** 2, axis=1))
        angle_diff = np.remainder(np.abs(pred_headings - gt_headings), 360)
        idx0 = angle_diff > 180
        angle_diff[idx0] = 360 - angle_diff[idx0]

        init_dis = np.sqrt(np.sum(gt_shifts ** 2, axis=1))
        init_angle = np.abs(gt_headings)

        metrics = [1, 3, 5]
    
        mean_distance = np.mean(distance)
        median_distance = np.median(distance)
        mean_angle = np.mean(angle_diff)
        median_angle = np.mean(angle_diff)
        lats = []
        lons = []
        angles = []
        diff_shifts = np.abs(pred_shifts - gt_shifts)
        for idx in range(len(metrics)):
            lat_pred = np.sum(diff_shifts[:, 0] < metrics[idx]) / diff_shifts.shape[0] * 100
            lats.append(lat_pred)            

            lon_pred = np.sum(diff_shifts[:, 1] < metrics[idx]) / diff_shifts.shape[0] * 100
            lons.append(lon_pred)
        
        for idx in range(len(metrics)):
            angle_pred = np.sum(angle_diff < metrics[idx]) / angle_diff.shape[0] * 100
            angles.append(angle_pred)
        return{
             "test2_mean_distance": mean_distance,
             "test2_median_distance": median_distance,
             "test2_mean_angle": mean_angle,
             "test2_median_angle": median_angle,
             "test2_lat_1": lats[0],
             "test2_lat_3": lats[1],
             "test2_lat_5": lats[2],
             "test2_lon_1": lons[0],
             "test2_lon_3": lons[1],
             "test2_lon_5": lons[2],
             "test2_angle_1": angles[0],
             "test2_angle_3": angles[1],
             "test2_angle_5": angles[2],
        }

class KittiBenchmark_new_end2end:
    def __init__(self, shift_range_lat=20, shift_range_lon=20, rotation_range=10, dpp=1, save_path = "./results/kitti") -> None:
        self.shift_range_lat = shift_range_lat
        self.shift_range_lon = shift_range_lon
        self.rotation_range = rotation_range
        self.dpp = dpp
        self.save_path = save_path
        self.dataloader = load_test1_data(1, self.shift_range_lat, self.shift_range_lon, self.rotation_range)
        self.dataloader2 = load_test2_data(1, self.shift_range_lat, self.shift_range_lon, self.rotation_range)
    def to_cuda(batch):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.cuda()
        return batch
    def benchmark1(self, model, model_name = None):
        with torch.no_grad():
            pred_shifts = []
            pred_headings = []
            
            gt_shifts = []
            gt_headings = []
            
            # RANSAC_E = RANSAC(0.5)
            # LS_weight = Least_Squares_weight(50)
            #'epe', '5px', '15px', '25px', '50px'
            test_met = [0,0,0,0,0]

            start_time = time.time()
            for i, data in enumerate(self.dataloader, 0):
                data = to_cuda(data)
                s_gt_u = data['gt_u'] * self.shift_range_lon
                s_gt_v = data['gt_v'] * self.shift_range_lat
                s_gt_heading = data["gt_heading"] * self.rotation_range
                
                corresps, mask_pyramid,pre_theta,pre_u, pre_v = model.BEVforward(data, batched=False, end2end=True)#, left_camera_k, gt_shift_u, gt_shift_v, gt_heading, end2end=0)
                shifts = torch.cat([pre_v, pre_u], dim=-1)
                gt_shift = torch.cat([s_gt_v, s_gt_u], dim=-1)
                pred_shifts.append(shifts.data.cpu().numpy())
                gt_shifts.append(gt_shift.data.cpu().numpy())

                pred_headings.append(pre_theta.data.cpu().numpy())
                gt_headings.append(s_gt_heading.data.cpu().numpy())
        end_time = time.time()
        duration = (end_time - start_time)/len(self.dataloader)
        pred_shifts = np.concatenate(pred_shifts, axis=0)
        pred_headings = np.concatenate(pred_headings, axis=0)
        gt_shifts = np.concatenate(gt_shifts, axis=0)
        gt_headings = np.concatenate(gt_headings, axis=0)

        distance = np.sqrt(np.sum((pred_shifts - gt_shifts) ** 2, axis=1))
        angle_diff = np.remainder(np.abs(pred_headings - gt_headings), 360)
        idx0 = angle_diff > 180
        angle_diff[idx0] = 360 - angle_diff[idx0]

        metrics = [1, 3, 5]
    
        mean_distance = np.mean(distance)
        median_distance = np.median(distance)
        mean_angle = np.mean(angle_diff)
        median_angle = np.mean(angle_diff)
        lats = []
        lons = []
        angles = []
        diff_shifts = np.abs(pred_shifts - gt_shifts)
        for idx in range(len(metrics)):
            lat_pred = np.sum(diff_shifts[:, 0] < metrics[idx]) / diff_shifts.shape[0] * 100
            lats.append(lat_pred)            

            lon_pred = np.sum(diff_shifts[:, 1] < metrics[idx]) / diff_shifts.shape[0] * 100
            lons.append(lon_pred)
        
        for idx in range(len(metrics)):
            angle_pred = np.sum(angle_diff < metrics[idx]) / angle_diff.shape[0] * 100
            angles.append(angle_pred)
        return{
             "mean_distance": mean_distance,
             "median_distance": median_distance,
             "mean_angle": mean_angle,
             "median_angle": median_angle,
             "lat_1": lats[0],
             "lat_3": lats[1],
             "lat_5": lats[2],
             "lon_1": lons[0],
             "lon_3": lons[1],
             "lon_5": lons[2],
             "angle_1": angles[0],
             "angle_3": angles[1],
             "angle_5": angles[2],
        }

    def benchmark2(self, model, model_name = None):
        with torch.no_grad():
            pred_shifts = []
            pred_headings = []
            
            gt_shifts = []
            gt_headings = []
            
            # RANSAC_E = RANSAC(0.5)
            # LS_weight = Least_Squares_weight(50)
            #'epe', '5px', '15px', '25px', '50px'
            test_met = [0,0,0,0,0]

            start_time = time.time()
            for i, data in enumerate(self.dataloader2, 0):
                s_gt_u = data['gt_u'] * self.shift_range_lon
                s_gt_v = data['gt_v'] * self.shift_range_lat
                s_gt_heading = data["gt_heading"] * self.rotation_range
                
                corresps, mask_pyramid, pre_theta,pre_u, pre_v = model.BEVforward(data, batched=False, end2end=True)#, left_camera_k, gt_shift_u, gt_shift_v, gt_heading, end2end=0)
                
                #print(pre_theta, s_gt_heading, pre_u, pre_v, s_gt_u, s_gt_v)
                shifts = torch.cat([pre_v, pre_u], dim=-1)
                gt_shift = torch.cat([s_gt_v, s_gt_u], dim=-1)
                pred_shifts.append(shifts.data.cpu().numpy())
                gt_shifts.append(gt_shift.data.cpu().numpy())

                pred_headings.append(pre_theta.data.cpu().numpy())
                gt_headings.append(s_gt_heading.data.cpu().numpy())
        end_time = time.time()
        duration = (end_time - start_time)/len(self.dataloader)
        pred_shifts = np.concatenate(pred_shifts, axis=0)
        pred_headings = np.concatenate(pred_headings, axis=0)
        gt_shifts = np.concatenate(gt_shifts, axis=0)
        gt_headings = np.concatenate(gt_headings, axis=0)

        distance = np.sqrt(np.sum((pred_shifts - gt_shifts) ** 2, axis=1))
        angle_diff = np.remainder(np.abs(pred_headings - gt_headings), 360)
        idx0 = angle_diff > 180
        angle_diff[idx0] = 360 - angle_diff[idx0]

        init_dis = np.sqrt(np.sum(gt_shifts ** 2, axis=1))
        init_angle = np.abs(gt_headings)

        metrics = [1, 3, 5]
    
        mean_distance = np.mean(distance)
        median_distance = np.median(distance)
        mean_angle = np.mean(angle_diff)
        median_angle = np.mean(angle_diff)
        lats = []
        lons = []
        angles = []
        diff_shifts = np.abs(pred_shifts - gt_shifts)
        for idx in range(len(metrics)):
            lat_pred = np.sum(diff_shifts[:, 0] < metrics[idx]) / diff_shifts.shape[0] * 100
            lats.append(lat_pred)            

            lon_pred = np.sum(diff_shifts[:, 1] < metrics[idx]) / diff_shifts.shape[0] * 100
            lons.append(lon_pred)
        
        for idx in range(len(metrics)):
            angle_pred = np.sum(angle_diff < metrics[idx]) / angle_diff.shape[0] * 100
            angles.append(angle_pred)
        return{
             "test2_mean_distance": mean_distance,
             "test2_median_distance": median_distance,
             "test2_mean_angle": mean_angle,
             "test2_median_angle": median_angle,
             "test2_lat_1": lats[0],
             "test2_lat_3": lats[1],
             "test2_lat_5": lats[2],
             "test2_lon_1": lons[0],
             "test2_lon_3": lons[1],
             "test2_lon_5": lons[2],
             "test2_angle_1": angles[0],
             "test2_angle_3": angles[1],
             "test2_angle_5": angles[2],
        }