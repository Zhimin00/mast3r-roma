from mast3r.model import AsymmetricMASt3R, AsymmetricMASt3R_only_warp, AsymmetricMASt3R_warp
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference, make_batch_symmetric
from dust3r.utils.image import load_images
from dust3r.utils.device import collate_with_cat
from PIL import Image
import torch.nn.functional as F
import pdb
import os
from romatch.utils.utils import tensor_to_pil
import cv2
import numpy as np
import torch

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
    kpts = torch.stack((W/2 * (coords[...,0]+1), H/2 * (coords[...,1]+1)),axis=-1)
    return kpts

@torch.no_grad()
def inference(batch, model, device, use_amp=False): 
    # to device. 
    for view in batch:
        for name in ['img', 'true_shape']:
            view[name] = view[name].to(device, non_blocking=True)
    # forward. 
    view1, view2 = batch
    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        return model(view1, view2)
    
if __name__ == '__main__':
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    #"naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"#
    model_name = "/cis/net/r24a/data/zshao/checkpoints/dust3r/MASt3R_freeze_trainwarp/checkpoint-last.pth" #finetune_megadepth-final.pth"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R_warp.from_pretrained(model_name).to(device)
    # im1_path = '/cis/net/r24a/data/zshao/data/wriva_processed_data/cross-view/A01/A01_s07/input/images/image_000001.jpg'
    # im2_path = '/cis/net/r24a/data/zshao/data/wriva_processed_data/cross-view/A01/A01_s07/input/images/image_000004.jpg'
    # output_path = '/cis/net/r24a/data/zshao/data/wriva_processed_data/cross-view/vis_all/A01_s07_2'
    # images = load_images([im1_path, im2_path], size=512)
    # batch = [tuple(images)]
    # batch = collate_with_cat(batch[:1])
    # view1, view2 = make_batch_symmetric(batch)
    # batch = (view1, view2)
    # _, _ , corresps = inference(batch, model, device, use_amp=False)

    # warp, dense_certainty = dense_match(corresps)#, symmetric=False)
    # sparse_matches, sparse_certainty = sample_to_sparse(warp, dense_certainty, 5000)

    # ## warp 
    # x1 = view1['img'][0]
    # x2 = view2['img'][0] #1, 3,H,W
    # _, H, W = x1.shape

    # im2_transfer_rgb = F.grid_sample(
    # x2[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
    # )[0]
    # im1_transfer_rgb = F.grid_sample(
    # x1[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
    # )[0]
    # warp_im = torch.cat((im2_transfer_rgb,im1_transfer_rgb),dim=2)
    # white_im = torch.ones((H,2*W),device=device)
    # vis_im = dense_certainty * warp_im + (1 - dense_certainty) * white_im
    # tensor_to_pil(vis_im, unnormalize=False).save(os.path.join(output_path, f'mast3r_onlywarp_warp.jpg'))
    
    # # draw matches
    # kpts1, kpts2 = to_pixel_coordinates(sparse_matches, H, W, H, W)    
    # _, mask = cv2.findFundamentalMat(
    #     kpts1.cpu().numpy(), kpts2.cpu().numpy(), ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000
    # )

    # canvas = np.zeros((H, 2*W, 3), dtype=np.uint8)
    # im1 = x1 * 0.5 + 0.5
    # im2 = x2 * 0.5 + 0.5
    # im1 = im1.clip(min=0, max=1)
    # im2 = im2.clip(min=0, max=1)
    # im1 = im1.permute(1,2,0).cpu().numpy()
    # im2 = im2.permute(1,2,0).cpu().numpy()
    # pdb.set_trace()
    # canvas[:H, :W] = im1 * 255
    # canvas[:H, W:] = im2 * 255
    # offset = np.array([W, 0])
        
    # # Draw matches
    # for i, v in enumerate(mask):
    #     if v[0] == 1:  # Valid match
    #         pt1 = kpts1[i].cpu().numpy().astype('int32')  # Keypoint in img1
    #         pt2 = kpts2[i].cpu().numpy().astype('int32') + offset  # Keypoint in img2 with offset
    #         # Draw circles at the keypoints
    #         cv2.circle(canvas, tuple(pt1), 2, (0, 0, 255), -1)  # Red for img1
    #         cv2.circle(canvas, tuple(pt2), 2, (0, 0, 255), -1)  # Red for img2
    #         # Draw line connecting the keypoints
    #         cv2.line(canvas, tuple(pt1), tuple(pt2), (0, 255, 0), 2)  # Green line

    # cv2.imwrite(output_path +f'/mast3r_onlywarp.png', canvas)

    im1_path = '/cis/net/r24a/data/zshao/data/wriva_processed_data/cross-view/BLH0001/input/images/image_000075.JPG'
    im2_path = '/cis/net/r24a/data/zshao/data/wriva_processed_data/cross-view/BLH0001/input/images/image_000003.JPG'
    output_path = '/cis/net/r24a/data/zshao/data/wriva_processed_data/cross-view/vis_all/BLH0001'
    images = load_images([im1_path, im2_path], size=512)
    batch = [tuple(images)]
    batch = collate_with_cat(batch[:1])
    view1, view2 = batch
    _, _ , corresps = inference(batch, model, device, use_amp=False)
    warp1, dense_certainty1 = dense_match(corresps, symmetric=False)
    sparse_matches1, sparse_certainty1 = sample_to_sparse(warp1, dense_certainty1, 5000)

    images2 = load_images([im2_path, im1_path], size=512)
    batch2 = [tuple(images2)]
    batch2 = collate_with_cat(batch2[:1])
    _, _ , corresps2 = inference(batch2, model, device, use_amp=False)
    warp2, dense_certainty2 = dense_match(corresps2, symmetric=False)
    sparse_matches2, sparse_certainty2 = sample_to_sparse(warp2, dense_certainty2, 5000)

    ## warp 
    x1 = view1['img'][0]
    x2 = view2['img'][0] #1, 3,H,W
    _, H1, W1 = x1.shape
    _, H2, W2 = x2.shape

    im2_transfer_rgb = F.grid_sample(
    x2[None], warp1[:,:, 2:][None], mode="bilinear", align_corners=False
    )[0]
    # warp_im = torch.cat((im2_transfer_rgb,im1_transfer_rgb),dim=2)
    warp_im = im2_transfer_rgb
    white_im = torch.ones((H1,W1),device=device)
    vis_im = dense_certainty1 * warp_im + (1 - dense_certainty1) * white_im
    tensor_to_pil(vis_im, unnormalize=False).save(os.path.join(output_path, f'mast3r_onlywarp_warp1.jpg'))
    
    im1_transfer_rgb = F.grid_sample(
    x1[None], warp2[:, :, 2:][None], mode="bilinear", align_corners=False
    )[0]
    # warp_im = torch.cat((im2_transfer_rgb,im1_transfer_rgb),dim=2)
    warp_im2 = im1_transfer_rgb
    white_im2 = torch.ones((H2,W2),device=device)
    vis_im2 = dense_certainty2 * warp_im2 + (1 - dense_certainty2) * white_im2
    tensor_to_pil(vis_im2, unnormalize=False).save(os.path.join(output_path, f'mast3r_onlywarp_warp2.jpg'))

    # draw matches
    kpts1, kpts2 = to_pixel_coordinates(sparse_matches1, H1, W1, H2, W2) 
    kpts2_, kpts1_ = to_pixel_coordinates(sparse_matches2, H2, W2, H1, W1)  
    kpts1 = torch.cat([kpts1, kpts1_], dim=0)
    kpts2 = torch.cat([kpts2, kpts2_], dim=0)
    _, mask = cv2.findFundamentalMat(
        kpts1.cpu().numpy(), kpts2.cpu().numpy(), ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000
    )
        
    canvas = np.zeros((max(H1, H2), W1 + W2, 3), dtype=np.uint8)
    im1 = x1 * 0.5 + 0.5
    im2 = x2 * 0.5 + 0.5
    im1 = im1.clip(min=0, max=1)
    im2 = im2.clip(min=0, max=1)
    im1 = im1.permute(1,2,0).cpu().numpy()
    im2 = im2.permute(1,2,0).cpu().numpy()
    pdb.set_trace()
    canvas[:H1, :W1] = im1 * 255
    canvas[:H2, W1:] = im2 * 255
    offset = np.array([W1, 0])
        
    # Draw matches
    for i, v in enumerate(mask):
        if v[0] == 1:  # Valid match
            pt1 = kpts1[i].cpu().numpy().astype('int32')  # Keypoint in img1
            pt2 = kpts2[i].cpu().numpy().astype('int32') + offset  # Keypoint in img2 with offset
            # Draw circles at the keypoints
            cv2.circle(canvas, tuple(pt1), 2, (0, 0, 255), -1)  # Red for img1
            cv2.circle(canvas, tuple(pt2), 2, (0, 0, 255), -1)  # Red for img2
            # Draw line connecting the keypoints
            cv2.line(canvas, tuple(pt1), tuple(pt2), (0, 255, 0), 2)  # Green line

    cv2.imwrite(output_path +f'/mast3r_onlywarp.png', canvas)

    # view1, view2 = images
    # _, _ , corresps1 = model(view1, view2)

    # warp1, dense_certainty1 = dense_match(corresps1, symmetric = False)
    # sparse_matches1, sparse_certainty1 = sample_to_sparse(warp1, dense_certainty1, 5000)

    # _, _, corresps2 = model(view2, view1)
    # warp2, dense_certainty2 = dense_match(corresps2, symmetric = False)
    # sparse_matches2, sparse_certainty2 = sample_to_sparse(warp2, dense_certainty2, 5000)
    
    # im1 = view1['img'][0]
    # im2 = view2['img'][0] #3,H,W

    # pdb.set_trace()
    # x2 = (torch.tensor(np.array(im2)) / 255).to(device).permute(2, 0, 1)
    # im2_transfer_rgb = F.grid_sample(
    # x2[None], warp1[:,:, 2:][None], mode="bilinear", align_corners=False
    # )[0]
    
    # x1 = (torch.tensor(np.array(im1)) / 255).to(device).permute(2, 0, 1)
    # im1_transfer_rgb = F.grid_sample(
    # x1[None], warp2[:,:, 2:][None], mode="bilinear", align_corners=False
    # )[0]
    # H0, W0 = view1['true_shape'][0]
    # warp_im = torch.cat((im2_transfer_rgb,im1_transfer_rgb),dim=2)
    # white_im = torch.ones((H0,W0),device=device)
    # vis_im1 = im2_transfer_rgb * dense_certainty1 + (1-dense_certainty1) * white_im
    # vis_im2 = im1_transfer_rgb * dense_certainty2 + (1-dense_certainty2) * white_im
    # vis_im = certainty * warp_im + (1 - certainty) * white_im
    # tensor_to_pil(vis_im, unnormalize=False).save(os.path.join(output_path, f'{name}_warp.jpg'))
    # print(f"Finished: {name}_warp")