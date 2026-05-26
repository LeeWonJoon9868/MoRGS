# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
import glob
import os
import json
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

from torchvision.transforms import InterpolationMode

from natsort import natsorted


class MultiViewVideoDataset(Dataset):
    def __init__(
        self,
        datadir,
        split,
        test_indices,
        max_frames=300,
        start_idx=0,
        img_format='png',
        verbose=False,
        resolution=1
    ):
        videos = glob.glob(os.path.join(datadir, "cam*"))
        videos = sorted(videos)
        self.n_cams = 0
        self.n_frames = None
        self.image_transform = transforms.ToTensor()
        self.image_resize = None
        self.image_paths = []
        self.resolution = resolution
        self.mask_paths = []
        # For Google Immersive some cameras might not exist in models.json, remove them
        if os.path.exists(os.path.join(datadir,"models.json")):
            import json 
            with open(os.path.join(datadir, "models.json"), "r") as f:
                meta = json.load(f)
            camera_names_meta = [camera['name'] for camera in meta]
            camera_names_images = sorted([os.path.basename(video) for video in videos])
            filtered_videos = []
            for i,camera_name_image in enumerate(camera_names_images):
                if camera_name_image in camera_names_meta:
                    filtered_videos.append(os.path.join(datadir, camera_name_image))
            videos = sorted(filtered_videos)
            
        # if "coffee_martini" in datadir:
        #     # Remove unsynchronized camera "cam12"
        #     filtered_videos = []
        #     for i, video in enumerate(videos):
        #         if os.path.basename(video) != "cam13":
        #             filtered_videos.append(video)
        #     videos = sorted(filtered_videos)
        #     print(f"MultiViewVideoDataset::__init__(): removed unsynchronized cam13, remaining {len(videos)} views")
    
        if verbose:
            print(f"MultiViewVideoDataset::__init__(): parsed {len(videos)} videos (sequences)")

        for idx, video in enumerate(videos):
            if (idx in test_indices and split == 'test') or \
                (idx not in test_indices and split=='train'):
                self.n_cams += 1
                if os.path.exists(os.path.join(datadir,"models.json")):
                    #image_list = sorted(glob.glob(os.path.join(video,'images_scaled_2','*.png')))
                    image_list = sorted(glob.glob(os.path.join(video,'*.png')))
                else:
                    #image_list = sorted(glob.glob(os.path.join(video,'images','*.png')))
                    image_list = sorted(glob.glob(os.path.join(video,'*.png')))
                image_list = image_list[start_idx:start_idx+max_frames]
                
                if "meetroom" in video:
                    image_list = natsorted(glob.glob(os.path.join(video, '*.png')))
                
                cam_name = os.path.basename(video)
                mask_list = [None] * len(image_list)   
                if "coffee_martini" in datadir and cam_name == "cam13":
                    _all_masks = sorted(glob.glob(os.path.join(datadir,'mask_cam13', '*.png')))
                    # frame 슬라이싱 범위와 맞춤
                    _all_masks = _all_masks[start_idx:start_idx+len(image_list)]
                    # 길이가 모자라면 None으로 패딩
                    if len(_all_masks) < len(image_list):
                        _all_masks += [None] * (len(image_list) - len(_all_masks))
                    mask_list = _all_masks
                
                if verbose and idx == 0:
                    print(f"MultiViewVideoDataset::__init__(): View {idx}: by start_idx ({start_idx}) and max_frames({max_frames}), selected image_list = {image_list}")
                self.image_paths += image_list
                self.mask_paths += mask_list
                if verbose:
                    print(f"MultiViewVideoDataset::__init__(): added {len(image_list)} image paths from: {video}")

                if self.n_frames is None:
                    self.n_frames = len(image_list)
                else:
                    assert self.n_frames == len(image_list), f"{video} contains"+\
                         f" {len(image_list)} frames but should contain {self.n_frames} frames."
        if verbose:
            print(f"MultiViewVideoDataset::__init__(): total parsed image paths: {len(self.image_paths)}")


        # Image Resolution Scaling
        if len(self.image_paths) > 0:
            sample_img = Image.open(self.image_paths[0])
            orig_w, orig_h = sample_img.size
            if self.resolution in [1, 2, 4, 8]:
                new_w = round(orig_w / self.resolution)
                new_h = round(orig_h / self.resolution)
            elif self.resolution == -1:
                if orig_w > 1600:
                    global_down = orig_w / 1600
                else:
                    global_down = 1
                new_w = int(orig_w / global_down)
                new_h = int(orig_h / global_down)
            else:
                global_down = orig_w / self.resolution
                new_w = int(orig_w / global_down)
                new_h = int(orig_h / global_down)

            self.image_resize = transforms.Resize((new_h, new_w))
            
            self.mask_resize  = transforms.Resize((new_h, new_w), interpolation=InterpolationMode.NEAREST)


    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        
        # orig_w, orig_h = img.size
        # # Resolution scaling logic
        # if self.resolution in [1, 2, 4, 8]:
        #     new_w = round(orig_w / self.resolution)
        #     new_h = round(orig_h / self.resolution)
        # elif self.resolution == -1:
        #     if orig_w > 1600:
        #         global WARNED
        #         if not WARNED:
        #             print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n"
        #                 "If this is not desired, please explicitly specify '--resolution/-r' as 1")
        #             WARNED = True
        #         global_down = orig_w / 1600
        #     else:
        #         global_down = 1
        #     new_w = int(orig_w / global_down)
        #     new_h = int(orig_h / global_down)
        # else:
        #     # Assume it's a custom resolution like 1024
        #     global_down = orig_w / self.resolution
        #     scale = global_down
        #     new_w = int(orig_w / scale)
        #     new_h = int(orig_h / scale)
        # #img = img.resize((new_w, new_h), resample=Image.BILINEAR)
        # img = self.image_resize((new_h, new_w))(img)

        img = self.image_resize(img)
    
        img = self.image_transform(img)
        
        H, W = img.shape[-2], img.shape[-1]
        mpath = self.mask_paths[index]
        mask_tensor = torch.ones((1, H, W), dtype=torch.uint8)
        if self.mask_paths:
            mpath = self.mask_paths[index]
            if (mpath is not None) and os.path.exists(mpath):
                m = Image.open(mpath).convert("L")              # 1채널 보장
                if hasattr(self, "mask_resize") and (self.mask_resize is not None):
                    m = self.mask_resize(m)                     # NEAREST 유지
                else:
                    # 혹시 mask_resize가 없다면 이미지 크기와 동일하게 맞추기
                    m = m.resize((W, H), resample=Image.NEAREST)

                #mask_tensor = self.image_transform(m) - 1
                m = np.array(m)
                m = (m > 0).astype(np.uint8)                    # binary
                mask_tensor = torch.from_numpy(1-m).unsqueeze(0).clone() # ★ clone으로 저장소 독립

        return img, self.image_paths[index], mask_tensor
    
    def __len__(self):
        return len(self.image_paths)
    
class SequentialMultiviewSampler(Sampler):
    
    def __init__(self, dataset: MultiViewVideoDataset) -> None:
        super().__init__(dataset)
        self.n_cams, self.n_frames = dataset.n_cams, dataset.n_frames

    def __iter__(self):
        rearrange = []
        for frame_idx in range(self.n_frames):
            rearrange += list(range(frame_idx, self.n_cams*self.n_frames, self.n_frames))
        return iter(rearrange) 
    
    def __len__(self):
        return self.n_cams*self.n_frames
    

class IndexedMultiviewSampler(Sampler):
    
    def __init__(self, dataset: MultiViewVideoDataset, frame_idx: int) -> None:
        super().__init__(dataset)
        self.n_cams, self.n_frames = dataset.n_cams, dataset.n_frames
        self.frame_idx = frame_idx

    def __iter__(self):
        return iter(list(range(self.frame_idx, self.n_cams*self.n_frames, self.n_frames)))
    
    def __len__(self):
        return self.n_cams