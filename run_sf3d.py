import os
from typing import Any, List, Literal, Optional, Tuple, Union
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import onnxruntime
import rembg
from tqdm import tqdm

from PIL import Image
from torch import Tensor

import PIL


def remove_background(
    image: Image,
    rembg_session: Any = None,
    force: bool = False,
    **rembg_kwargs,
) -> Image:
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    return image


def resize_foreground(
    image: Image,
    ratio: float,
) -> Image:
    image = np.array(image)
    assert image.shape[-1] == 4
    alpha = np.where(image[..., 3] > 0.5)
    y1, y2, x1, x2 = (
        alpha[0].min(),
        alpha[0].max(),
        alpha[1].min(),
        alpha[1].max(),
    )
    # crop the foreground
    fg = image[y1:y2, x1:x2]
    # pad to square
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = np.pad(
        fg,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )

    # compute padding according to the ratio
    new_size = int(new_image.shape[0] / ratio)
    # pad to size, double side
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = np.pad(
        new_image,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )
    new_image = Image.fromarray(new_image, mode="RGBA")
    return new_image

def handle_image(image_path, idx):
    image = remove_background(
        Image.open(image_path).convert("RGBA"), rembg_session
    )
    image = resize_foreground(image, 0.85)
    os.makedirs(os.path.join(output_dir, str(idx)), exist_ok=True)
    image.save(os.path.join(output_dir, str(idx), "input.png"))
    images.append(image)
    
class ImageProcessor:
    def convert_and_resize(
        self,
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
        size: int,
    ):
        if isinstance(image, PIL.Image.Image):
            image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
        elif isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = torch.from_numpy(image.astype(np.float32) / 255.0)
            else:
                image = torch.from_numpy(image)
        elif isinstance(image, torch.Tensor):
            pass

        batched = image.ndim == 4

        if not batched:
            image = image[None, ...]
        image = F.interpolate(
            image.permute(0, 3, 1, 2),
            (size, size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).permute(0, 2, 3, 1)
        if not batched:
            image = image[0]
        return image

    def __call__(
        self,
        image: Union[
            PIL.Image.Image,
            np.ndarray,
            torch.FloatTensor,
            List[PIL.Image.Image],
            List[np.ndarray],
            List[torch.FloatTensor],
        ],
        size: int,
    ) -> Any:
        if isinstance(image, (np.ndarray, torch.FloatTensor)) and image.ndim == 4:
            image = self.convert_and_resize(image, size)
        else:
            if not isinstance(image, list):
                image = [image]
            image = [self.convert_and_resize(im, size) for im in image]
            image = torch.stack(image, dim=0)
        return image

def get_intrinsic_from_fov(fov, H, W, bs=-1):
    focal_length = 0.5 * H / np.tan(0.5 * fov)
    intrinsic = np.identity(3, dtype=np.float32)
    intrinsic[0, 0] = focal_length
    intrinsic[1, 1] = focal_length
    intrinsic[0, 2] = W / 2.0
    intrinsic[1, 2] = H / 2.0

    if bs > 0:
        intrinsic = intrinsic[None].repeat(bs, axis=0)

    return torch.from_numpy(intrinsic)

def create_intrinsic_from_fov_deg(fov_deg: float, cond_height: int, cond_width: int):
    intrinsic = get_intrinsic_from_fov(
        np.deg2rad(fov_deg),
        H=cond_height,
        W=cond_width,
    )
    intrinsic_normed_cond = intrinsic.clone()
    intrinsic_normed_cond[..., 0, 2] /= cond_width
    intrinsic_normed_cond[..., 1, 2] /= cond_height
    intrinsic_normed_cond[..., 0, 0] /= cond_width
    intrinsic_normed_cond[..., 1, 1] /= cond_height

    return intrinsic, intrinsic_normed_cond


def default_cond_c2w(distance: float):
    c2w_cond = torch.as_tensor(
        [
            [0, 0, 1, distance],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]
    ).float()
    return c2w_cond

if __name__ == "__main__":

    output_dir = '/disk_hdd/wd_ksw/Table_Recognition/sf3d/output'
    os.makedirs(output_dir, exist_ok=True)
    image_processor = ImageProcessor()
    sess_options = onnxruntime.SessionOptions()
    #sess_options.enable_profiling = True

    # ONNX 모델 로드
    providers = ["CUDAExecutionProvider"]  # GPU 사용을 위한 설정
    # onnx_session = onnxruntime.InferenceSession("onnx/sf3d.onnx", sess_options)
    # onnx_session.set_providers(['CUDAExecutionProvider'])


    # # ONNX 모델 warmup
    # input_name = onnx_session.get_inputs()[0].name
    # output_names = [out.name for out in onnx_session.get_outputs()]
    #


    rembg_session = rembg.new_session()
    images = []
    idx = 0
    batch_size = 1
    for image_path in ['demo_files/examples/chair1.png']:



        if os.path.isdir(image_path):
            image_paths = [
                os.path.join(image_path, f)
                for f in os.listdir(image_path)
                if f.endswith((".png", ".jpg", ".jpeg"))
            ]
            for image_path in image_paths:
                handle_image(image_path, idx)
                idx += 1
        else:
            handle_image(image_path, idx)
            idx += 1

    for i in tqdm(range(0, len(images), batch_size)):
        image = images[i : i + batch_size]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            rgb_cond = []
            mask_cond = []
            for img in image:
                img_cond = (
                    torch.from_numpy(
                        np.asarray(
                            img.resize((512, 512))
                        ).astype(np.float32)
                        / 255.0
                    )
                    .float()
                    .clip(0, 1)
                    .to('cuda')
                )
                mask = img_cond[:, :, -1:]
                rgb = torch.lerp(
                    torch.tensor([0.5, 0.5, 0.5], device='cuda')[None, None, :],
                    img_cond[:, :, :3],
                    mask,
                )
                mask_cond.append(mask)
                rgb_cond.append(rgb)
            rgb_cond = torch.stack(rgb_cond, 0)
            mask_cond = torch.stack(mask_cond, 0)
            batch_size = rgb_cond.shape[0]
            
            c2w_cond = default_cond_c2w(1.6).to('cuda')
            intrinsic, intrinsic_normed_cond = create_intrinsic_from_fov_deg(
                40.0,
                512,
                512,
            )

            batch = {
                "rgb_cond": rgb_cond,
                "mask_cond": mask_cond,
                "c2w_cond": c2w_cond.view(1, 1, 4, 4).repeat(batch_size, 1, 1, 1),
                "intrinsic_cond": intrinsic.to('cuda')
                .view(1, 1, 3, 3)
                .repeat(batch_size, 1, 1, 1),
                "intrinsic_normed_cond": intrinsic_normed_cond.to('cuda')
                .view(1, 1, 3, 3)
                .repeat(batch_size, 1, 1, 1),
            }
            batch["rgb_cond"] = image_processor(
                batch["rgb_cond"], 512
            )
            #([1, 512, 512, 3])
            batch["mask_cond"] = image_processor(
                batch["mask_cond"], 512
            )
            if len(batch["rgb_cond"].shape) == 4:
                batch["rgb_cond"] = batch["rgb_cond"].unsqueeze(1)
                batch["mask_cond"] = batch["mask_cond"].unsqueeze(1)
                batch["c2w_cond"] = batch["c2w_cond"].unsqueeze(1)
                batch["intrinsic_cond"] = batch["intrinsic_cond"].unsqueeze(1)
                batch["intrinsic_normed_cond"] = batch["intrinsic_normed_cond"].unsqueeze(1)
        # onnx_outputs = onnx_session.run(output_names, {input_name: input_list[0][0]})
