import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple, Union
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import onnxruntime
import rembg
from tqdm import tqdm

from einops import rearrange
from huggingface_hub import hf_hub_download
from jaxtyping import Float
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_model
from torch import Tensor
from sf3d.utils import get_device, remove_background, resize_foreground

from sf3d.models.utils import (
    BaseModule,
    ImageProcessor,
    convert_data,
    dilate_fill,
    dot,
    find_class,
    float32_to_uint8_np,
    normalize,
    scale_tensor,
)
from sf3d.utils import create_intrinsic_from_fov_deg, default_cond_c2w, get_device

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

        def handle_image(image_path, idx):
            image = remove_background(
                Image.open(image_path).convert("RGBA"), rembg_session
            )
            image = resize_foreground(image, 0.85)
            os.makedirs(os.path.join(output_dir, str(idx)), exist_ok=True)
            image.save(os.path.join(output_dir, str(idx), "input.png"))
            images.append(image)

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
        # mesh, glob_dict = model.run_image(
        #     image,
        #     bake_resolution=args.texture_resolution,
        #     remesh=args.remesh_option,
        #     vertex_count=args.target_vertex_count,
        # )
