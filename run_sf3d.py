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
from einops import rearrange
from huggingface_hub import hf_hub_download
from jaxtyping import Float
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_model
from torch import Tensor

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

    sess_options = onnxruntime.SessionOptions()
    sess_options.enable_profiling = True

    # ONNX 모델 로드
    providers = ["CUDAExecutionProvider"]  # GPU 사용을 위한 설정
    onnx_session = onnxruntime.InferenceSession("onnx/sf3d.onnx", sess_options)
    onnx_session.set_providers(['CUDAExecutionProvider'])


    # ONNX 모델 warmup
    input_name = onnx_session.get_inputs()[0].name
    output_names = [out.name for out in onnx_session.get_outputs()]
    #onnx_outputs = onnx_session.run(output_names, {input_name: input_list[0][0]})



    model = SF3D.from_pretrained(
        'stabilityai/stable-fast-3d',
        config_name="config.yaml",
        weight_name="model.safetensors",
    )


    images = []
    idx = 0
    for image_path in ['demo_files/examples/chair1.png']:

        def handle_image(image_path, idx):
            image = remove_background(
                Image.open(image_path).convert("RGBA"), rembg_session
            )
            image = resize_foreground(image, args.foreground_ratio)
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

    for i in tqdm(range(0, len(images), args.batch_size)):
        image = images[i : i + args.batch_size]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            with torch.autocast(
                device_type=device, dtype=torch.float16
            ) if "cuda" in device else nullcontext():
                mesh, glob_dict = model.run_image(
                    image,
                    bake_resolution=args.texture_resolution,
                    remesh=args.remesh_option,
                    vertex_count=args.target_vertex_count,
                )