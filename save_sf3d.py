import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple, Union
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import hf_hub_download
from jaxtyping import Float
from omegaconf import OmegaConf
from safetensors.torch import load_model
from torch import Tensor
from sf3d.models.isosurface import MarchingTetrahedraHelper
try:
    from texture_baker import TextureBaker
except ImportError:
    import logging
    
from sf3d.models.utils import (
    BaseModule,
    ImageProcessor,
    find_class,
)

class SF3D(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        cond_image_size: int
        isosurface_resolution: int
        isosurface_threshold: float = 10.0
        radius: float = 1.0
        background_color: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
        default_fovy_deg: float = 40.0
        default_distance: float = 1.6

        camera_embedder_cls: str = ""
        camera_embedder: dict = field(default_factory=dict)

        image_tokenizer_cls: str = ""
        image_tokenizer: dict = field(default_factory=dict)

        tokenizer_cls: str = ""
        tokenizer: dict = field(default_factory=dict)

        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)

        post_processor_cls: str = ""
        post_processor: dict = field(default_factory=dict)

        decoder_cls: str = ""
        decoder: dict = field(default_factory=dict)

        image_estimator_cls: str = ""
        image_estimator: dict = field(default_factory=dict)

        global_estimator_cls: str = ""
        global_estimator: dict = field(default_factory=dict)

    cfg: Config

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, config_name: str, weight_name: str
    ):
        if os.path.isdir(pretrained_model_name_or_path):
            config_path = os.path.join(pretrained_model_name_or_path, config_name)
            weight_path = os.path.join(pretrained_model_name_or_path, weight_name)
        else:
            config_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=config_name
            )
            weight_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=weight_name
            )

        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        model = cls(cfg)
        load_model(model, weight_path)
        return model
    
    def configure(self):
        self.image_tokenizer = find_class(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )
        self.tokenizer = find_class(self.cfg.tokenizer_cls)(self.cfg.tokenizer)
        self.camera_embedder = find_class(self.cfg.camera_embedder_cls)(
            self.cfg.camera_embedder
        )
        self.backbone = find_class(self.cfg.backbone_cls)(self.cfg.backbone)
        self.post_processor = find_class(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )
        self.decoder = find_class(self.cfg.decoder_cls)(self.cfg.decoder)
        self.image_estimator = find_class(self.cfg.image_estimator_cls)(
            self.cfg.image_estimator
        )
        self.global_estimator = find_class(self.cfg.global_estimator_cls)(
            self.cfg.global_estimator
        )

        self.bbox: Float[Tensor, "2 3"]
        self.register_buffer(
            "bbox",
            torch.as_tensor(
                [
                    [-self.cfg.radius, -self.cfg.radius, -self.cfg.radius],
                    [self.cfg.radius, self.cfg.radius, self.cfg.radius],
                ],
                dtype=torch.float32,
            ),
        )
        self.isosurface_helper = MarchingTetrahedraHelper(
            self.cfg.isosurface_resolution,
            os.path.join(
                os.path.dirname(__file__),
                "load",
                "tets",
                f"{self.cfg.isosurface_resolution}_tets.npz",
            ),
        )

        self.baker = TextureBaker()
        self.image_processor = ImageProcessor()
            
    def forward(self, rgb_cond, mask_cond, c2w_cond, intrinsic_normed_cond):
        
        camera_embeds: Optional[Float[Tensor, "B Nv Cc"]]
        camera_embeds = self.camera_embedder(c2w_cond, intrinsic_normed_cond)
        
        input_image_tokens: Float[Tensor, "B Nv Cit Nit"] = self.image_tokenizer(
            rearrange(rgb_cond, 'B Nv H W C -> B Nv C H W'), 
            modulation_cond=camera_embeds
        )
        
        input_image_tokens = rearrange(
            input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=rgb_cond.shape[1]
        )
        
        tokens: Float[Tensor, "B Ct Nt"] = self.tokenizer(rgb_cond.shape[0])
        
        tokens = self.backbone(
            tokens,
            encoder_hidden_states=input_image_tokens,
            modulation_cond=None,
        )
        
        non_postprocessed_codes = self.tokenizer.detokenize(tokens)
        scene_codes = self.post_processor(non_postprocessed_codes)
        
        global_dict = {}
        if self.image_estimator is not None:
            global_dict.update(
                self.image_estimator(rgb_cond * mask_cond)
            )
        
        return scene_codes


if __name__ == "__main__":
    model = SF3D.from_pretrained(
        'stabilityai/stable-fast-3d',
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    model.eval()
    # 입력 크기 정의
    batch_size = 1
    Nv = 1
    H = 512
    W = 512
    C = 3
    modulation_dim = 768

    # 더미 입력 생성
    rgb_cond = torch.randn(batch_size, Nv, H, W, C)
    mask_cond = torch.randn(batch_size, Nv, H, W, C)
    c2w_cond = torch.randn(batch_size, Nv, 4, 4)
    intrinsic_normed_cond = torch.randn(batch_size, Nv, 3, 3)
    #model(rgb_cond, mask_cond, c2w_cond, intrinsic_normed_cond)
    
    torch.onnx.export(
        model,  # 내보낼 모델
        (rgb_cond, mask_cond, c2w_cond, intrinsic_normed_cond),  # 모델 입력
        'onnx/sf3d.onnx',  # 저장할 ONNX 파일 이름
        input_names=['rgb_cond', 'mask_cond', 'c2w_cond', 'intrinsic_normed_cond'],  # 입력 노드 이름
        output_names=['scene_codes'],  # 출력 노드 이름
        dynamic_axes={
            'rgb_cond': {0: 'B', 1: 'Nv'},
            'mask_cond': {0: 'B', 1: 'Nv'},
            'c2w_cond': {0: 'B', 1: 'Nv'},
            'intrinsic_normed_cond': {0: 'B', 1: 'Nv'},
            'scene_codes': {0: 'B'},
        },
        opset_version=14,  # 사용하려는 ONNX opset 버전
    )