import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple, Union
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
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

class SF3D(nn.Module):
    
    def __init__(self):
        super().__init__()
        cond_image_size = 512
        isosurface_resolution = 160
        isosurface_threshold: float = 10.0
        radius: float = 0.87
        background_color: list[float] = [0.5, 0.5, 0.5]
        default_fovy_deg: float = 40.0
        default_distance: float = 1.6

        camera_embedder_cls: str = 'sf3d.models.new_camera.LinearCameraEmbedder'
        camera_embedder: dict = {'in_channels': 25, 'out_channels': 768, 'conditions': ['c2w_cond', 'intrinsic_normed_cond']}

        image_tokenizer_cls = 'sf3d.models.tokenizers.new_image.DINOV2SingleImageTokenizer'
        image_tokenizer: dict = {'pretrained_model_name_or_path': 'facebook/dinov2-large', 
                           'width': 512, 'height': 512, 
                           'modulation_cond_dim': 768}

        tokenizer_cls: str = 'sf3d.models.tokenizers.triplane.TriplaneLearnablePositionalEmbedding'
        tokenizer: dict = {'plane_size': 96, 'num_channels': 1024}

        backbone_cls: str = 'sf3d.models.transformers.new_backbone.TwoStreamInterleaveTransformer'
        backbone: dict = {'num_attention_heads': 16, 
                    'attention_head_dim': 64, 
                    'raw_triplane_channels': 1024, 
                    'triplane_channels': 1024, 
                    'raw_image_channels': 1024, 
                    'num_latents': 1792, 
                    'num_blocks': 4, 
                    'num_basic_blocks': 3}

        post_processor_cls: str = 'sf3d.models.network.PixelShuffleUpsampleNetwork'
        post_processor: dict = {'in_channels': 1024, 'out_channels': 40, 'scale_factor': 4, 'conv_layers': 4}

        decoder_cls: str = 'sf3d.models.network.MaterialMLP'
        decoder: dict = {'in_channels': 120, 
                   'n_neurons': 64, 
                   'activation': 'silu', 
                   'heads': [{'name': 'density', 'out_channels': 1, 'out_bias': -1.0, 'n_hidden_layers': 2, 'output_activation': 'trunc_exp'}, 
                             {'name': 'features', 'out_channels': 3, 'n_hidden_layers': 3, 'output_activation': 'sigmoid'}, 
                             {'name': 'perturb_normal', 'out_channels': 3, 'n_hidden_layers': 3, 'output_activation': 'normalize_channel_last'}, 
                             {'name': 'vertex_offset', 'out_channels': 3, 'n_hidden_layers': 2}]}

        image_estimator_cls: str = 'sf3d.models.image_estimator.clip_based_estimator.ClipBasedHeadEstimator'
        image_estimator: dict = {'distribution': 'beta', 
                           'distribution_eval': 'mode', 
                           'heads': [{'name': 'roughness', 
                                      'out_channels': 1, 
                                      'n_hidden_layers': 3, 
                                      'output_activation': 'linear', 
                                      'add_to_decoder_features': True, 
                                      'output_bias': 1.0, 
                                      'shape': [-1, 1, 1]}, 
                                     {'name': 'metallic', 
                                      'out_channels': 1, 
                                      'n_hidden_layers': 3, 
                                      'output_activation': 'linear', 
                                      'add_to_decoder_features': True, 'output_bias': 1.0, 'shape': [-1, 1, 1]}]}

        global_estimator_cls: str = 'sf3d.models.global_estimator.multi_head_estimator.MultiHeadEstimator'
        global_estimator: dict = {'triplane_features': 1024, 
                            'heads': [{'name': 'sg_amplitudes', 'out_channels': 24, 
                                       'n_hidden_layers': 3, 
                                       'output_activation': 'softplus', 
                                       'output_bias': 1.0, 
                                       'shape': [-1, 24, 1]}]}
        
        self.image_tokenizer = find_class(image_tokenizer_cls)(
        image_tokenizer
        )
        self.tokenizer = find_class(tokenizer_cls)(tokenizer)
        self.camera_embedder = find_class(camera_embedder_cls)(
            camera_embedder
        )
        self.backbone = find_class(backbone_cls)(backbone)
        self.post_processor = find_class(post_processor_cls)(
            post_processor
        )
        self.decoder = find_class(decoder_cls)(decoder)
        self.image_estimator = find_class(image_estimator_cls)(
            image_estimator
        )
        self.global_estimator = find_class(global_estimator_cls)(
            global_estimator
        )

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
    model = SF3D()
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
    
    