from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn

from sf3d.models.utils import BaseModule


class LinearCameraEmbedder(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 25
        out_channels: int = 768
        conditions: List[str] = field(default_factory=list)

    cfg: Config

    def configure(self) -> None:
        self.linear = nn.Linear(self.cfg.in_channels, self.cfg.out_channels)

    def forward(self, c2w_cond, intrinsic_normed_cond):
        cond_tensor = torch.cat([c2w_cond.view(*c2w_cond.shape[:2], -1), intrinsic_normed_cond.view(*intrinsic_normed_cond.shape[:2], -1)], dim=-1)
        embedding = self.linear(cond_tensor)
        return embedding
