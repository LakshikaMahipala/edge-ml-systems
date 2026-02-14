from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import random

@dataclass(frozen=True)
class StageSpace:
    depth_choices: List[int]
    out_ch_choices: List[int]
    block_choices: List[str]     # e.g., ["conv", "mbconv"]
    k_choices: List[int]         # kernel sizes
    exp_choices: List[int]       # expansion ratios (mbconv)
    se_choices: List[int]        # 0/1

@dataclass(frozen=True)
class SearchSpace:
    stem_ch_choices: List[int]
    head_ch_choices: List[int]
    stage_spaces: List[StageSpace]
    num_classes: int = 10

def default_search_space(num_classes: int = 10) -> SearchSpace:
    # Tiny mobile-ish space (small on purpose)
    stages = [
        StageSpace(depth_choices=[1,2], out_ch_choices=[16,24], block_choices=["conv","mbconv"], k_choices=[3,5], exp_choices=[2,4], se_choices=[0,1]),
        StageSpace(depth_choices=[2,3], out_ch_choices=[24,32,40], block_choices=["mbconv"], k_choices=[3,5], exp_choices=[2,4,6], se_choices=[0,1]),
        StageSpace(depth_choices=[2,3], out_ch_choices=[40,64], block_choices=["mbconv"], k_choices=[3,5], exp_choices=[4,6], se_choices=[0,1]),
    ]
    return SearchSpace(
        stem_ch_choices=[8,16],
        head_ch_choices=[64,128],
        stage_spaces=stages,
        num_classes=num_classes,
    )

def sample_arch(space: SearchSpace, rng: random.Random | None = None) -> Dict[str, Any]:
    rng = rng or random.Random()
    arch = {
        "stem_channels": rng.choice(space.stem_ch_choices),
        "head_channels": rng.choice(space.head_ch_choices),
        "num_classes": space.num_classes,
        "stages": [],
    }
    for ss in space.stage_spaces:
        arch["stages"].append({
            "depth": rng.choice(ss.depth_choices),
            "out_ch": rng.choice(ss.out_ch_choices),
            "block": rng.choice(ss.block_choices),
            "k": rng.choice(ss.k_choices),
            "exp": rng.choice(ss.exp_choices),
            "se": rng.choice(ss.se_choices),
        })
    return arch
