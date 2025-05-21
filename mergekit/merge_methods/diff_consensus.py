# Copyright (C) 2025 Andrew Toomey
# Modified from (C) 2025 ArceeAI 
# SPDX-License-Identifier: BUSL-1.1

from typing import Any, Dict, Optional

import torch
from typing_extensions import override

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes


class SkillTargetingMergeTask(Task[torch.Tensor]):
    """
    Task that merges three models: base, skillA, skillB.
    Only the weight‐deltas that point in the same direction
    in both skill models are summed and applied to the base.
    """
    gather_tensors: MergeTensorInput
    weight_info: WeightInfo

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(
        self, tensors: Dict[ModelReference, torch.Tensor], **_kwargs
    ) -> torch.Tensor:
        # Expect exactly three models
        if len(tensors) != 3:
            raise RuntimeError(f"SkillTargetingMergeTask requires 3 models, got {len(tensors)}")

        # Rely on insertion order: [base, skillA, skillB]
        keys = list(tensors.keys())
        W_base, W_skillA, W_skillB = (tensors[k] for k in keys)

        # Ensure embeddings line up (if this is an embedding table)
        rectify_embed_sizes(self.weight_info, [W_base, W_skillA, W_skillB])

        # Shape check
        shapes = {W_base.shape, W_skillA.shape, W_skillB.shape}
        if len(shapes) != 1:
            raise RuntimeError(
                f"Tensor size mismatch for {self.weight_info.name}: {shapes}"
            )

        # Compute diffs
        diffA = W_skillA - W_base
        diffB = W_skillB - W_base

        # Build a mask of positions where both diffs share the same non‐zero sign
        # torch.sign returns -1, 0, or +1
        signA = torch.sign(diffA)
        signB = torch.sign(diffB)
        common_mask = (signA * signB) > 0  # True where both +1 or both -1

        merge_magnitude = 0.5  # 0.0 = no change from original model; 0.5 = halfway blend between original and average of common different values; 1.0 = full replacement with sourced models
        
        # Average only the diffs with same signing
        avg_skills = 0.5 * (W_skillA + W_skillB)
        W_new = torch.where(common_mask,
                            torch.lerp(W_base, avg_skills, merge_magnitude),  
                            W_base)

        # Apply to base
        return W_new

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class SkillTargetingMerge(MergeMethod):
    def name(self) -> str:
        return "skill_targeting"

    @override
    def pretty_name(self) -> Optional[str]:
        return "Skill-Targeting Merge"

    @override
    def reference_url(self) -> Optional[str]:
        return None

    def parameters(self):
        # Could add thresholds or weighting later
        return []

    def tensor_parameters(self):
        return []

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        **_kwargs,
    ) -> Task:
        return SkillTargetingMergeTask(
            gather_tensors=tensors,
            weight_info=output_weight,
        )
