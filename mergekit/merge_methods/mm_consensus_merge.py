# by BM/MM-8b
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
    def __init__(
        self,
        gather_tensors: MergeTensorInput,
        weight_info: WeightInfo,
        step: float,
        min_diff: float,
        quantile: Optional[float],
    ):
        self.gather_tensors = gather_tensors
        self.weight_info = weight_info
        self.step = step            # how far toward avg_skills (0…1)
        self.min_diff = min_diff    # absolute threshold
        self.quantile = quantile    # if not None, use tensor‐specific threshold

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor], **_):
        # Unpack in insertion order: base, skillA, skillB
        Wb, Wa, Wc = tensors.values()

        # Align embeddings if needed
        rectify_embed_sizes(self.weight_info, [Wb, Wa, Wc])

        # Shape check
        unique_shapes = {t.shape for t in [Wb, Wa, Wc]}
        if len(unique_shapes) != 1:
            raise RuntimeError(
                f"Tensor size mismatch for {self.weight_info.name}: {unique_shapes}"
            )

        # Compute deltas
        diffA = Wa - Wb
        diffB = Wc - Wb

        # Same sign mask
        signA = torch.sign(diffA)
        signB = torch.sign(diffB)
        same_sign = (signA * signB) > 0

        # Magnitudes
        absA, absB = diffA.abs(), diffB.abs()

        # Threshold mask (quantile or fixed)
        if self.quantile is not None:
            tA = torch.quantile(absA, self.quantile)
            tB = torch.quantile(absB, self.quantile)
            thresh_mask = (absA >= tA) & (absB >= tB)
        else:
            thresh_mask = (absA >= self.min_diff) & (absB >= self.min_diff)

        common_mask = same_sign & thresh_mask

        # Average of the two skill models
        avg_skills = 0.5 * (Wa + Wc)

        # Blend partway toward that average
        W_new = torch.where(
            common_mask,
            torch.lerp(Wb, avg_skills, self.step),
            Wb
        )
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

    def parameters(self) -> list[ConfigParameterDef]:
        return [
            ConfigParameterDef("step",     required=False, default_value=0.5),
            ConfigParameterDef("min_diff", required=False, default_value=1e-3),
            ConfigParameterDef("quantile", required=False, default_value=None),
        ]

    def tensor_parameters(self) -> list[ConfigParameterDef]:
        return []

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        parameters: Dict[str, Any],
        **_
    ) -> Task[torch.Tensor]:
        return SkillTargetingMergeTask(
            gather_tensors=tensors,
            weight_info=output_weight,
            step=parameters["step"],
            min_diff=parameters["min_diff"],
            quantile=parameters["quantile"],
        )
