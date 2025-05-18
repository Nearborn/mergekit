# by BM/MM-8b

class ThreeModelMergeTask(Task[torch.Tensor]):
    gather_tensors: MergeTensorInput
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]
    weight_info: WeightInfo
    base_weight_info: WeightInfo

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(
        self, tensors: Dict[ModelReference, torch.Tensor], **_kwargs
    ) -> torch.Tensor:
        base_key, skill1_key, skill2_key = list(tensors.keys())

        tensor1 = tensors[base_key]
        tensor2 = tensors[skill1_key]
        tensor3 = tensors[skill2_key]

        weights1 = self.tensor_parameters[base_key]["weight"]
        weights2 = self.tensor_parameters[skill1_key]["weight"]
        weights3 = self.tensor_parameters[skill2_key]["weight"]

        res = (weights1 * tensor1 + weights2 * tensor2 + weights3 * tensor3) / (
            weights1 + weights2 + weights3
        )

        return res

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class ThreeModelLinear(MergeMethod):
    def name(self) -> str:
        return "three_model_linear"

    @override
    def pretty_name(self) -> Optional[str]:
        return "Three-Model Linear Merge"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="normalize", required=False, default_value=True),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="base_weight", required=True),
            ConfigParameterDef(name="skill1_weight", required=True),
            ConfigParameterDef(name="skill2_weight", required=True),
        ]

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        base_weight: WeightInfo,
        tensors: MergeTensorInput,
        parameters: Dict[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        **_kwargs,
    ) -> Task:
        return ThreeModelMergeTask(
            gather_tensors=tensors,
            tensor_parameters=tensor_parameters,
            weight_info=output_weight,
            base_weight_info=base_weight,
        )
