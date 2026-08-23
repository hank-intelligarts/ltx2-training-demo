import torch

from ltx_core.model.model_protocol import ModelConfigurator


class GemmaFeaturesExtractorProjLinear(torch.nn.Module, ModelConfigurator["GemmaFeaturesExtractorProjLinear"]):
    """
    Feature extractor module for Gemma models.
    This module applies a single linear projection to the input tensor.
    It expects a flattened feature tensor of shape (batch_size, 3840*49).
    The linear layer maps this to a (batch_size, 3840) embedding.
    Attributes:
        aggregate_embed (torch.nn.Linear): Linear projection layer.
    """

    def __init__(self) -> None:
        """
        Initialize the GemmaFeaturesExtractorProjLinear module.
        The input dimension is expected to be 3840 * 49, and the output is 3840.
        """
        super().__init__()
        self.aggregate_embed = torch.nn.Linear(3840 * 49, 3840, bias=False)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # v2.3 checkpoints store audio/video split keys; remap to unified key
        if "video_aggregate_embed.weight" in state_dict and "aggregate_embed.weight" not in state_dict:
            state_dict = {"aggregate_embed.weight": state_dict["video_aggregate_embed.weight"]}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the feature extractor.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3840 * 49).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 3840).
        """
        return self.aggregate_embed(x)

    @classmethod
    def from_config(cls: type["GemmaFeaturesExtractorProjLinear"], _config: dict) -> "GemmaFeaturesExtractorProjLinear":
        return cls()
