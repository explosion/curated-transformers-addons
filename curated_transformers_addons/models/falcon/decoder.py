from functools import partial
from typing import Any, Dict, Mapping, Optional, Tuple, Type, TypeVar

import torch
from torch import Tensor
from torch.nn import Dropout, LayerNorm, ModuleList

from curated_transformers.layers.attention import (
    AttentionHeads,
    AttentionLinearBiases,
    QkvMode,
    ScaledDotProductAttention,
    SelfAttention,
)
from curated_transformers.layers.embeddings import QueryKeyRotaryEmbeddings
from curated_transformers.layers.feedforward import PointwiseFeedForward
from curated_transformers.layers.transformer import (
    DecoderLayer,
    EmbeddingDropouts,
    EmbeddingLayerNorms,
    TransformerDropouts,
    TransformerEmbeddings,
    TransformerLayerNorms,
)
from curated_transformers.models.hf_hub import FromHFHub
from curated_transformers.models.hf_hub.conversion import (
    state_dict_from_hf,
    state_dict_to_hf,
)
from curated_transformers.models.transformer import TransformerDecoder
from ._hf import DECODER_HF_PARAM_KEY_TRANSFORMS, convert_hf_config
from .config import FalconConfig
from .layer import OldFalconDecoderLayer

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="FalconDecoder")


class FalconDecoder(TransformerDecoder[FalconConfig], FromHFHub):
    """
    Falcon (`Penedo et al., 2019`_) decoder.

    .. _Penedo et al., 2019: https://arxiv.org/abs/2306.01116
    """

    def __init__(
        self, config: FalconConfig, *, device: Optional[torch.device] = None
    ) -> None:
        """
        Construct a Falcon decoder.

        :param config:
            Decoder configuration.
        :param device:
            Device to which the module is to be moved.
        :returns:
            The decoder.
        """
        super().__init__(config)

        self.embeddings = TransformerEmbeddings(
            dropouts=EmbeddingDropouts(
                embed_output_dropout=Dropout(config.embedding.dropout_prob)
            ),
            embedding_width=config.embedding.embedding_width,
            hidden_width=config.layer.feedforward.hidden_width,
            layer_norms=EmbeddingLayerNorms(),
            n_pieces=config.embedding.n_pieces,
            n_positions=None,
            n_types=None,
            device=device,
        )

        self.layers = ModuleList(
            [
                OldFalconDecoderLayer(config.layer, device=device)
                for _ in range(config.layer.n_hidden_layers)
            ]
        )

        self.output_layer_norm = LayerNorm(
            config.layer.feedforward.hidden_width,
            config.layer.layer_norm_eps,
            device=device,
        )

    @classmethod
    def is_supported(cls: Type[Self], config: Dict[str, Any]) -> bool:
        return config.get("model_type") in ("falcon", "RefinedWeb", "RefinedWebModel")

    @classmethod
    def state_dict_from_hf(
        cls: Type[Self], params: Mapping[str, Tensor]
    ) -> Mapping[str, Tensor]:
        return state_dict_from_hf(params, DECODER_HF_PARAM_KEY_TRANSFORMS)

    @classmethod
    def state_dict_to_hf(
        cls: Type[Self], params: Mapping[str, Tensor]
    ) -> Mapping[str, Tensor]:
        return state_dict_to_hf(params, DECODER_HF_PARAM_KEY_TRANSFORMS)

    @classmethod
    def from_hf_config(
        cls: Type[Self],
        *,
        hf_config: Any,
        device: Optional[torch.device] = None,
    ) -> Self:
        config = convert_hf_config(hf_config)
        return cls(config, device=device)
