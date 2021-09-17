r"""GRU language model with residual connection."""

from typing import ClassVar, Dict, Optional

import torch.nn as nn

from lmp.model._res_rnn import ResRNNBlock, ResRNNModel
from lmp.tknzr._base import BaseTknzr


class ResGRUBlock(ResRNNBlock):
    r"""Residual connected GRU blocks.

    Same architecture as :py:class:`lmp.model.ResRNNBlock` but replace RNN
    with GRU instead.

    Parameters
    ==========
    d_hid: int
        Hidden dimension for residual connected GRU.
        Must be bigger than or equal to ``1``.
    kwargs: Dict, optional
        Useless parameter.
        Intently left for subclass parameters extension.
    n_hid_lyr: int
        Number of residual connected GRU layers.
        Must be bigger than or equal to ``1``.
    p_hid: float
        Dropout probability for every hidden representations.
        Must satisfy ``0.0 <= p_hid <= 1.0``.

    Attributes
    ==========
    dp: torch.nn.ModuleList[torch.nn.Dropout]
        Drop each output temporal features of ``self.recur`` with
        probability ``p_hid``.
        Do not dropout last temporal features output from ``self.recur[-1]``
        since :py:class:`lmp.model.ResGRUModel` have ``self.post_hid`` which
        drop output of ``self.hid``.
    recur: torch.nn.ModuleList[torch.nn.GRU]
        List of GRU which encode temporal features.
        Each time step's hidden state depends on current input and previous
        hidden state.
    """

    def __init__(
            self,
            *,
            d_hid: int,
            n_hid_lyr: int,
            p_hid: float,
            **kwargs: Optional[Dict],
    ):
        super().__init__(
            d_hid=d_hid,
            n_hid_lyr=n_hid_lyr,
            p_hid=p_hid,
            **kwargs,
        )

        # Override RNN layer with GRU.
        # Input tensor : Output of `ResGRUModel.pre_hid`.
        # Input shape  : `(B, S, H)`.
        # Input dtype  : `torch.float32`.
        # Output tensor: Batch of recurrent token hidden states.
        # Output shape : `(B, S, H)`.
        # Output dtype : `torch.float32`.
        self.recur = nn.ModuleList([
            nn.GRU(input_size=d_hid, hidden_size=d_hid, batch_first=True)
            for _ in range(n_hid_lyr)
        ])


class ResGRUModel(ResRNNModel):
    r"""GRU language model with residual connection.

    Same architecture as :py:class:`lmp.model.ResRNNModel` but use residual
    connection on GRU layer.

    Parameters
    ==========
    d_emb: int
        Token embedding dimension.
        Must be bigger than or equal to ``1``.
    d_hid: int
        Hidden dimension for MLP and residual connected GRU.
        Must be bigger than or equal to ``1``.
    kwargs: Dict, optional
        Useless parameter.
        Intently left for subclass parameters extension.
    n_hid_lyr: int
        Number of residual connected GRU layers.
        Must be bigger than or equal to ``1``.
    n_post_hid_lyr: int
        Number of MLP layers ``+1`` after residual connected GRU layer.
        ``+1`` since we need at least one MLP layer to transform dimension.
        (If you want 2 layers, then you need to set ``n_post_hid_lyr = 1``.)
        Must be bigger than or equal to ``1``.
    n_pre_hid_lyr: int
        Number of MLP layers ``+1`` before residual connected GRU layer.
        ``+1`` since we need at least one MLP layer to transform dimension.
        (If you want 2 layers, then you need to set ``n_pre_hid_lyr = 1``.)
        Must be bigger than or equal to ``1``.
    p_emb: float
        Dropout probability for token embeddings.
        Must satisfy ``0.0 <= p_emb <= 1.0``.
    p_hid: float
        Dropout probability for every hidden representations.
        Must satisfy ``0.0 <= p_hid <= 1.0``.
    tknzr: lmp.tknzr.BaseTknzr
        Tokenizer instance with attributes ``pad_tkid`` and ``vocab_size``.

    Attributes
    ==========
    hid: ResGRUBlock
        Residual connected GRU which encode temporal features.
        Each time step's hidden state depends on current input and previous
        hidden state.
        Drop temporal features with probability ``p_hid``.
    model_name: ClassVar[str]
        Model name is ``res-GRU``.
        Used for command line argument parsing.
    """
    model_name: ClassVar[str] = 'res-GRU'

    def __init__(
            self,
            *,
            d_emb: int,
            d_hid: int,
            n_hid_lyr: int,
            n_post_hid_lyr: int,
            n_pre_hid_lyr: int,
            p_emb: float,
            p_hid: float,
            tknzr: BaseTknzr,
            **kwargs: Optional[Dict],
    ):
        super().__init__(
            d_emb=d_emb,
            d_hid=d_hid,
            n_hid_lyr=n_hid_lyr,
            n_post_hid_lyr=n_post_hid_lyr,
            n_pre_hid_lyr=n_pre_hid_lyr,
            p_emb=p_emb,
            p_hid=p_hid,
            tknzr=tknzr,
            **kwargs,
        )

        # Override residual connected RNN layer with residual connected GRU.
        # Input tensor : Output of `self.pre_hid`.
        # Input shape  : `(B, S, H)`.
        # Input dtype  : `torch.float32`.
        # Output tensor: Batch of recurrent token hidden states.
        # Output shape : `(B, S, H)`.
        # Output dtype : `torch.float32`.
        self.hid = ResGRUBlock(
            d_hid=d_hid,
            n_hid_lyr=n_hid_lyr,
            p_hid=p_hid,
            **kwargs,
        )
