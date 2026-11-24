"""HAMS: compatibility shim for `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask`.

That module was removed in transformers 5.x in favor of a new masking-function
abstraction (`transformers.masking_utils`). The vendored VITS training code
(ylacombe/finetune-hf-vits, pinned to transformers>=4.35.1) only ever used
this one function, for a plain non-causal padding-mask expansion — so rather
than pull in the new masking API (a bigger refactor for a 2-line helper),
we keep the old, well-known implementation verbatim.
"""

from __future__ import annotations

import torch


def _prepare_4d_attention_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int | None = None) -> torch.Tensor:
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
