import math
import torch

# Tolerance overrides for bfloat16
atol = 1e-2
rtol = 1e-2


@torch.no_grad()
def reference(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, output, lse,
              baseline=None, **_kwargs):
    if baseline == "flashinfer":
        _reference_flashinfer(q, k_cache, v_cache, kv_indptr, kv_indices,
                              sm_scale, output, lse)
    elif baseline == "torch-compile":
        _reference_torch_compile(q, k_cache, v_cache, kv_indptr, kv_indices,
                                 sm_scale, output, lse)
    else:
        _reference_eager(q, k_cache, v_cache, kv_indptr, kv_indices,
                         sm_scale, output, lse)


def _reference_eager(q, k_cache, v_cache, kv_indptr, kv_indices,
                     sm_scale, output, lse):
    batch_size, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    gqa_ratio = num_qo_heads // num_kv_heads  # 8

    if output.dim() == 1:
        out_view = output.view(batch_size, num_qo_heads, head_dim)
    else:
        out_view = output
    if lse.dim() == 1:
        lse_view = lse.view(batch_size, num_qo_heads)
    else:
        lse_view = lse

    k_flat = k_cache.squeeze(1).to(torch.float32)
    v_flat = v_cache.squeeze(1).to(torch.float32)
    q_f32 = q.to(torch.float32)

    for b in range(batch_size):
        ps = int(kv_indptr[b].item())
        pe = int(kv_indptr[b + 1].item())
        if ps >= pe:
            out_view[b].zero_()
            lse_view[b].fill_(-float("inf"))
            continue

        idx = kv_indices[ps:pe].to(torch.long)
        k = k_flat[idx].permute(1, 0, 2).repeat_interleave(gqa_ratio, dim=0)
        v = v_flat[idx].permute(1, 0, 2).repeat_interleave(gqa_ratio, dim=0)
        q_b = q_f32[b].unsqueeze(1)

        logits = torch.bmm(q_b, k.transpose(1, 2)).squeeze(1) * sm_scale
        lse_view[b] = torch.logsumexp(logits, dim=-1) / math.log(2.0)
        attn = torch.softmax(logits, dim=-1)
        out_view[b] = torch.bmm(attn.unsqueeze(1), v).squeeze(1).to(torch.bfloat16)


@torch.no_grad()
def _reference_flashinfer(q, k_cache, v_cache, kv_indptr, kv_indices,
                          sm_scale, output, lse):
    import flashinfer

    batch_size, num_qo_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, _ = k_cache.shape

    if output.dim() == 1:
        out_view = output.view(batch_size, num_qo_heads, head_dim)
    else:
        out_view = output
    if lse.dim() == 1:
        lse_view = lse.view(batch_size, num_qo_heads)
    else:
        lse_view = lse

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", use_tensor_cores=False
    )
    last_page_len = torch.ones(batch_size, dtype=torch.int32, device=q.device)
    wrapper.plan(
        kv_indptr,
        kv_indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type="bfloat16",
        sm_scale=sm_scale,
    )

    pkv = (k_cache.contiguous().view(num_pages, page_size, num_kv_heads, head_dim),
           v_cache.contiguous().view(num_pages, page_size, num_kv_heads, head_dim))
    o = wrapper.run(q.contiguous(), pkv, return_lse=True)
    if isinstance(o, tuple):
        out_view.copy_(o[0].view_as(out_view))
        lse_view.copy_(o[1].view_as(lse_view))
    else:
        out_view.copy_(o.view_as(out_view))


_tc_fn = None

@torch.no_grad()
def _reference_torch_compile(q, k_cache, v_cache, kv_indptr, kv_indices,
                             sm_scale, output, lse):
    global _tc_fn
    if _tc_fn is None:
        @torch.compile(fullgraph=True, dynamic=True)
        def _tc_fn_impl(q, k_cache, v_cache, kv_indptr, kv_indices,
                        sm_scale, out, lse):
            _reference_eager(q, k_cache, v_cache, kv_indptr, kv_indices,
                             sm_scale, out, lse)
            return out, lse
        _tc_fn = _tc_fn_impl
    _tc_fn(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, output, lse)
