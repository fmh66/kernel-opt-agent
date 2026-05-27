import torch
import triton
import triton.language as tl


def setup(**kwargs):
    batch_size = kwargs.get("batch_size", 16)
    num_qo_heads = kwargs.get("num_qo_heads", 16)
    num_kv_heads = kwargs.get("num_kv_heads", 2)
    head_dim = kwargs.get("head_dim", 128)
    num_pages = kwargs.get("num_pages", 256)
    page_size = kwargs.get("page_size", 1)
    len_indptr = kwargs.get("len_indptr", batch_size + 1)
    num_kv_indices = kwargs.get("num_kv_indices", num_pages)
    seed = kwargs.get("seed", 42)

    torch.manual_seed(seed)

    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    k_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")

    pages_per_batch = max(1, num_pages // batch_size)
    kv_indptr = torch.zeros(len_indptr, dtype=torch.int32, device="cuda")
    kv_indices_list = []
    for b in range(batch_size):
        ps = b * pages_per_batch
        pe = min(num_pages, (b + 1) * pages_per_batch)
        kv_indptr[b] = ps
        kv_indices_list.append(torch.arange(ps, pe, dtype=torch.int32))
    kv_indptr[batch_size] = num_pages
    kv_indices = torch.cat(kv_indices_list).cuda()
    num_kv_indices_actual = kv_indices.numel()

    sm_scale = 1.0 / (head_dim ** 0.5)
    output = torch.empty(batch_size, num_qo_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    lse = torch.empty(batch_size, num_qo_heads, dtype=torch.float32, device="cuda")

    return {
        "inputs": {
            "q": q,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "kv_indptr": kv_indptr,
            "kv_indices": kv_indices,
            "sm_scale": sm_scale,
            "output": output,
            "lse": lse,
            "batch_size": batch_size,
            "num_qo_heads": num_qo_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "num_pages": num_pages,
            "page_size": page_size,
            "len_indptr": len_indptr,
            "num_kv_indices": num_kv_indices_actual,
        },
        "outputs": ["output", "lse"],
    }


@triton.jit
def _gqa_paged_decode_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, kv_indptr_ptr, kv_indices_ptr,
    sm_scale, output_ptr, lse_ptr,
    batch_size, num_qo_heads, head_dim, num_kv_heads,
    num_pages, num_kv_indices, gqa_ratio,
    BLOCK_D: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)

    if pid_batch >= batch_size or pid_head >= num_qo_heads:
        return

    kv_head = pid_head // gqa_ratio

    q_offset = pid_batch * (num_qo_heads * head_dim) + pid_head * head_dim
    q_ptrs = q_ptr + q_offset + tl.arange(0, BLOCK_D)
    q = tl.load(q_ptrs, mask=tl.arange(0, BLOCK_D) < head_dim, other=0.0).to(tl.float32)

    ps = tl.load(kv_indptr_ptr + pid_batch)
    pe = tl.load(kv_indptr_ptr + pid_batch + 1)
    n_pages = pe - ps

    m_i = tl.full((1,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((1,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    stride_kv_head = head_dim
    stride_kv_page = num_kv_heads * head_dim

    for page_idx in range(n_pages):
        kv_idx = tl.load(kv_indices_ptr + ps + page_idx)
        kv_offset = kv_idx * stride_kv_page + kv_head * head_dim

        k_ptrs = k_cache_ptr + kv_offset + tl.arange(0, BLOCK_D)
        k = tl.load(k_ptrs, mask=tl.arange(0, BLOCK_D) < head_dim, other=0.0).to(tl.float32)

        v_ptrs = v_cache_ptr + kv_offset + tl.arange(0, BLOCK_D)
        v = tl.load(v_ptrs, mask=tl.arange(0, BLOCK_D) < head_dim, other=0.0).to(tl.float32)

        qk = tl.sum(q * k)
        s = qk * sm_scale

        m_new = tl.maximum(m_i, s)
        scale = tl.exp(m_i - m_new)
        l_i = l_i * scale + tl.exp(s - m_new)
        m_i = m_new

        acc = acc * scale + tl.exp(s - m_new) * v

    acc = acc / l_i

    out_offset = pid_batch * (num_qo_heads * head_dim) + pid_head * head_dim
    out_ptrs = output_ptr + out_offset + tl.arange(0, BLOCK_D)
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=tl.arange(0, BLOCK_D) < head_dim)

    lse_offset = pid_batch * num_qo_heads + pid_head
    lse_ptrs = lse_ptr + lse_offset + tl.arange(0, 1)
    tl.store(lse_ptrs, (m_i + tl.log(l_i)) * 1.4426950408889634)


def run_kernel(
    q, k_cache, v_cache, kv_indptr, kv_indices,
    sm_scale, output, lse,
    batch_size, num_qo_heads, head_dim, num_kv_heads,
    num_pages, page_size, len_indptr, num_kv_indices,
    **_kwargs,
):
    gqa_ratio = num_qo_heads // num_kv_heads
    BLOCK_D = triton.next_power_of_2(head_dim)

    grid = (batch_size, num_qo_heads)

    _gqa_paged_decode_kernel[grid](
        q, k_cache, v_cache, kv_indptr, kv_indices,
        sm_scale, output, lse,
        batch_size, num_qo_heads, head_dim, num_kv_heads,
        num_pages, num_kv_indices, gqa_ratio,
        BLOCK_D=BLOCK_D,
    )
