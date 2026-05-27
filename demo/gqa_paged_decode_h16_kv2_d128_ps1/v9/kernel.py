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
    num_pages, num_kv_indices,
    PAIRS_PER_KV: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_pair = tl.program_id(1)

    total_pairs = num_qo_heads // 2
    if pid_batch >= batch_size or pid_pair >= total_pairs:
        return

    kv_head = (pid_pair * 2) // GQA_RATIO

    ps = tl.load(kv_indptr_ptr + pid_batch)
    pe = tl.load(kv_indptr_ptr + pid_batch + 1)
    n_pages = pe - ps

    stride_kv_page = num_kv_heads * head_dim
    q_stride = num_qo_heads * head_dim
    log2e = 1.4426950408889634

    # Load 2 Q vectors
    q_head_0 = pid_pair * 2
    q_offset_0 = pid_batch * q_stride + q_head_0 * head_dim
    q_ptrs_0 = q_ptr + q_offset_0 + tl.arange(0, BLOCK_D)
    q_0 = tl.load(q_ptrs_0, mask=tl.arange(0, BLOCK_D) < head_dim, other=0.0).to(tl.float32)

    q_head_1 = pid_pair * 2 + 1
    q_offset_1 = pid_batch * q_stride + q_head_1 * head_dim
    q_ptrs_1 = q_ptr + q_offset_1 + tl.arange(0, BLOCK_D)
    q_1 = tl.load(q_ptrs_1, mask=tl.arange(0, BLOCK_D) < head_dim, other=0.0).to(tl.float32)

    # 2 sets of online softmax state
    m_0 = tl.full((1,), -float("inf"), dtype=tl.float32)
    l_0 = tl.zeros((1,), dtype=tl.float32)
    acc_0 = tl.zeros((BLOCK_D,), dtype=tl.float32)

    m_1 = tl.full((1,), -float("inf"), dtype=tl.float32)
    l_1 = tl.zeros((1,), dtype=tl.float32)
    acc_1 = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Precompute base offset: kv_indices are sequential, so we can compute incrementally
    kv_head_offset = kv_head * head_dim
    first_kv_idx = tl.load(kv_indices_ptr + ps)
    base_page_offset = first_kv_idx * stride_kv_page + kv_head_offset

    # Page loop — K/V loaded once per page, reused for both Q-heads
    for page_idx in range(n_pages):
        kv_offset = base_page_offset + page_idx * stride_kv_page

        k = tl.load(k_cache_ptr + kv_offset + tl.arange(0, BLOCK_D),
                    mask=tl.arange(0, BLOCK_D) < head_dim, other=0.0).to(tl.float32)
        v = tl.load(v_cache_ptr + kv_offset + tl.arange(0, BLOCK_D),
                    mask=tl.arange(0, BLOCK_D) < head_dim, other=0.0).to(tl.float32)

        # Q-head 0
        qk_0 = tl.sum(q_0 * k)
        s_0 = qk_0 * sm_scale
        m_new_0 = tl.maximum(m_0, s_0)
        scale_0 = tl.exp(m_0 - m_new_0)
        l_0 = l_0 * scale_0 + tl.exp(s_0 - m_new_0)
        m_0 = m_new_0
        acc_0 = acc_0 * scale_0 + tl.exp(s_0 - m_new_0) * v

        # Q-head 1
        qk_1 = tl.sum(q_1 * k)
        s_1 = qk_1 * sm_scale
        m_new_1 = tl.maximum(m_1, s_1)
        scale_1 = tl.exp(m_1 - m_new_1)
        l_1 = l_1 * scale_1 + tl.exp(s_1 - m_new_1)
        m_1 = m_new_1
        acc_1 = acc_1 * scale_1 + tl.exp(s_1 - m_new_1) * v

    # Write outputs for both Q-heads
    acc_0 = acc_0 / l_0
    out_ptrs_0 = output_ptr + pid_batch * q_stride + q_head_0 * head_dim + tl.arange(0, BLOCK_D)
    tl.store(out_ptrs_0, acc_0.to(tl.bfloat16), mask=tl.arange(0, BLOCK_D) < head_dim)
    lse_ptrs_0 = lse_ptr + pid_batch * num_qo_heads + q_head_0 + tl.arange(0, 1)
    tl.store(lse_ptrs_0, (m_0 + tl.log(l_0)) * log2e)

    acc_1 = acc_1 / l_1
    out_ptrs_1 = output_ptr + pid_batch * q_stride + q_head_1 * head_dim + tl.arange(0, BLOCK_D)
    tl.store(out_ptrs_1, acc_1.to(tl.bfloat16), mask=tl.arange(0, BLOCK_D) < head_dim)
    lse_ptrs_1 = lse_ptr + pid_batch * num_qo_heads + q_head_1 + tl.arange(0, 1)
    tl.store(lse_ptrs_1, (m_1 + tl.log(l_1)) * log2e)


def run_kernel(
    q, k_cache, v_cache, kv_indptr, kv_indices,
    sm_scale, output, lse,
    batch_size, num_qo_heads, head_dim, num_kv_heads,
    num_pages, page_size, len_indptr, num_kv_indices,
    **_kwargs,
):
    gqa_ratio = num_qo_heads // num_kv_heads
    BLOCK_D = triton.next_power_of_2(head_dim)
    total_pairs = num_qo_heads // 2

    grid = (batch_size, total_pairs)

    _gqa_paged_decode_kernel[grid](
        q, k_cache, v_cache, kv_indptr, kv_indices,
        sm_scale, output, lse,
        batch_size, num_qo_heads, head_dim, num_kv_heads,
        num_pages, num_kv_indices,
        PAIRS_PER_KV=gqa_ratio // 2,
        GQA_RATIO=gqa_ratio,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
