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

    sm_scale = 1.0 / (head_dim ** 0.5)
    output = torch.empty(batch_size, num_qo_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    lse = torch.empty(batch_size, num_qo_heads, dtype=torch.float32, device="cuda")

    return {
        "inputs": {
            "q": q, "k_cache": k_cache, "v_cache": v_cache,
            "kv_indptr": kv_indptr, "kv_indices": kv_indices,
            "sm_scale": sm_scale, "output": output, "lse": lse,
            "batch_size": batch_size, "num_qo_heads": num_qo_heads,
            "num_kv_heads": num_kv_heads, "head_dim": head_dim,
            "num_pages": num_pages, "page_size": page_size,
            "len_indptr": len_indptr, "num_kv_indices": kv_indices.numel(),
        },
        "outputs": ["output", "lse"],
    }


@triton.jit
def _gqa_paged_decode_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, kv_indptr_ptr, kv_indices_ptr,
    sm_scale, output_ptr, lse_ptr,
    batch_size, num_qo_heads, head_dim, num_kv_heads,
    num_pages, num_kv_indices,
    HEADS_PER_BLOCK: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_block = tl.program_id(1)

    total = num_qo_heads // HEADS_PER_BLOCK
    if pid_batch >= batch_size or pid_block >= total:
        return

    kv_head = (pid_block * HEADS_PER_BLOCK) // GQA_RATIO

    ps = tl.load(kv_indptr_ptr + pid_batch)
    pe = tl.load(kv_indptr_ptr + pid_batch + 1)
    n_pages = pe - ps

    stride_kv_page = num_kv_heads * head_dim
    q_stride = num_qo_heads * head_dim
    log2e = 1.4426950408889634

    q_head_base = pid_block * HEADS_PER_BLOCK

    # Load 4 Q vectors
    q0 = tl.load(q_ptr + pid_batch * q_stride + (q_head_base + 0) * head_dim + tl.arange(0, BLOCK_D),
                 mask=tl.arange(0, BLOCK_D) < head_dim, other=0.0).to(tl.float32)
    q1 = tl.load(q_ptr + pid_batch * q_stride + (q_head_base + 1) * head_dim + tl.arange(0, BLOCK_D),
                 mask=tl.arange(0, BLOCK_D) < head_dim, other=0.0).to(tl.float32)
    q2 = tl.load(q_ptr + pid_batch * q_stride + (q_head_base + 2) * head_dim + tl.arange(0, BLOCK_D),
                 mask=tl.arange(0, BLOCK_D) < head_dim, other=0.0).to(tl.float32)
    q3 = tl.load(q_ptr + pid_batch * q_stride + (q_head_base + 3) * head_dim + tl.arange(0, BLOCK_D),
                 mask=tl.arange(0, BLOCK_D) < head_dim, other=0.0).to(tl.float32)

    m0 = tl.full((1,), -float("inf"), dtype=tl.float32)
    l0 = tl.zeros((1,), dtype=tl.float32)
    acc0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    m1 = tl.full((1,), -float("inf"), dtype=tl.float32)
    l1 = tl.zeros((1,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    m2 = tl.full((1,), -float("inf"), dtype=tl.float32)
    l2 = tl.zeros((1,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    m3 = tl.full((1,), -float("inf"), dtype=tl.float32)
    l3 = tl.zeros((1,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for page_idx in range(n_pages):
        kv_idx = tl.load(kv_indices_ptr + ps + page_idx)
        kv_offset = kv_idx * stride_kv_page + kv_head * head_dim

        k = tl.load(k_cache_ptr + kv_offset + tl.arange(0, BLOCK_D),
                    mask=tl.arange(0, BLOCK_D) < head_dim, other=0.0).to(tl.float32)
        v = tl.load(v_cache_ptr + kv_offset + tl.arange(0, BLOCK_D),
                    mask=tl.arange(0, BLOCK_D) < head_dim, other=0.0).to(tl.float32)

        # Head 0
        s = tl.sum(q0 * k) * sm_scale
        mn = tl.maximum(m0, s); sc = tl.exp(m0 - mn)
        l0 = l0 * sc + tl.exp(s - mn); m0 = mn
        acc0 = acc0 * sc + tl.exp(s - mn) * v
        # Head 1
        s = tl.sum(q1 * k) * sm_scale
        mn = tl.maximum(m1, s); sc = tl.exp(m1 - mn)
        l1 = l1 * sc + tl.exp(s - mn); m1 = mn
        acc1 = acc1 * sc + tl.exp(s - mn) * v
        # Head 2
        s = tl.sum(q2 * k) * sm_scale
        mn = tl.maximum(m2, s); sc = tl.exp(m2 - mn)
        l2 = l2 * sc + tl.exp(s - mn); m2 = mn
        acc2 = acc2 * sc + tl.exp(s - mn) * v
        # Head 3
        s = tl.sum(q3 * k) * sm_scale
        mn = tl.maximum(m3, s); sc = tl.exp(m3 - mn)
        l3 = l3 * sc + tl.exp(s - mn); m3 = mn
        acc3 = acc3 * sc + tl.exp(s - mn) * v

    # Write all 4 outputs
    acc0 = acc0 / l0
    tl.store(output_ptr + pid_batch * q_stride + (q_head_base + 0) * head_dim + tl.arange(0, BLOCK_D),
             acc0.to(tl.bfloat16), mask=tl.arange(0, BLOCK_D) < head_dim)
    tl.store(lse_ptr + pid_batch * num_qo_heads + (q_head_base + 0) + tl.arange(0, 1),
             (m0 + tl.log(l0)) * log2e)

    acc1 = acc1 / l1
    tl.store(output_ptr + pid_batch * q_stride + (q_head_base + 1) * head_dim + tl.arange(0, BLOCK_D),
             acc1.to(tl.bfloat16), mask=tl.arange(0, BLOCK_D) < head_dim)
    tl.store(lse_ptr + pid_batch * num_qo_heads + (q_head_base + 1) + tl.arange(0, 1),
             (m1 + tl.log(l1)) * log2e)

    acc2 = acc2 / l2
    tl.store(output_ptr + pid_batch * q_stride + (q_head_base + 2) * head_dim + tl.arange(0, BLOCK_D),
             acc2.to(tl.bfloat16), mask=tl.arange(0, BLOCK_D) < head_dim)
    tl.store(lse_ptr + pid_batch * num_qo_heads + (q_head_base + 2) + tl.arange(0, 1),
             (m2 + tl.log(l2)) * log2e)

    acc3 = acc3 / l3
    tl.store(output_ptr + pid_batch * q_stride + (q_head_base + 3) * head_dim + tl.arange(0, BLOCK_D),
             acc3.to(tl.bfloat16), mask=tl.arange(0, BLOCK_D) < head_dim)
    tl.store(lse_ptr + pid_batch * num_qo_heads + (q_head_base + 3) + tl.arange(0, 1),
             (m3 + tl.log(l3)) * log2e)


def run_kernel(
    q, k_cache, v_cache, kv_indptr, kv_indices,
    sm_scale, output, lse,
    batch_size, num_qo_heads, head_dim, num_kv_heads,
    num_pages, page_size, len_indptr, num_kv_indices,
    **_kwargs,
):
    gqa_ratio = num_qo_heads // num_kv_heads
    BLOCK_D = triton.next_power_of_2(head_dim)
    HEADS_PER_BLOCK = 4
    total_blocks = num_qo_heads // HEADS_PER_BLOCK

    grid = (batch_size, total_blocks)

    _gqa_paged_decode_kernel[grid](
        q, k_cache, v_cache, kv_indptr, kv_indices,
        sm_scale, output, lse,
        batch_size, num_qo_heads, head_dim, num_kv_heads,
        num_pages, num_kv_indices,
        HEADS_PER_BLOCK=HEADS_PER_BLOCK,
        GQA_RATIO=gqa_ratio,
        BLOCK_D=BLOCK_D,
        num_warps=8,
    )
