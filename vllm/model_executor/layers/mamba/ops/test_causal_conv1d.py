import torch

from causal_conv1d import (
    causal_conv1d_fn as triton_conv_fn,
    causal_conv1d_update as triton_conv_update,
)
from causal_conv1d_pytorch import (
    causal_conv1d_fn as torch_conv_fn,
    causal_conv1d_update as torch_conv_update,
)


UPDATE_TEST_CASES = [(1, 4, 8, 3), (2, 3, 50, 5)]


def _compare_conv_update(batch: int, dim: int, seqlen: int, width: int) -> None:
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(0)
    x = torch.randn(batch, dim, seqlen, device=device, dtype=dtype)
    conv_state = torch.randn(batch, width - 1, dim, device=device, dtype=dtype)
    conv_state = conv_state.transpose(-1, -2)
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)
    cache_indices = torch.arange(batch, dtype=torch.int32, device=device)

    conv_state_triton = conv_state.clone()
    conv_state_torch = conv_state.clone()

    out_triton = triton_conv_update(
        x.clone(), conv_state_triton, weight, bias=bias, conv_state_indices=cache_indices
    )
    out_torch = torch_conv_update(
        x.clone(), conv_state_torch, weight, bias=bias, conv_state_indices=cache_indices
    )

    torch.testing.assert_close(out_triton, out_torch, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(conv_state_triton, conv_state_torch, rtol=1e-5, atol=1e-6)


def _build_query_start_loc(lengths: list[int], device: torch.device) -> torch.Tensor:
    prefix = torch.zeros(1, dtype=torch.int32, device=device)
    cumulative = torch.cumsum(torch.tensor(lengths, dtype=torch.int32, device=device), dim=0)
    return torch.cat([prefix, cumulative])


def _compare_conv_fn() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    torch.manual_seed(123)
    dim = 8
    width = 4
    lengths = [9, 5, 7]
    query_start_loc = _build_query_start_loc(lengths, device)
    total_tokens = int(query_start_loc[-1].item())

    x = torch.randn(dim, total_tokens, device=device, dtype=dtype)
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)
    conv_states = torch.randn(len(lengths), width - 1, dim, device=device, dtype=dtype)
    conv_states = conv_states.transpose(-1, -2)
    cache_indices = torch.arange(len(lengths), dtype=torch.int32, device=device)
    has_initial_state = torch.randint(0, 2, (len(lengths),), dtype=torch.bool, device=device)

    triton_states = conv_states.clone()
    torch_states = conv_states.clone()

    out_triton = triton_conv_fn(
        x.clone(),
        weight,
        bias,
        triton_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation="silu",
    )
    out_torch = torch_conv_fn(
        x.clone(),
        weight,
        bias,
        torch_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation="silu",
    )

    torch.testing.assert_close(out_triton, out_torch, rtol=1e-5, atol=1e-6)
    
    torch.testing.assert_close(triton_states, torch_states, rtol=1e-5, atol=1e-6)


def main() -> None:
    _compare_conv_fn()
    for case in UPDATE_TEST_CASES:
        _compare_conv_update(*case)
    print("All causal_conv1d reference tests passed.")


if __name__ == "__main__":
    main()