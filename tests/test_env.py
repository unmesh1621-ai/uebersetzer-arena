import torch


def test_cuda():
    assert torch.cuda.is_available()
    assert torch.cuda.is_bf16_supported()
