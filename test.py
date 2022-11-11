import spectral
import pandas as pd
import torch
import time
import numpy as np
import ast
from configs.training_cfg import device
from PIL import Image
from models.ProbSparse_Self_Attention import ProbSparse_Self_Attention_Block
from models.attention_series import MCV_Split_Attn
from models.SoftPool import AttnPool_2d
from torch.cuda.amp import autocast

def test4():
    m = MCV_Split_Attn(attn=True, sample_factor=5).to(device)
    input = torch.rand(25,300).to(device)
    with autocast():
        output = m(input)
    print("")

def test_attn_pool2d():
    m = AttnPool_2d(2,8,16,24,4,8,2).to(device)
    input = torch.rand(4,20,20,8).to(device)
    output = m(input)
    print("")


def test_attn_2d():
    m1 = ProbSparse_Self_Attention_Block(input_dim=8, sparse=True, sf_q=5, sf_K=5).to(device)
    # m2 = ProbSparse_Self_Attention_Block(input_dim=16, sparse=False).to(device)

    input = torch.randn(11,148,8).to(device)


    # pool = SoftPool_1d(2,8,16,stride=2).to(device)
    # input = torch.randn(4,200,8).to(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:


    start.record()
    output = m1(input)
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))

    start.record()
    output = m1(input)
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))

    start.record()
    output = m1(input)
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))

    start.record()
    output = m1(input)
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))

    start.record()
    output = m1(input)
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))

    # prof.export_chrome_trace('./classic_attn_gpu.json')

def test2():
    input = torch.randn(4,200,200,8).to(device)

    pool = SoftPool_2d(2,8,4,stride=2).to(device)
    

    output = pool(input)

    print("")

if __name__ == "__main__":
    test4()
    print("")

