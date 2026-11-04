import torch

print("PyTorch 版本：", torch.__version__)
print("GPU 可用：", torch.cuda.is_available())

if torch.cuda.is_available():
    print("显卡：", torch.cuda.get_device_name(0))
    x = torch.rand(3, 3, device="cuda")
    print("GPU 上的张量：\n", x)