import torchvision
import torchvision.transforms as transforms

# 指定存放数据集的路径
data_dir = "pycls/datasets/data/"

# 下载 CIFAR-10 数据集（训练集）
trainset = torchvision.datasets.CIFAR10(
    root=data_dir, train=True, download=True, transform=transforms.ToTensor()
)

# 下载 CIFAR-10 数据集（测试集）
testset = torchvision.datasets.CIFAR10(
    root=data_dir, train=False, download=True, transform=transforms.ToTensor()
)

print("CIFAR-10 数据集下载完成，存放在:", data_dir)
