import torch
import torchvision

# Hyperparameters
BATCH_SIZE = 128
NUM_WORKERS = 4

mnist_train = torchvision.datasets.MNIST(
    "../Datasets/MNIST_PyTorch",
    train=True,
    transform=torchvision.transforms.ToTensor()
)
mnist_loader = torch.utils.data.DataLoader(
    mnist_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

# On windows, using multi-worker DataLoader outside of
# if causes recursive process creation (runtime error)
if __name__ == "__main__":
    pass