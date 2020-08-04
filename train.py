import torch
import torchvision
import torchvision.transforms as transforms

# Hyperparameters
BATCH_SIZE = 128
NUM_WORKERS = 4

mnist_train = torchvision.datasets.MNIST(
    "../Datasets/MNIST_PyTorch",
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        # ToTensor() returns [0, 1]-ranged tensor.
        # Normalize() performs x' = (x - m)/s for every channel
        # Thus, Normalize((0.5), (0.5)) scales tensor to [-1, 1]
        # as (0-0.5)/0.5=-1, (1-0.5)/0.5=1
        transforms.Normalize((0.5), (0.5))
    ])
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