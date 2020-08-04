import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from dcgan import Generator, Discriminator

# Hyperparameters
## Data Loader
BATCH_SIZE = 128
NUM_WORKERS = 4
## Weight Initialization
WEIGHT_MEAN = 0
WEIGHT_STD = 0.02
## Adam Optimizer
LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.99
## Training
NUM_EPOCHS = 90
REPORT_RATE = 100

# Model Parameters
DEPTHS = 128
IMAGE_SIZE = 32
NUM_NOISES = 100
NUM_COLORS = 1

# Device Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mnist_train = torchvision.datasets.MNIST(
    "../Datasets/MNIST_PyTorch",
    train=True,
    transform=transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
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

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, WEIGHT_MEAN, WEIGHT_STD)
    # As the paper doesn't specify about the initialization of Batch Normalization,
    # I'll leave it default
    # elif classname.find('BatchNorm') != -1:
    #    pass
            
G = Generator(NUM_NOISES, NUM_COLORS, DEPTHS, IMAGE_SIZE).to(device)
G.apply(init_weights)

D = Discriminator(NUM_COLORS, DEPTHS, IMAGE_SIZE).to(device)
D.apply(init_weights)

# Assume BCELoss, since it's based on GAN (Goodfellow, 2014)
criterion = nn.BCELoss()
optimizerG = torch.optim.Adam(
    G.parameters(),
    lr=LEARNING_RATE,
    betas=[BETA1, BETA2]
)
optimizerD = torch.optim.Adam(
    D.parameters(),
    lr=LEARNING_RATE,
    betas=[BETA1, BETA2]
)

# On windows, using multi-worker DataLoader outside of
# if causes recursive process creation (runtime error)
if __name__ == "__main__":
    for epoch in range(NUM_EPOCHS):
        lossG = 0.0
        lossD = 0.0
        for i, data in enumerate(mnist_loader):
            data = data[0].to(device)
            # Train D with genuine data
            optimizerD.zero_grad()

            output = D(data)
            loss = criterion(output, torch.ones((BATCH_SIZE, 1)))
            loss.backward()
            lossD = loss.item()

            # Train D with fake data
            noise = torch.FloatTensor(NUM_NOISES).uniform_(-1, 1)
            fake = G(noise)

            output = D(fake)
            loss = criterion(output, torch.zeros((BATCH_SIZE, 1)))
            loss.backward()
            lossD += loss.item()

            optimizerD.step()

            # Train G
            optimizerG.zero_grad()

            output = D(fake)
            loss = criterion(output, torch.ones((BATCH_SIZE, 1)))
            loss.backward()
            lossG += loss.item()

            optimizerG.step()

            if i % REPORT_RATE == REPORT_RATE-1:
                print("Epoch: %d - [D: %d, G: %d]" % 
                    (epoch, lossD / REPORT_RATE*2, lossG / REPORT_RATE))
                lossG = 0.0
                lossD = 0.0

    torch.save(G, "./Models/G-%d.pt" % NUM_EPOCHS)
