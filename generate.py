import argparse
import os.path as path
import torch.cuda
import torchvision.transforms as T
from dcgan import Generator
from train import DEPTHS, IMAGE_SIZE, NUM_NOISES, NUM_COLORS

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", "-e", type=int, default=10)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./Models/G-%d.pt" % args.epochs
OUTPUT_PATH = "./Outputs/G-%d/" % args.epochs
NUM_IMAGES = 10

G = Generator(NUM_NOISES, NUM_COLORS, DEPTHS, IMAGE_SIZE)
G.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

for i in range(NUM_IMAGES):
    noise = torch.FloatTensor(NUM_NOISES).uniform_(-1, 1)
    fake = G(noise)
    image = T.ToPILImage()(fake.view(1, IMAGE_SIZE, IMAGE_SIZE))
    image.save(path.join(OUTPUT_PATH, "image%d.bmp" % i))
