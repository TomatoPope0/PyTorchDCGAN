import os.path as path
import torch.cuda
import torchvision.transforms as T
from dcgan import Generator
from train import DEPTHS, IMAGE_SIZE, NUM_NOISES, NUM_COLORS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./Models/G-1.pt"
OUTPUT_PATH = "./Outputs/G-1/"
NUM_IMAGES = 10

G = Generator(DEPTHS, IMAGE_SIZE, NUM_NOISES, NUM_COLORS)
G.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

for i in range(NUM_IMAGES):
    noise = torch.FloatTensor(NUM_NOISES).uniform_(-1, 1)
    fake = G(noise)
    image = T.ToPILImage()(fake)
    image.save(path.join(OUTPUT_PATH, "image%d.bmp" % i))
