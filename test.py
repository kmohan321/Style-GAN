import torch
from torchvision.utils import save_image
device = 'cuda' if torch.cuda.is_available() else 'cpu'

gen = torch.load('generator_path')
gen = gen.to(device) 

num_images = 128

noise = torch.randn(num_images,256).to(device)
with torch.no_grad():
    images = gen(noise,1,4)
save_image(images,'test_images.png')

      


