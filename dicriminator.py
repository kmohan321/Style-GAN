import torch.nn as nn
import torch

class DISC_BLOCK(nn.Module):
  def __init__(self,in_channels,out_channels):
    super().__init__()
    
    self.seq = nn.Sequential(
      nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1,stride=1),
      nn.LeakyReLU(0.2),
      nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=1),
      nn.LeakyReLU(0.2),
      nn.Upsample(scale_factor=0.5,mode='nearest') #downsampling
    )
    
  def forward(self,x):
    return self.seq(x)
  
class DISCRIMINATOR(nn.Module):
  def __init__(self,num_blocks):
    super().__init__()
    
    self.initial_depth = 0
    self.num_blocks = num_blocks
    self.latent_dim_blocks = [256,256,128,64,32,16,8] # increase as per num_blocks increase 
    self.first_layers = nn.ModuleList([
      nn.Conv2d(3,self.latent_dim_blocks[i],kernel_size=1,stride=1) for i in range(self.num_blocks)
    ])
    
    self.module_list_disc = nn.ModuleList([DISC_BLOCK(self.latent_dim_blocks[i+1],self.latent_dim_blocks[i]) \
      for i in range(self.num_blocks-1)]) 
    
    self.last_conv_block = nn.Sequential(
      nn.Conv2d(self.latent_dim_blocks[0],self.latent_dim_blocks[0],kernel_size=3,stride=1,padding=1),
      nn.LeakyReLU(0.2),
      nn.Conv2d(self.latent_dim_blocks[0],self.latent_dim_blocks[0],kernel_size=4,stride=1),
      nn.LeakyReLU(0.2)
    )
    self.last_linear = nn.Linear(self.latent_dim_blocks[0],1)
    

  def grow_discriminator(self):
    if self.initial_depth < self.num_blocks-1:
      self.initial_depth += 1 # num_blocks-1
    
  def forward(self,image):
    
    x = self.first_layers[self.initial_depth](image)  #RGB LAYER
    
    for block_idx in reversed(range(self.initial_depth)):
      x = self.module_list_disc[block_idx](x)
  
    return self.last_linear(self.last_conv_block(x).reshape(x.shape[0],-1))

# for checking
# disc = DISCRIMINATOR(num_blocks=7)
# a=4
# for i in range(7):
#   img = torch.randn(1,3,a,a)
#   print('image',img.shape)
#   print('block..',i+1)
#   print(disc(img).shape)
#   disc.grow_discriminator()
#   a = a*2
  
  

    
    