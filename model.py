import torch
import torch.nn as nn

# generator 

# generator mapping block

class MAPPING(nn.Module):
  def __init__(self,input_size,output_size,layers):
    super().__init__()
    '''args:
    -----------------------------------
    input_size : int
      input to first layer
      
    output_size: int
      output of the last layer
      
    layers: int
      no.of layers
    '''
    self.first_layer = nn.Linear(input_size,output_size)
    self.module_list = nn.ModuleList([nn.Linear(output_size,output_size) for _ in range(layers-1)])
    
  def forward(self,x):
    # input_size (B,in_latent_dim)
    x = self.first_layer(x)
    for layer in self.module_list:
      x = layer(x)
    # output_size (B,out_latent_dim) if input_size==output_size
    return x

class GEN_BLOCK(nn.Module):
  def __init__(self,latent_dim,in_channels,out_channels,batch,first_block=False):
    super().__init__()
    '''args:
    --------------------------
    first_block : True
      whether to use first block or not
    '''
    self.first_block = first_block
    self.out_channels = out_channels
    self.batch = batch
    if first_block==True:
      # constant layer initialization with learnable parameters
      self.const_layer = nn.Parameter(torch.randn(self.batch,in_channels,4,4))
      self.act1 = nn.LeakyReLU(0.2)
      self.layer_norm1 = nn.InstanceNorm2d(in_channels,affine=False)
      self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,stride=1)
      self.act2 = nn.LeakyReLU(0.2)
      self.layer_norm2 = nn.InstanceNorm2d(out_channels,affine=False)
      
      # preparing the transformations
      self.A = nn.Linear(latent_dim,out_channels*4)
      self.B1 = nn.Conv2d(1,out_channels,kernel_size=3,stride=1,padding=1)
      self.B2 = nn.Conv2d(1,out_channels,kernel_size=3,stride=1,padding=1)
      
    else:
      self.up = nn.Upsample(scale_factor=2,mode='nearest')
      self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
      self.act1 = nn.LeakyReLU(0.2)
      self.layer_norm1 = nn.InstanceNorm2d(out_channels,affine=False)
      self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
      self.act2 = nn.LeakyReLU(0.2)
      self.layer_norm2 = nn.InstanceNorm2d(out_channels,affine=False)
      
      # preparing the transformations
      self.A = nn.Linear(latent_dim,out_channels*4)
      self.B1 = nn.Conv2d(1,out_channels,kernel_size=3,stride=1,padding=1)
      self.B2 = nn.Conv2d(1,out_channels,kernel_size=3,stride=1,padding=1)
  
  def forward(self,w,*args):
    y_scale1, y_shift1, y_scale2, y_shift2 = torch.chunk(self.A(w).reshape(w.shape[0],-1,1,1),chunks=4,dim=1)
    
    if self.first_block==True:
      x = self.const_layer
      x = self.act1(x)
      noise1 = torch.randn(x.shape[0],x.shape[2],x.shape[3])
      x = x + self.B1(noise1.unsqueeze(1))
      x = self.layer_norm1(x) * y_scale1 + y_shift1
      x = self.conv(x)
      x = self.act2(x)
      noise2 = torch.randn(x.shape[0],x.shape[2],x.shape[3])
      x = x + self.B2(noise2.unsqueeze(1))
      x = self.layer_norm2(x)*y_scale2 + y_shift2
      return x
    
    else:
      x = self.up(args[0])
      x = self.conv1(x)
      x = self.act1(x)
      noise1 = torch.randn(x.shape[0],x.shape[2],x.shape[3])
      x = x + self.B1(noise1.unsqueeze(1))
      x = self.layer_norm1(x)*y_scale1 + y_shift1
      x = self.conv2(x)
      x = self.act2(x)
      noise2 = torch.randn(x.shape[0],x.shape[2],x.shape[3])
      x = x + self.B2(noise2.unsqueeze(1))
      x = self.layer_norm2(x)*y_scale2 + y_shift2
      return x
      
class GENERATOR(nn.Module):
  def __init__(self,latent_dim,mapping_layers,batch):
    super().__init__()
    self.mapping = MAPPING(input_size=latent_dim,output_size=latent_dim,layers=mapping_layers)
    self.first_gen_block = GEN_BLOCK(latent_dim,512,512,batch,first_block=True)
    
    self.block2 = GEN_BLOCK(latent_dim,512,512,batch,first_block=False)
    self.block3 = GEN_BLOCK(latent_dim,512,512,batch,first_block=False)
    self.block4 = GEN_BLOCK(latent_dim,512,256,batch,first_block=False)
    
    self.block5 = GEN_BLOCK(latent_dim,256,128,batch,first_block=False)
    self.block6 = GEN_BLOCK(latent_dim,128,64,batch,first_block=False)
    self.block7 = GEN_BLOCK(latent_dim,64,32,batch,first_block=False)
    self.block8 = GEN_BLOCK(latent_dim,32,16,batch,first_block=False)
    
    self.final_layer = nn.Conv2d(16,3,kernel_size=1,stride=1)
    
  def forward(self,z):
    w = self.mapping(z)
    x = self.first_gen_block(w)
    
    x = self.block2(w,x)
    x = self.block3(w,x)
    x = self.block4(w,x)
    x = self.block5(w,x)
    x = self.block6(w,x)
    x = self.block7(w,x)
    x = self.block8(w,x)
    
    return self.final_layer(x)
    
from torchsummary import summary
model = GENERATOR(512,8,1)
summary(model)

    
    
  
      
      
      
      


