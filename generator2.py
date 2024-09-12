import torch
import torch.nn as nn
# these classes(inside dotted block) are taken from official repo
#............................. offcial repo .................................#
# weighted linear layer #
class WSLinear(nn.Module):
    def __init__(
        self, in_features, out_features,):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = (2 / in_features)**0.5
        self.bias = self.linear.bias
        self.linear.bias = None

        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.linear(x * self.scale) + self.bias
      
# pixel norm #
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)
       
# weighted convolution layer#
class WSConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (2 / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)
      
# ..........................................................................#

# for reducing the channels 
channels_factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]

class MAPPING(nn.Module):
  def __init__(self, z_dim,w_dim):
      super().__init__()
      self.network = nn.Sequential(
        PixelNorm(),
        WSLinear(z_dim,z_dim),
        nn.ReLU(),
        WSLinear(z_dim,z_dim),
        nn.ReLU(),
        WSLinear(z_dim,z_dim),
        nn.ReLU(),
        WSLinear(z_dim,z_dim),
        nn.ReLU(),
        WSLinear(z_dim,z_dim),
        nn.ReLU(),
        WSLinear(z_dim,z_dim),
        nn.ReLU(),
        WSLinear(z_dim,z_dim),
        nn.ReLU(),
        WSLinear(z_dim,w_dim)
      )
  def forward(self, z):
      w = self.network(z)
      return w

class ADAPTIVE_IN(nn.Module):
  def __init__(self,out_channels,w_dim):
    super().__init__()
    self.norm = nn.InstanceNorm2d(out_channels,affine=False)
    self.style = WSLinear(w_dim,out_channels)
    self.shift = WSLinear(w_dim,out_channels)
    
  def forward(self,x,w):
    x = self.norm(x)
    return x * self.style(w).unsqueeze(2).unsqueeze(3) + self.shift(w).unsqueeze(2).unsqueeze(3)
    
class NOISE(nn.Module):
  def __init__(self,channels):
    super().__init__()
    self.channel_scaler = nn.Parameter(torch.zeros(1,channels,1,1))
  
  def forward(self,x):
    noise = torch.randn(x.shape[0],1,x.shape[2],x.shape[3],device=x.device)
    return x + noise * self.channel_scaler
    

# generator blocks
class GEN_BLOCK(nn.Module):
  def __init__(self,in_channels,out_channels,w_dim):
    super().__init__()
    self.conv1 = WSConv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
    self.conv2 = WSConv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
    self.noise1 = NOISE(out_channels)
    self.noise2 = NOISE(out_channels)
    self.adain1 = ADAPTIVE_IN(out_channels,w_dim)
    self.adain2 = ADAPTIVE_IN(out_channels,w_dim)
    self.act  = nn.LeakyReLU(0.2,inplace=True) 
    
  def forward(self,x,w):
    x = self.adain1(self.act(self.noise1(self.conv1(x))),w)
    x = self.adain2(self.act(self.noise2(self.conv2(x))),w)
    return x
  
class GENERATOR(nn.Module):
  def __init__(self,
               in_channels,
               z_dim,
               w_dim):
    super().__init__()
    
    # first block
    self.const_noise = nn.Parameter(torch.randn(1,in_channels,4,4))
    self.first_adain1 = ADAPTIVE_IN(in_channels,w_dim)
    self.first_adain2 = ADAPTIVE_IN(in_channels,w_dim)
    self.first_noise1 = NOISE(in_channels)
    self.first_noise2 = NOISE(in_channels)
    self.first_conv = nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1)
    self.first_act = nn.LeakyReLU(0.2)
    self.first_rgb = WSConv2d(in_channels,3,kernel_size=1,stride=1,padding=0)
    
    self.map = MAPPING(z_dim,w_dim)
    
    self.block_list = nn.ModuleList([])
    self.rgb_layers = nn.ModuleList([self.first_rgb])
    
    self.up = nn.Upsample(scale_factor=2,mode='bilinear')
    
    for i in range(len(channels_factors)-1):
      in_conv = int(in_channels*channels_factors[i])
      out_conv = int(in_channels*channels_factors[i+1])
      self.block_list.append(GEN_BLOCK(in_conv,out_conv,z_dim))
      self.rgb_layers.append(WSConv2d(out_conv,3,kernel_size=1,stride=1))
  
  def transition(self,x,y,alpha):
    return (1 - alpha) * x + alpha * y
   
  def forward(self,z,alpha,step):
      
      w = self.map(z)
      x = self.first_adain1(self.first_noise1(self.const_noise),w)
      y = self.first_conv(x)
      x = self.first_adain2(self.first_act(self.first_noise2(y)),w)
      
      if step == 0:
        return self.first_rgb(y)

      
      for i in range(step-1):
        x = self.up(x)
        x = self.block_list[i](x,w)
        
      y = self.block_list[step-1](self.up(x),w)
      y = self.rgb_layers[step](y)
      x = self.rgb_layers[step-1](self.up(x))
      x = self.transition(x,y,alpha)
        
      return x 

# z = torch.randn(1,256).to('cuda')    
# gen = GENERATOR(256,256,256).to('cuda')
# for i in range(1,6):
#   print(gen(z,1,i).shape)

# from torchsummary import summary
# summary(gen,input_size=(256,))