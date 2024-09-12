import torch.nn as nn
import torch

channels_factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]

class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (2 / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)
      
      
class CRITIC_BLOCK(nn.Module):
  def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = WSConv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.conv2 = WSConv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.act = nn.LeakyReLU(0.2)
        
  def forward(self,x):
    x = self.act(self.conv1(x))
    return self.act(self.conv2(x))

class DISCRIMINATOR(nn.Module):
  def __init__(self,in_channels):
      super().__init__()
      self.blocks = nn.ModuleList([])
      self.rgb_layers = nn.ModuleList([])
      
      
      for i in range(len(channels_factors)-1,0,-1):
        in_conv = int(channels_factors[i] * in_channels)
        out_conv = int(channels_factors[i-1] * in_channels)
        self.blocks.append(CRITIC_BLOCK(in_conv,out_conv))
        self.rgb_layers.append(WSConv2d(3,in_conv,kernel_size=3,stride=1,padding=1))
        
      self.first_rgb = WSConv2d(3,in_channels,kernel_size=1,stride=1,padding=0)
      
      self.rgb_layers.append(self.first_rgb)
      self.down = nn.AvgPool2d(kernel_size=2,stride=2)
      
      
      self.final_block = nn.Sequential(
            # +1 to in_channels because we concatenate from MiniBatch std
            WSConv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(
                in_channels, 1, kernel_size=1, padding=0, stride=1
            ),
        )
      
      
  def transition(self,x,y,alpha):
    return (1 - alpha) * x + alpha * y
  
  # to know batch_variation
  def minibatch_std(self, x):
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        return torch.cat([x, batch_statistics], dim=1)
      
  def forward(self,x,alpha,step):
      
      if step == 0 :
        x = self.rgb_layers[-1](x)
        x = self.minibatch_std(x)
        return self.final_block(x).reshape(x.shape[0],-1)

      cur_step = len(self.blocks) - step
      
      y = self.rgb_layers[cur_step](x)
      y = self.blocks[cur_step](y)
      y = self.down(y)
      x = self.down(x)
      x = self.rgb_layers[cur_step+1](x)
      x = self.transition(x , y ,alpha)
      
      for i in range(cur_step+1,len(self.blocks)):
        x = self.blocks[i](x)
        x = self.down(x)
      
      x = self.minibatch_std(x)   
      return self.final_block(x).reshape(x.shape[0],-1)
    
      
   
# disc = DISCRIMINATOR(256).to('cuda')
# j=8
# for i in range(1,6):
#   z = torch.randn(4,3,j,j).to('cuda') 
#   print(disc(z,1,i).shape)
#   j = j*2