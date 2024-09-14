import torch
import torch.nn as nn
import torch.optim as optim
from data import GET_DATALOADER
from generator import GENERATOR
from discriminator import DISCRIMINATOR
from torch.amp import GradScaler
from tqdm import tqdm
from torchvision.utils import save_image
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter


device = 'cuda' if torch.cuda.is_available() else 'cpu'


#hyperparameters
lr_gen = 0.0001
lr_disc = 0.0001
epochs = 500
latent_dim = 256 #increase for more quality
batch_list = [128,64,32,16,8] #batch_size for each resoltion(default for 128 resolution training)
folder_path='image_folder_path'
save_path = 'saved_images\images_{}.png'
load_model = False

def train(epoch,global_steps,gen,disc,dataloader,optim_gen,optim_disc,scaler_gen,scaler_disc,\
  alpha,device,writer,batch_size,dataset,epoch_counter,step):
  
    def compute_gradient_penalty(disc, real_samples, fake_samples,alpha,step, device):
      # Random weight term for interpolation between real and fake samples
      alpha_gd = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
      # Get random interpolation between real and fake samples
      interpolates = (alpha_gd * real_samples + ((1 - alpha_gd) * fake_samples)).requires_grad_(True)
      d_interpolates = disc(interpolates,alpha,step)
      fake = torch.ones(real_samples.shape[0], 1).to(device)
      # Get gradient w.r.t. interpolates
      gradients = autograd.grad(
          outputs=d_interpolates,
          inputs=interpolates,
          grad_outputs=fake,
          create_graph=True,
          retain_graph=True,
          only_inputs=True,
      )[0]
      gradients = gradients.view(gradients.size(0), -1)
      gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
      return gradient_penalty
    
    def epoch_progression(img_start_size):
      # change accordingly for each image resolution
      if img_start_size == 4 or img_start_size == 8:
        return 35
      elif img_start_size == 16:
        return 60
      else:
        return 100
      
    epoch_counter += 1
    progress_epoch = epoch_progression(4*2**step)
    
    for batch_idx,image in tqdm(enumerate(dataloader)):
      gen.train()
      disc.train()
      global_steps += 1
      alpha += batch_size / (
            (30 * 0.5) * len(dataset)
        )
      alpha = min(1,alpha)
       
      
      with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        image = image.to(device)
        
        # disc training
        noise = torch.randn(image.shape[0],latent_dim).to(device)
        fake_img = gen(noise,alpha,step)
        critic_real = disc(image,alpha,step)
        critic_fake = disc(fake_img.detach(),alpha,step)
        loss_critic = -(torch.mean(critic_real)-torch.mean(critic_fake))
        
        gradient_penalty = compute_gradient_penalty(disc, image, fake_img.detach(), alpha,step,device)
        lambda_gp = 10  # Gradient penalty coefficient
        loss_critic += lambda_gp * gradient_penalty + (0.001 * torch.mean(critic_real**2))
        
      optim_disc.zero_grad()
      scaler_disc.scale(loss_critic).backward()
      scaler_disc.step(optim_disc)
      scaler_disc.update()
      writer.add_scalar('disc_loss',loss_critic.item(),global_steps)
        
      
      with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
          # gen training
          gen_fake = disc(fake_img,alpha,step)
          loss_gen = -torch.mean(gen_fake)
          
      optim_gen.zero_grad()
      scaler_gen.scale(loss_gen).backward()
      scaler_gen.step(optim_gen)
      scaler_gen.update()
      writer.add_scalar('gen_loss',loss_gen.item(),global_steps)
          
    torch.save(gen,'gen.pth')
    torch.save(disc,'disc.pth')
    with torch.no_grad():
        # genearting 16 images change accordingly
        noise = torch.randn(16,256).to(device)
        images = gen(noise,alpha=1,step=step)
        save_image(images,save_path.format(epoch))
        writer.add_images('gen_images',images,epoch)
        writer.add_images('original',image[:16],epoch)
    print('models saved..')
    
    if epoch_counter == progress_epoch:
        step += 1
        print('disc and gen progresed...')
        alpha = 1e-5  
        epoch_counter = 0
    return global_steps , alpha ,epoch_counter ,step

def main():
  gen = GENERATOR(in_channels=256,z_dim=256,w_dim=256).to(device)
  disc = DISCRIMINATOR(in_channels=256).to(device)
  
  scaler_gen = GradScaler('cuda')
  scaler_disc = GradScaler('cuda')
  
  writer = SummaryWriter()
  
  if load_model == True:
    gen = torch.load('generator_path')
    disc = torch.load('discriminator_path')
    gen = gen.to(device)
    disc = disc.to(device)
    
  optim_gen = optim.Adam([{"params": [param for name, param in gen.named_parameters() if "map" not in name]},
                        {"params": gen.map.parameters(), "lr": 1e-5}], lr=lr_gen, betas=(0.0, 0.99))
  optim_disc = optim.Adam(disc.parameters(),lr=lr_disc,betas=(0,0.99))
  
  
  Data = GET_DATALOADER()
  global_steps = 0
  epoch_counter = 0
  step = 1
  
  for epoch in range(epochs):
    alpha = 1e-5 if epoch == 0 else alpha
    batch_size = batch_list[step-1]
    print(batch_size)
    dataloader,dataset = Data.get_jpg_dataloader(image_dir=folder_path,image_size=(4*2**step,4*2**step),batch_size=batch_size)
    print(f"epoch {epoch} | img_size {4*2**step} | step {step}")
    print(f"alpha start: {alpha}, epoch_counter {epoch_counter}" )
    
    global_steps,alpha, epoch_counter, step = train(
      epoch,global_steps,gen,disc,dataloader,optim_gen,optim_disc,\
        scaler_gen,scaler_disc,alpha,device,writer,batch_size,dataset,epoch_counter,step
    )
    
if __name__=='__main__':
  main()
