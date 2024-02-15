import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights

#Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 200
FEATURES_DISC = 64
FEATURES_GEN = 64

from torchvision import datasets, transforms

writer = SummaryWriter(f"logs")

# Define the transform to resize and tensorize the images
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize all images to 64x64
    transforms.ToTensor(),
])


# dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms,

#                          download=True)
dataset = datasets.ImageFolder(root="art_dataset", transform=transform)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
        fake = gen(noise)

        ### TRAIN DISCRIMINATOR ###
        ### TRAIN discriminator max log(D(x)) + log(1 - D(G(z)))
        # compute loss with real images
        disc_real = disc(real).reshape(-1)
        real_labels = 0.9 * torch.ones_like(disc_real) #soft labels for images
        loss_disc_real = criterion(disc_real, real_labels)
        
        
        # Compute loss with fake images
        disc_fake = disc(fake.detach()).reshape(-1) #using detach to avoid backprop through gen
        fake_labels = 0.1 + 0.1 * torch.rand_like(disc_fake) #soft labels for fake images
        loss_disc_fake = criterion(disc_fake, fake_labels)
        
        #Combine losses and update discriminator
        loss_disc = (loss_disc_real + loss_disc_fake / 2)
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        ### TRAIN GENERATOR ###
        ###Train generator min log()1 - D(G(z))) <--> max log(D(G(z))) 
        output = disc(fake).reshape(-1)
        # for generator i left target labels to be ones (since generator's goal is to fool the discriminator)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        
        ### DOUBLE GENERATOR OPTIMIZATION STEP ###
        for _ in range(2):  # Perform generator update twice
            # Regenerate fake images for fresh gradients
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake = gen(noise)
            output = disc(fake).reshape(-1)
            # For generator training, we aim to fool the discriminator, so target labels are ones
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
         # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )


            loss_disc_value = int(loss_disc.item())
            loss_gen_value = int(loss_gen.item())
            
            writer.add_scalar('disc_loss',loss_disc_value,epoch)
            writer.add_scalar('gen_loss',loss_gen_value,epoch)
            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1