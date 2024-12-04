import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor

#Importing the dataset
train_directory = r"F:\Perception\CV-Proj-Contrastive\train_set"
val_directory = r"F:\Perception\CV-Proj-Contrastive\validation_set\val_images"
transform = transforms.Compose([transforms.Resize((80,80)),transforms.ToTensor()])
val_dataset = datasets.ImageFolder(root=val_directory,transform=transform)

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

'''
#Visualizing
for i, (data, _) in enumerate(trainset):
    if i == 2:
        a = data[2]
        plt.imshow(a,cmap='gray')
        plt.show()
'''

#Network architecture, augmentation and loss function 
#Utilising colour distortion(jitter), sharpening and crop&resize as augmentation tasks as mentioned by SimCLR                          

#For each image in the batch, we need to sample two different, random augmentations. CAN REDUCE CROP SIZE WITH MORE DATA
augmentations = v2.Compose([transforms.ToTensor(),transforms.Resize(80),v2.RandomCrop(size=(40)),v2.ColorJitter(brightness=0.5,hue=0.2)])
class ContrastiveTransformation(object):
    def __init__(self,transforms,samples):
        self.transforms = transforms
        self.views = samples

    def __call__(self,x):
        return torch.stack([self.transforms(x) for i in range(self.views)], dim=0)

#Train Dataset
train_dataset = datasets.ImageFolder(root=train_directory,transform=ContrastiveTransformation(augmentations,2))    

# A lightweight encoder to extract some features. Trying not to use something excessively heavy, such as ResNet18, owing to simplicity of data
class EncoderNetworkF(nn.Module):
    def __init__(self):
        super(EncoderNetworkF,self).__init__()
        self.conv1 = nn.Conv2d(3,12,3,padding='same')
        self.pool = nn.MaxPool2d(2,2) #20x20 feature map
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12,50,3,padding='same')
        self.bn2 = nn.BatchNorm2d(50) #10x10 feature map
        self.drop1 = nn.Dropout2d(p=0.2)

        self.fc1 = nn.Linear(50*10*10,300)
        self.fc2 = nn.Linear(300,100)

    def forward(self,x):
        x = self.pool(F.relu(self.bn1((self.conv1(x)))))  #Might want to drop the batchnorms
        x = self.pool(F.relu(self.bn2((self.conv2(x)))))

        x =  torch.flatten(x,1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x #Returns 100 (linear flattened) output

#Responsible for mapping the representation vector to the latent space where we apply contrastive loss
class ProjectionHead(nn.Module):
    def __init__(self,hidden_dim):
        super(ProjectionHead,self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(100,4*hidden_dim)
        self.fc2 = nn.Linear(4*hidden_dim,hidden_dim)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))   #Just a small MLP
        x = self.fc2(x)
        return x

#Using Lightning Module as a shortcut to having to write training and test loops.
class SimCLR(pl.LightningModule):

    def __init__(self,lr,temperature,weight_decay,max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        self.convnet = torchvision.models.resnet18(num_classes=100)
        self.projectionhead = ProjectionHead(15)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),lr=self.hparams.lr,weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr/50)
        return [optimizer],[lr_scheduler]
    
    def info_nce_loss(self,batch,mode='train'):
      
        imgs, _ = batch
        imgs = imgs.reshape(-1,3,40,40)
        #Aggregate into augmentedviews*images,H,W shape

        #Encode all images
        feat1 = self.convnet(imgs)
        features = self.projectionhead(feat1)

        #Find cosine similarity
        cos_sim = F.cosine_similarity(features[:,None,:],features[None,:,:],dim= -1)   
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
         # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode+'_loss', nll)
        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
                              cos_sim.masked_fill(pos_mask, -9e15)],
                             dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode+'_acc_top1', (sim_argsort == 0).float().mean())
        self.log(mode+'_acc_top5', (sim_argsort < 5).float().mean())
        self.log(mode+'_acc_mean_pos', 1+sim_argsort.float().mean())
        return nll
    
    def training_step(self,batch,batch_idx):
        return self.info_nce_loss(batch,mode='train')
    
    def validation_step(self,batch,batch_idx):
        self.info_nce_loss(batch,mode='val')

PATH = "F:\Perception\CV-Proj-Contrastive"
logger = TensorBoardLogger("logs",name='contrastive')
def train_simclr(batch_size,max_epochs=30,**kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(PATH,'SimCLR'),accelerator="gpu" if str(device).startswith("cuda") else "cpu",devices=1,logger=logger,max_epochs=max_epochs,callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc_top5'),
               LearningRateMonitor('epoch')])
    model = SimCLR(max_epochs=max_epochs,**kwargs)
    train_loader = DataLoader(train_dataset,batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size)
    trainer.fit(model,train_loader,val_loader)
    return model

simclr_model = train_simclr(batch_size=20,lr=1e-4,temperature=0.2,weight_decay=1e-4)

    



