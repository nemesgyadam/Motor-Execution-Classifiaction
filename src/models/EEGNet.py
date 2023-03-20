import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

class EEGNet(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.n = 32*6
        self.n_classes = 4

        
        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 100), stride = (1, 16))
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        
        # Layer 2
        self.conv2 = nn.Conv2d(16, 32, (3, 1), stride = (1, 1))
        self.batchnorm2 = nn.BatchNorm2d(32, False)

        # Layer 3

        
      
        self.fc1 = nn.Linear(self.n, self.n_classes)
        

    def forward(self, x):
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.1)
       
        # Layer 2
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.1)
       

        x  = nn.MaxPool2d((1, 4))(x)
        #print(x.shape)
        # FC Layer
        x = x.view(-1, self.n)
       
    
        x = F.softmax(self.fc1(x))
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)

        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)