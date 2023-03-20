import torch
import lightning as L
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

class NemesNet(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
     

        self.size = 8
        self.n = self.size *6
        self.n_classes = 4

        
        # Layer 1
        self.conv1 = nn.Conv2d(1, self.size , (1, 100), stride = (1, 16))
        self.batchnorm1 = nn.BatchNorm2d(self.size , False)
        
        # Layer 2
        self.conv2 = nn.Conv2d(self.size , self.size , (3, 1), stride = (1, 1))
        self.batchnorm2 = nn.BatchNorm2d(self.size , False)

        # Layer 3
        self.fc1 = nn.Linear(self.n, self.n_classes)
           # self.softmax = Softmax(dim=2)
        self.loss_fun = CrossEntropyLoss()

       
    def forward(self, x):
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.2)
       
        # Layer 2
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.2)
       

        x  = nn.MaxPool2d((1, 4))(x)
        #print(x.shape)
        # FC Layer
        x = x.view(-1, self.n)
       
    
        x = self.fc1(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        yy = self(x)
        loss = self.loss_fun(yy, y)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch
            yy = self(x)
            loss = self.loss_fun(yy, y)
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.2)
        #return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        return {"optimizer": optimizer}
