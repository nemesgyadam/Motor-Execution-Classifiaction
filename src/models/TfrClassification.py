import lightning as L

import torch
from torch.utils.data.dataset import T_co
from torch.nn import Linear, CrossEntropyLoss


class TfrClassification(L.LightningModule):
    def __init__(self, model_cls, model_weights, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = model_cls(weights=model_weights)
        self.model.fc = Linear(
            self.model.fc.in_features, len(cfg["event_name_cls_map"])
        ).to(
            "cuda"
        )  # TODO rm to cuda
        # self.softmax = Softmax(dim=2)
        self.loss_fun = CrossEntropyLoss()

        self.model.requires_grad_(True)
        self.model.fc.requires_grad_(True)
        print(self.model)

    def training_step(self, batch, batch_idx):
        x, y = batch
        yy = self.model(x)
        loss = self.loss_fun(yy, y)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch
            yy = self.model(x)
            loss = self.loss_fun(yy, y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, verbose=True, factor=0.2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
# TODO augmentations (torchvision.transforms)

