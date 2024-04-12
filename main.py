from trainer.amlsim import EGAT_trainer
from trainer.citation import Citation_trainer
import pytorch_lightning as pl
import yaml
import os
from logger import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


with open("config.yml", "r") as f:
    hparams = yaml.load(f, Loader=yaml.FullLoader)

Trainer = EGAT_trainer if hparams["net"] == "amlsim" else Citation_trainer

wandb_logger = WandbLogger(project="amlsim-batch", config=hparams)

trainer = pl.Trainer(benchmark=True,
                     callbacks=[ModelCheckpoint(monitor="val_loss"), EarlyStopping("val_loss", patience=200, strict=True)],
                     devices=1,
                     accelerator='gpu',
                     max_epochs=1000,
                     logger=wandb_logger)

# model = MLPTrainer(hparams["mlp"])
model = Trainer(wandb_logger.config)
wandb_logger.watch(model, log="all")

trainer.fit(model)

weight_path = trainer.checkpoint_callback.best_model_path
model = Trainer.load_from_checkpoint(weight_path)
print("Load from {}\n".format(weight_path))

trainer.test(model)
wandb_logger.log_metrics(trainer.callback_metrics)
