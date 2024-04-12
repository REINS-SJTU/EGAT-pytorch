import pytorch_lightning as pl
from argparse import Namespace
import typing as T
from model.net import CitationNet
import torch_geometric as pyg
import torch.nn.functional as F
import torch
from transforms.graph import AttachEdgeAttr, ToUndirected, Split, HandpickEdgeFeature, AddSelfLoop
from torch_geometric.transforms import Compose
from dataset import Cora, CiteSeer, Pubmed
from torch_geometric.datasets import Planetoid
from transforms.normalize import NormalizeFeatures, RowNormalizeFeatures


class Citation_trainer(pl.LightningModule):
    def __init__(self, hparams: T.Union[T.Dict, Namespace]):
        super(Citation_trainer, self).__init__()
        self.validation_outputs=[]
        if isinstance(hparams, Namespace):
            hparams = hparams
        else:
            hparams = Namespace(**hparams)
        self.save_hyperparameters(hparams)
        dataset = self.dataset()
        if isinstance(hparams, Namespace):
            hparams = vars(hparams)
        self.model = CitationNet(
            vertex_in_feature=dataset.num_node_features,
            edge_in_feature=dataset.num_edge_features,
            num_classes=dataset.num_classes,
            **hparams
        )

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        data: pyg.data.Data = batch

        y_pred = self.model(data)
        y_pred_train = y_pred[data.train_mask]
        y_true_train = data.y[data.train_mask]

        y_true_train = y_true_train.to(y_pred.device)

        loss = F.cross_entropy(y_pred_train, y_true_train)

        tensorboard_log = {
            "train_loss": loss
        }

        return {"loss": loss, "log": tensorboard_log}

    def validation_step(self, batch, batch_idx):

        data: pyg.data.Data = batch

        y_pred = self.model(data)

        y_pred_val = y_pred[data.val_mask]
        y_true_val = data.y[data.val_mask]

        y_true_val = y_true_val.to(y_pred.device)

        loss = F.cross_entropy(y_pred_val, y_true_val)

        choose = torch.argmax(y_pred_val, dim=-1)
        correct = (choose == y_true_val)
        total = y_pred_val.size(0)

        category_recall_correct = []
        category_recall_total = []
        for label in range(1, y_true_val.max() + 1):
            category_recall_correct.append(correct[y_true_val == label].sum())
            category_recall_total.append((y_true_val == label).sum())

        category_precise_correct = []
        category_precise_total = []
        for label in range(1, y_true_val.max() + 1):
            category_precise_correct.append(correct[choose == label].sum())
            category_precise_total.append((choose == label).sum())

        self.validation_outputs.append({
            "val_loss": loss,
            "correct": correct.sum(),
            "total": total
        })

        return {
            "val_loss": loss,
            "correct": correct.sum(),
            "total": total
        }

    def test_step(self, batch, batch_idx):
        data: pyg.data.Data = batch

        y_pred = self.model(data)

        y_pred_test = y_pred[data.test_mask]
        y_true_test = data.y[data.test_mask]

        y_true_test = y_true_test.to(y_pred.device)

        loss = F.cross_entropy(y_pred_test, y_true_test)

        choose = torch.argmax(y_pred_test, dim=-1)
        correct = (choose == y_true_test)
        total = y_pred_test.size(0)

        return {
            "test_loss": loss,
            "correct": correct.sum(),
            "total": total
        }

    def on_validation_epoch_end(self, outputs):
        outputs=self.validation_outputs
        avg_loss = torch.stack([output["val_loss"]
                                for output in outputs]).mean()
        correct = torch.stack([output["correct"] for output in outputs]).sum()
        total = sum([output["total"] for output in outputs])
        acc = correct.float() / total

        self.log("val_loss",avg_loss)
        self.log("val_acc",acc)

        self.validation_outputs=[]

        #tensorboard_log = {
         #   "val_loss": avg_loss,
          #  "val_acc": acc
        #}

        #return {"val_loss": avg_loss, "val_acc": acc, "log": tensorboard_log, 'progress_bar': {"val_loss": avg_loss, "val_acc": acc}}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([output["test_loss"]
                                for output in outputs]).mean()
        correct = torch.stack([output["correct"] for output in outputs]).sum()
        total = sum([output["total"] for output in outputs])
        acc = correct.float() / total

        tensorboard_log = {
            "test_loss": avg_loss,
            "test_acc": acc
        }

        return {"test_loss": avg_loss, "test_acc": acc, "log": tensorboard_log}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.l2norm)
        # schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(op, mode="min", patience=150)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                              factor=0.5, patience=40,
                                                              verbose=True,
                                                              min_lr=0.0001)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val_loss"  # 指定要监控的指标
        }
        }

    def dataset(self):
        seed = self.hparams.seed
        if self.hparams.dataset == "cora":
            return Cora(root='data/cora', transform=Compose([
                ToUndirected(),
                AttachEdgeAttr(),
                Split(0.03, seed),
                pyg.transforms.NormalizeFeatures()
            ]))
        elif self.hparams.dataset == "citeseer":
            return CiteSeer(root='data/citeseer', transform=Compose([
                ToUndirected(),
                AttachEdgeAttr(),
                # HandpickEdgeFeature(),
                # AddSelfLoop(fill_method="zero"),
                Split(0.01, seed),
                # MinMaxScaler(),
                pyg.transforms.NormalizeFeatures(),
            ]))
        elif self.hparams.dataset == "pubmed":
            return Pubmed(root="data/pubmed", transform=Compose([
                ToUndirected(),
                AttachEdgeAttr(),
                # AddSelfLoop(fill_method="mean"),
                Split(0.001, seed),
                pyg.transforms.NormalizeFeatures()
            ]))
        else:
            raise ValueError("Unknown dataset")

    def train_dataloader(self):
        dataset = self.dataset()
        return pyg.data.DataLoader(dataset, num_workers=0)

    def val_dataloader(self):
        dataset = self.dataset()
        return pyg.data.DataLoader(dataset, num_workers=0)

    def test_dataloader(self):
        dataset = self.dataset()
        return pyg.data.DataLoader(dataset, num_workers=0)
