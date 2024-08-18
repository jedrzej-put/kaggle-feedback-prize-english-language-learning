import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from transformers import AutoModel

from src.feedback_prize_english_language_learning.lib.config import ModuleConfig


class BertRegression(pl.LightningModule):
    def __init__(self, pretrained_model: str, learning_rate: float = ModuleConfig.learning_rate):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model, output_attentions=True)
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 2 * self.bert.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2 * self.bert.config.hidden_size, 1),
            nn.Sigmoid()
        )
        self.learning_rate = learning_rate

    def forward(self, input_ids, attention_mask, token_type_ids) -> torch.Tensor:
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        h_cls = bert_outputs.last_hidden_state[:, 0]  # cls token
        outputs = self.regressor(h_cls)
        return outputs

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        inputs, labels = batch
        outputs = self.forward(**inputs).squeeze(dim=1)
        train_loss = torch.sqrt(F.mse_loss(outputs, labels))
        self.log("train-loss", train_loss, sync_dist=True)
        return train_loss

    def validation_step(self, batch, batch_idx) -> None:
        inputs, labels = batch
        outputs = self.forward(**inputs).squeeze(dim=1)
        val_loss = torch.sqrt(F.mse_loss(outputs, labels))
        self.log("val-loss", val_loss, sync_dist=True)

        inverse_scaled_labels = labels * 4.0 + 1.0
        inverse_scaled_outputs = outputs * 4.0 + 1.0
        inverse_scaled_val_loss = torch.sqrt(F.mse_loss(inverse_scaled_outputs, inverse_scaled_labels))
        self.log("val-RMSE", inverse_scaled_val_loss, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx) -> None:
        inputs, labels = batch
        outputs = self.forward(**inputs).squeeze(dim=1)
        test_loss = torch.sqrt(F.mse_loss(outputs, labels))
        self.log("test-loss", test_loss, sync_dist=True)

        inverse_scaled_labels = labels * 4.0 + 1.0
        inverse_scaled_outputs = outputs * 4.0 + 1.0
        inverse_scaled_test_loss = torch.sqrt(F.mse_loss(inverse_scaled_outputs, inverse_scaled_labels))
        self.log("test-RMSE", inverse_scaled_test_loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-7)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val-RMSE"}
