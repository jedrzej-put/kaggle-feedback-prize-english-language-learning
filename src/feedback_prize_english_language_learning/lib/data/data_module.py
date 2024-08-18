import os

import lightning.pytorch as pl
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src.feedback_prize_english_language_learning.lib.config import Config, DataModuleConfig


class NLPDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, max_length: int, label_col: str):
        self.dataframe: pd.DataFrame = dataframe
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.max_length: int = max_length
        self.label: str = label_col

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        row = self.dataframe.iloc[index]
        text = row['full_text']
        label = row[self.label]

        encoding: dict = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        inputs: dict[str, torch.Tensor] = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
        }
        return inputs, torch.tensor(label, dtype=torch.float32)


class NLPDataModule(pl.LightningDataModule):
    def __init__(self, label_col: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, pretrained_model_name: str, batch_size=DataModuleConfig.batch_size, num_workers: int = DataModuleConfig.num_workers, seed: int = Config.seed):
        super().__init__()
        self.label_col: str = label_col
        self.train_df: pd.DataFrame = train_df
        self.val_df: pd.DataFrame = val_df
        self.test_df: pd.DataFrame = test_df
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.pretrained_model: PreTrainedModel = AutoModel.from_pretrained(pretrained_model_name, output_attentions=True)
        self.batch_size: int = batch_size
        self.max_length: int = self.pretrained_model.config.max_position_embeddings
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage=None):
        self.train_dataset = NLPDataset(self.train_df, self.tokenizer, self.max_length, self.label_col)
        self.val_dataset = NLPDataset(self.val_df, self.tokenizer, self.max_length, self.label_col)
        self.test_dataset = NLPDataset(self.test_df, self.tokenizer, self.max_length, self.label_col)

    def prepare_data(self) -> None:
        pl.seed_everything(self.seed)
        # disable parrelism to avoid deadlocks
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)
