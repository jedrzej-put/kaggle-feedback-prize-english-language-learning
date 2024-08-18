import os, sys
from datetime import datetime
from time import perf_counter
from typing import Optional, Union
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.environ.get('PYTHONPATH'))
tqdm.pandas()
import lightning.pytorch as pl
import torch
from config import Config, DataModuleConfig, ModuleConfig, TrainerConfig
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler

from src.feedback_prize_english_language_learning.lib.data.data_utils import (
    load_datasets, 
    preprocessing_datasets,
    select_features_split_datasets,
)
from src.feedback_prize_english_language_learning.lib.data.data_module import NLPDataModule
from src.feedback_prize_english_language_learning.lib.models.BertRegression import BertRegression
from src.feedback_prize_english_language_learning.lib.config import Config, DataModuleConfig, ModuleConfig
from src.feedback_prize_english_language_learning.lib.utils import create_dirs, log_perf



def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    accelerator: str = TrainerConfig.accelerator,  # Trainer flag
    devices: Union[int, str] = TrainerConfig.devices,  # Trainer flag
    strategy: str = TrainerConfig.strategy,  # Trainer flag
    precision: Optional[str] = TrainerConfig.precision,  # Trainer flag
    max_epochs: int = TrainerConfig.max_epochs,  # Trainer flag
    lr: float = ModuleConfig.learning_rate,  # learning rate for LightningModule
    batch_size: int = DataModuleConfig.batch_size,  # batch size for LightningDataModule DataLoaders
    perf: bool = False,  # set to True to log training time and other run information
    profile: bool = False,  # set to True to profile. only use profiler to identify bottlenecks
) -> None:
    """a custom Lightning Trainer utility

    Note:
        for all Trainer flags, see https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags
    """

    # ## LightningDataModule ## #
    lit_datamodule = NLPDataModule(
        DataModuleConfig.label_column,
        train_df,
        val_df,
        test_df,
        pretrained_model_name=ModuleConfig.model_name,
        batch_size=batch_size,
        num_workers=DataModuleConfig.num_workers,
        seed=Config.seed,
    )

    # ## LightningModule ## #
    lit_module = BertRegression(pretrained_model=ModuleConfig.model_name, learning_rate=lr)

    # ## Lightning Trainer callbacks, loggers, plugins ## #

    loggers = [
        CSVLogger(
            save_dir=Config.log_dir,
            name="csv-logs",
        ),
        TensorBoardLogger(
            Config.log_dir / "tb_logs", 
            name="my_model"
        ),
    ]

    # set callbacks
    if perf:  # do not use EarlyStopping if getting perf benchmark
        callbacks = [
            ModelCheckpoint(
                dirpath=Config.ckpt_dir,
                filename="model",
            ),
            LearningRateMonitor(logging_interval='step'),
        ]
    else:
        callbacks = [
            EarlyStopping(monitor="val-RMSE", mode="min", verbose=True, patience=10),
            ModelCheckpoint(
                dirpath=Config.ckpt_dir,
                filename="model",
            ),
            LearningRateMonitor(logging_interval='step'),
        ]

    # set profiler
    if profile:
        profiler = PyTorchProfiler(dirpath=Config.prof_dir)
    else:
        profiler = None

    # ## create Trainer and call .fit ## #
    lit_trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_epochs=max_epochs,
        logger=loggers,
        callbacks=callbacks,
        profiler=profiler,
        log_every_n_steps=50,
        deterministic=True,
    )
    start = perf_counter()
    lit_trainer.fit(model=lit_module, datamodule=lit_datamodule)
    stop = perf_counter()

    lit_trainer.test(model=lit_module, datamodule=lit_datamodule)

    # ## log perf results ## #
    if perf:
        log_perf(start, stop, Config.perf_dir, lit_trainer)


if __name__ == "__main__":
    create_dirs([Config.cache_dir, Config.log_dir, Config.ckpt_dir, Config.prof_dir, Config.perf_dir])
    torch.set_float32_matmul_precision("medium")
    data: dict[str, pd.DataFrame] = load_datasets(Config.data_dir)
    train_df: pd.DataFrame = data['train']
    predict_df: pd.DataFrame = data['test']

    train_df, predict_df = preprocessing_datasets(train_df, predict_df, ModuleConfig.model_name)
    train_df, val_df, test_df = select_features_split_datasets(train_df, DataModuleConfig.test_size)
    print(len(train_df), len(val_df), len(test_df), len(predict_df))
    train(train_df, val_df, test_df, perf=False, profile=False)