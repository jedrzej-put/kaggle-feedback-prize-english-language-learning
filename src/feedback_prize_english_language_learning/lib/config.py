import os
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv

from src.feedback_prize_english_language_learning.lib.consts import PrizeTypes

load_dotenv()


ROOT_PATH: Path = Path(os.getenv('ROOT_PATH'))


@dataclass
class Config:
    data_dir: Path = ROOT_PATH / "data"
    cache_dir: Path = ROOT_PATH / "cache"
    log_dir: Path = ROOT_PATH / "logs"
    ckpt_dir: str = ROOT_PATH / "checkpoints"
    prof_dir: Path = ROOT_PATH / "logs" / "profiler"
    perf_dir: Path = ROOT_PATH / "logs" / "perf"
    seed: int = 42


@dataclass
class ModuleConfig:
    model_name: str = "microsoft/deberta-v3-base"
    learning_rate: float = 5e-05
    finetuned: str = "checkpoints/<some_ckpt>.ckpt"


@dataclass
class DataModuleConfig:
    prize_types: PrizeTypes = PrizeTypes()
    batch_size: int = 16
    test_size: float = 0.2
    label_column: str = PrizeTypes().cohesion
    num_workers: int = cpu_count()


@dataclass
class TrainerConfig:
    accelerator: str = "gpu"  # Trainer flag
    devices: Union[int, str] = 2  # Trainer flag
    strategy: str = "ddp"  # Trainer flag
    precision: Optional[str] = "16-mixed"  # Trainer flag
    max_epochs: int = 25  # Trainer flag
