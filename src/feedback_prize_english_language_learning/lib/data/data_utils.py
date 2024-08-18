from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from src.feedback_prize_english_language_learning.lib.consts import PrizeTypes


def load_file(data_path: Path) -> pd.DataFrame:
    return pd.read_csv(data_path, sep=',', encoding='latin1')


def load_datasets(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        'train': load_file(data_dir / 'train.csv'),
        'test': load_file(data_dir / 'test.csv')
    }


def count_token(texts: pd.Series, tokenizer: PreTrainedTokenizerBase) -> pd.Series:
    return texts.apply(lambda x: len(tokenizer.tokenize(x)))


def scale_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scales values in the specified columns of the DataFrame from a range of 1.0-5.0 to 0-1.0.
    Creates new columns with the scaled values based on the original column names.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the columns to be scaled.
    columns (list): List of column names to be scaled.

    Returns:
    pd.DataFrame: DataFrame with the new scaled columns added.
    """
    for column, new_column_name in zip(PrizeTypes.all(), PrizeTypes.all_scaled()):
        df[new_column_name] = (df[column] - 1.0) / 4.0  # Scale from 1-5 to 0-1
    return df


def preprocessing_datasets(train: pd.DataFrame, test: pd.DataFrame, pretrained_model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pretrained_tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(pretrained_model_name)
    train['token_count'] = count_token(train['full_text'], pretrained_tokenizer)
    test['token_count'] = count_token(test['full_text'], pretrained_tokenizer)
    train = scale_values(train)
    return train, test


def select_features_split_datasets(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = df.loc[:, ["text_id", "full_text"]]
    y = df.loc[:, ["text_id", *PrizeTypes.all_scaled()]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=42)
    train_df: pd.DataFrame = pd.merge(X_train, y_train, on='text_id')
    val_df: pd.DataFrame = pd.merge(X_val, y_val, on='text_id')
    test_df: pd.DataFrame = pd.merge(X_test, y_test, on='text_id')
    return train_df, val_df, test_df
