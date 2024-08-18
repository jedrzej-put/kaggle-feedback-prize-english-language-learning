import matplotlib.pyplot as plt
import pandas as pd

from src.feedback_prize_english_language_learning.lib.consts import PrizeTypes


def plot_value_counts(df: pd.DataFrame) -> None:
    value_counts: dict[str, int] = {}
    for column in PrizeTypes.all():
        value_counts[column] = df[column].value_counts().sort_index()
    value_counts_df: pd.DataFrame = pd.DataFrame(value_counts).fillna(0).astype(int)
    value_counts_df: pd.DataFrame = value_counts_df.unstack().reset_index()
    value_counts_df.columns = ['Column', 'Value', 'Count']
    pivot_df: pd.DataFrame = value_counts_df.pivot(index='Value', columns='Column', values='Count').fillna(0)
    pivot_df.plot(kind='bar', figsize=(12, 8))
    plt.xlabel('Values')
    plt.ylabel('Counts')
    plt.title('Value Counts for All Columns')
    plt.legend(title='Prize Types')


plt.show()
