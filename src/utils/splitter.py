from __future__ import annotations

import datetime
from typing import Literal, Optional, List, Tuple, Generator

import polars as pl
from dateutil.relativedelta import relativedelta

from src.settings import CVConfig

Interval = Tuple[datetime.datetime, datetime.datetime]


def parse_duration(duration: str) -> relativedelta:
    """
    Parse a duration string (e.g., "3mo", "3m", "1w", "10d") into a relativedelta object.

    :param duration: The duration string to parse.
    :type duration: str
    :return: A relativedelta object representing the parsed duration.
    :rtype: relativedelta
    """
    unit_char = duration[-1]

    if unit_char == "m":
        return relativedelta(months=int(duration[:-1]))
    if unit_char == "w":
        return relativedelta(weeks=int(duration[:-1]))
    if unit_char == "d":
        return relativedelta(days=int(duration[:-1]))

    unit_str = duration[-2:]
    value = int(duration[:-2])
    if unit_str == "mo":
        return relativedelta(months=value)
    raise ValueError(f"Unknown duration unit: {duration}")


class TimeSeriesSplitter:
    """
    Split time series data based on the specified configuration.
    """

    def __init__(
        self,
        *,
        strategy: Literal["sliding_window", "expanding_window"],
        n_splits: int,
        validation_duration: str,
        gap_duration: str,
        step_duration: str,
        train_duration: Optional[str] = None,
        test_end_date: str,
    ):
        self.strategy = strategy
        self.n_splits = n_splits
        self.validation_duration = validation_duration
        self.gap_duration = gap_duration
        self.step_duration = step_duration
        self.train_duration = train_duration
        self.test_end_date = datetime.datetime.strptime(test_end_date, "%Y-%m-%d")

    @classmethod
    def from_config(cls, config: CVConfig) -> "TimeSeriesSplitter":
        """
        Create a TimeSeriesSplitter instance from a CVConfig instance.

        :param config: The CVConfig instance.
        :type config: CVConfig
        :return: A TimeSeriesSplitter instance.
        :rtype: TimeSeriesSplitter
        """
        return cls(**config.__dict__)

    def sliding_window_split(
        self, full_data_start_date: datetime.datetime
    ) -> List[Tuple[Interval, Interval]]:
        """
        Split the data into training and validation sets based on the sliding window strategy.

        :param full_data_start_date: The start date of the full dataset.
        :type full_data_start_date: datetime.datetime
        :return: A list of tuples containing training and validation intervals.
        :rtype: List[Tuple[Interval, Interval]]
        """
        validation_duration = parse_duration(self.validation_duration)
        gap_duration = parse_duration(self.gap_duration)
        train_duration = (
            parse_duration(self.train_duration) if self.train_duration else None
        )
        step_duration = parse_duration(self.step_duration)

        splits = []
        for i in range(self.n_splits):
            current_offset = step_duration * i

            validation_end = self.test_end_date - current_offset
            validation_start = (
                validation_end - validation_duration + relativedelta(days=1)
            )
            train_end = validation_start - gap_duration - relativedelta(days=1)

            if train_duration:
                train_start = train_end - train_duration
            else:
                train_start = (
                    full_data_start_date + (self.n_splits - 1 - i) * step_duration
                )

            splits.append(
                ((train_start, train_end), (validation_start, validation_end))
            )

        return sorted(splits, key=lambda x: x[0][0])

    def expanding_window_split(
        self, full_data_start_date: datetime.datetime
    ) -> List[Tuple[Interval, Interval]]:
        """
        Split the data into training and validation sets based on the expanding window strategy.

        :param full_data_start_date: The start date of the full dataset.
        :type full_data_start_date: datetime.datetime
        :return: A list of tuples containing training and validation intervals.
        :rtype: List[Tuple[Interval, Interval]]
        """
        validation_duration = parse_duration(self.validation_duration)
        gap_duration = parse_duration(self.gap_duration)
        step_duration = parse_duration(self.step_duration)

        splits = []
        for i in range(self.n_splits):
            current_offset = step_duration * i

            validation_end = self.test_end_date - current_offset
            validation_start = (
                validation_end - validation_duration + relativedelta(days=1)
            )
            train_end = validation_start - gap_duration - relativedelta(days=1)
            train_start = full_data_start_date

            splits.append(
                ((train_start, train_end), (validation_start, validation_end))
            )

        return sorted(splits, key=lambda x: x[0][1])

    def split(
        self, full_data_start_date: datetime.datetime
    ) -> List[Tuple[Interval, Interval]]:
        """
        Split the data into training and validation sets based on the specified strategy.

        :param full_data_start_date: The start date of the full dataset.
        :type full_data_start_date: datetime.datetime
        :return: A list of tuples containing training and validation intervals.
        :rtype: List[Tuple[Interval, Interval]]
        """
        if self.strategy == "sliding_window":
            return self.sliding_window_split(full_data_start_date)
        elif self.strategy == "expanding_window":
            return self.expanding_window_split(full_data_start_date)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


class TimeSeriesDataSplitter:
    def __init__(
        self,
        *,
        strategy: Literal["sliding_window", "expanding_window"],
        date_column_name: str,
        n_splits: int,
        validation_duration: str,
        gap_duration: str,
        step_duration: str,
        train_duration: Optional[str] = None,
        test_end_date: str,
    ):
        self.splitter = TimeSeriesSplitter(
            strategy=strategy,
            n_splits=n_splits,
            validation_duration=validation_duration,
            gap_duration=gap_duration,
            step_duration=step_duration,
            train_duration=train_duration,
            test_end_date=test_end_date,
        )
        self.date_column_name = date_column_name

    @classmethod
    def from_config(cls, config: CVConfig) -> "TimeSeriesDataSplitter":
        """
        Create a TimeSeriesDataSplitter instance from a CVConfig instance.
        """
        return cls(**config.__dict__)

    def split(
        self, df: pl.DataFrame
    ) -> Generator[Tuple[pl.DataFrame, pl.DataFrame], None, None]:
        """
        Split the DataFrame into training and validation sets based on the specified date column.

        :param df: The DataFrame to split.
        :type df: pl.DataFrame

        :return: A generator that yields training and validation DataFrames.
        :rtype: Generator[Tuple[pl.DataFrame, pl.DataFrame], None, None]
        """
        full_data_start_date = df[self.date_column_name].min()
        for train_interval, val_interval in self.splitter.split(
            full_data_start_date=full_data_start_date
        ):
            train_data = df.filter(
                pl.col(self.date_column_name).is_between(
                    train_interval[0], train_interval[1]
                )
            )
            val_data = df.filter(
                pl.col(self.date_column_name).is_between(
                    val_interval[0], val_interval[1]
                )
            )
            yield train_data, val_data
