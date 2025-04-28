import os
import json
import torch
import textwrap
import numpy as np
import pandas as pd
from typing import List
from datasets import load_dataset, Dataset

from utils.constants import _WEATHER_COLUMNS


class WeatherPreprocessor:

    def __init__(
        self,
        csv_file: str,
        features: List[str] = None,
        lookback_days: int = 14,
        forecast_days: int = 7,
    ):

        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found at {csv_file}")

        self.df = None
        self.csv_file = csv_file
        self.lookback_days = lookback_days
        self.forecast_days = forecast_days
        self.features = features or _WEATHER_COLUMNS

    def clean_dataset(self) -> None:

        df = pd.read_csv(self.csv_file)

        # Data cleaning techniques...

        # Convert and split
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["day_of_year"] = df["datetime"].dt.dayofyear
        df["year"] = df["datetime"].dt.year
        df["month"] = df["datetime"].dt.month
        df["day"] = df["datetime"].dt.day
        df = df.drop(columns=["datetime"])

        df["precip"] = np.log1p(df["precip"])
        df["winddir_sin"] = np.sin(np.deg2rad(df["winddir"]))
        df["winddir_cos"] = np.cos(np.deg2rad(df["winddir"]))

        df = df[_WEATHER_COLUMNS]
        df = df.dropna()

        self.df = df

    def to_tensor(self, normalize: bool = True):

        if self.df is None:
            self.df = pd.read_csv(self.csv_file)
            self.df["datetime"] = pd.to_datetime(self.df["datetime"], errors="coerce")

        numeric_features = [
            f for f in self.features if np.issubdtype(self.df[f].dtype, np.number)
        ]
        self.features = numeric_features

        sequences, targets = [], []

        for i in range(len(self.df) - self.lookback_days - self.forecast_days):
            seq = self.df.iloc[i : i + self.lookback_days][self.features].values
            target = self.df.iloc[
                i + self.lookback_days : i + self.lookback_days + self.forecast_days
            ][self.features].values
            sequences.append(seq)
            targets.append(target)

        X = np.array(sequences)  # Shape: (num_samples, 14, num_features)
        y = np.array(targets)  # Shape: (num_samples, 7, num_features)

        if normalize:
            # Compute mean and std per feature
            X_mean = X.mean(
                axis=(0, 1)
            )  # Mean across samples and time steps, shape: (num_features,)
            X_std = X.std(
                axis=(0, 1)
            )  # Std across samples and time steps, shape: (num_features,)
            y_mean = y.mean(
                axis=(0, 1)
            )  # Mean across samples and time steps, shape: (num_features,)
            y_std = y.std(
                axis=(0, 1)
            )  # Std across samples and time steps, shape: (num_features,)

            # Normalize
            X = (X - X_mean[np.newaxis, np.newaxis, :]) / (
                X_std[np.newaxis, np.newaxis, :] + 1e-8
            )
            y = (y - y_mean[np.newaxis, np.newaxis, :]) / (
                y_std[np.newaxis, np.newaxis, :] + 1e-8
            )

            self.normalization_params = {
                "X_mean": X_mean,
                "X_std": X_std,
                "y_mean": y_mean,
                "y_std": y_std,
            }

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return X_tensor, y_tensor

    def generate_jsonl(self, output_dir: str) -> str:

        if self.df is None:
            self.df = pd.read_csv(self.csv_file)
            self.df["datetime"] = pd.to_datetime(self.df["datetime"], errors="coerce")

        sequences, targets = [], []

        for i in range(len(self.df) - self.lookback_days - self.forecast_days):
            seq = self.df.iloc[i : i + self.lookback_days][self.features].values
            target = self.df.iloc[
                i + self.lookback_days : i + self.lookback_days + self.forecast_days
            ][self.features].values
            sequences.append(seq)
            targets.append(target)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dataset_name = os.path.basename(self.csv_file).split(".")[0]
        jsonl_dataset = os.path.join(output_dir, f"{dataset_name}.jsonl")

        with open(jsonl_dataset, "w") as f:
            for i in range(len(sequences)):
                prompt = textwrap.dedent(
                    """\
                    Given the following weather data for the past 14 days in San Jose, CA, predict the weather for the next 7 days (all columns, comma-separated per day):

                    {}
                """.format(
                        ",".join(self.features)
                    )
                )

                for j in range(self.lookback_days):
                    day = self.df["datetime"].iloc[i + j].strftime("%Y-%m-%d")
                    data = sequences[i][j]
                    formatted_data = []
                    for x in data:
                        if isinstance(x, (int, float, np.number)):
                            formatted_data.append(f"{x:.1f}")
                        else:
                            formatted_data.append(str(x) if x else "None")
                    prompt += f"{day}," + ",".join(formatted_data) + "\n"

                prompt += "\nPredict the weather for the next 7 days (one row per day, comma-separated values):"

                completion = ""
                for day in range(self.forecast_days):
                    target_data = targets[i][day]
                    formatted_target = []
                    for x in target_data:
                        if isinstance(x, (int, float, np.number)):
                            formatted_target.append(f"{x:.1f}")
                        else:
                            formatted_target.append(str(x) if x else "None")
                    completion += ",".join(formatted_target)
                    if day < self.forecast_days - 1:
                        completion += "\n"

                json.dump({"input": prompt, "output": completion}, f)
                f.write("\n")

        return jsonl_dataset

    def tokenize_dataset(self, tokenizer, jsonl_file: str = None) -> Dataset:
        if jsonl_file is None:
            raise ValueError("JSONL file required for tokenization")
        if not os.path.exists(jsonl_file):
            raise FileNotFoundError(f"JSONL file not found at {jsonl_file}")

        dataset = load_dataset("json", data_files=jsonl_file, split="train")

        # Make sure pad_token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def tokenize_function(example):
            full_text = example["input"] + "\n" + example["output"]
            tokenized = tokenizer(
                full_text,
                truncation=True,
                padding="max_length",
                max_length=1024,
                return_tensors="pt",
            )
            labels = tokenized["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            return {
                "input_ids": tokenized["input_ids"].squeeze(),
                "attention_mask": tokenized["attention_mask"].squeeze(),
                "labels": labels.squeeze(),
            }

        tokenized_dataset = dataset.map(tokenize_function, batched=False)
        return tokenized_dataset
