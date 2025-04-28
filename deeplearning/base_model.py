import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
from tqdm import tqdm
from utils.constants import _WEATHER_COLUMNS

class BaseModel(nn.Module):
    
    def __init__(self, config):
        super(BaseModel, self).__init__()
        
        if config is None:
            raise ValueError("Model configuration required")

        self.model_name = config.get("name")
        self.setup_config = config.get("setup", {})
        self.training_config = config.get("training", {})

        self.device = self.setup_config.get("device", "cpu")
        if self.device != "cpu" and not torch.cuda.is_available():
            print("GPU unavailable. Switching back to CPU")
            self.device = "cpu"

        self.input_size = self.training_config.get("input_size", len(_WEATHER_COLUMNS))
        self.learning_rate = self.training_config.get("learning_rate", 0.001) 
        self.output_dim = self.setup_config.get("output_dim", 1) 
        self.forecast_steps = self.setup_config.get("forecast_steps", 7) 
        
        self.optimizer = None
        self.criterion = None
        self.normalization_params = None

    def setup_training(self):
        optimizer_name = self.training_config.get("optimizer", "Adam")
        optimizer_cls = getattr(torch.optim, optimizer_name)

        criterion_name = self.training_config.get("criterion", "MSELoss")
        criterion_cls = getattr(torch.nn, criterion_name)

        self.optimizer = optimizer_cls(self.parameters(), lr=self.learning_rate)
        self.criterion = criterion_cls()

    def fit(self, train_loader, val_loader=None, epochs=None):
        self.to(self.device)
        if epochs is None:
            epochs = self.training_config.get("epochs", 10)

        total_steps = epochs * len(train_loader)

        # Initialize regression metrics
        mse = MeanSquaredError().to(self.device)
        mae = MeanAbsoluteError().to(self.device)
        r2 = R2Score().to(self.device)

        global_loader = []
        for epoch in range(epochs):
            for batch in train_loader:
                global_loader.append((epoch, batch))

        with tqdm(total=total_steps, desc="Training", dynamic_ncols=True) as pbar:
            total_loss = 0
            self.train()
            for epoch, (X, y) in global_loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                preds = self.forward(X)
                loss = self.criterion(preds, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                # Denormalize for metrics if normalized
                if self.normalization_params is not None:
                    y_mean = torch.tensor(self.normalization_params["y_mean"], device=self.device)
                    y_std = torch.tensor(self.normalization_params["y_std"], device=self.device)
                    preds_denorm = preds * y_std.view(1, 1, -1) + y_mean.view(1, 1, -1)
                    y_denorm = y * y_std.view(1, 1, -1) + y_mean.view(1, 1, -1)
                else:
                    preds_denorm, y_denorm = preds, y

                # Reshape for metrics
                preds_flat = preds_denorm.view(preds.size(0), -1)
                y_flat = y_denorm.view(y.size(0), -1)

                # Compute metrics
                mse(preds_flat, y_flat)
                mae(preds_flat, y_flat)
                r2(preds_flat, y_flat)

                # Update progress bar
                pbar.set_postfix({
                    "epoch": epoch + 1,
                    "loss": total_loss / ((epoch * len(train_loader)) + 1),
                    "mse": mse.compute().item(),
                    "mae": mae.compute().item(),
                    "r2": r2.compute().item()
                })
                pbar.update(1)

        final_train_loss = total_loss / len(global_loader)
        print(f"\nFinal Train Loss: {final_train_loss:.4f}")
        print(f"Final Train MSE: {mse.compute().item():.4f}")
        print(f"Final Train MAE: {mae.compute().item():.4f}")
        print(f"Final Train R²: {r2.compute().item():.4f}")

        mse.reset()
        mae.reset()
        r2.reset()

        if val_loader:
            val_loss, val_metrics = self.evaluate(val_loader, metrics=[mse, mae, r2])
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation MSE: {val_metrics['mse']:.4f}")
            print(f"Validation MAE: {val_metrics['mae']:.4f}")
            print(f"Validation R²: {val_metrics['r2']:.4f}")

    def evaluate(self, val_loader, metrics=None):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                preds = self.forward(X)
                loss = self.criterion(preds, y)
                total_loss += loss.item()

                if metrics:
                    # Denormalize for metrics if normalized
                    if self.normalization_params is not None:
                        y_mean = torch.tensor(self.normalization_params["y_mean"], device=self.device)
                        y_std = torch.tensor(self.normalization_params["y_std"], device=self.device)
                        preds_denorm = preds * y_std.view(1, 1, -1) + y_mean.view(1, 1, -1)
                        y_denorm = y * y_std.view(1, 1, -1) + y_mean.view(1, 1, -1)
                    else:
                        preds_denorm, y_denorm = preds, y

                    # Reshape for metrics
                    preds_flat = preds_denorm.view(preds.size(0), -1)
                    y_flat = y_denorm.view(y.size(0), -1)
                    for metric in metrics:
                        metric(preds_flat, y_flat)

        avg_loss = total_loss / len(val_loader)
        if metrics:
            metric_values = {
                "mse": metrics[0].compute().item(),
                "mae": metrics[1].compute().item(),
                "r2": metrics[2].compute().item()
            }
            return avg_loss, metric_values
        return avg_loss

    def visualize_predictions(self, X_sample, y_sample, feature_names=None, start_date=None):
        self.eval()

        # Ensure all input tensors are on the same device as the model
        X_sample = X_sample.to(self.device)
        y_sample = y_sample.to(self.device)

        # Unsqueeze y_sample to match preds shape: (7, 23) -> (1, 7, 23)
        y_sample = y_sample.unsqueeze(0)

        with torch.no_grad():
            preds = self(X_sample.unsqueeze(0))

        # Denormalize data
        if self.normalization_params is not None:
            # Convert normalization params to tensors and move to the same device
            y_mean = torch.tensor(self.normalization_params["y_mean"], dtype=torch.float32).to(self.device)
            y_std = torch.tensor(self.normalization_params["y_std"], dtype=torch.float32).to(self.device)
            X_mean = torch.tensor(self.normalization_params["X_mean"], dtype=torch.float32).to(self.device)
            X_std = torch.tensor(self.normalization_params["X_std"], dtype=torch.float32).to(self.device)

            # Denormalize: shape (1, 7, num_features) for preds and y_sample, (1, 14, num_features) for X_sample
            preds = preds * y_std.view(1, 1, -1) + y_mean.view(1, 1, -1)
            y_sample = y_sample * y_std.view(1, 1, -1) + y_mean.view(1, 1, -1)
            X_sample = X_sample * X_std.view(1, 1, -1) + X_mean.view(1, 1, -1)
        else:
            raise ValueError("normalization_params must be set for denormalization")

        preds = preds.squeeze(0).cpu().numpy()
        y_true_future = y_sample.squeeze(0).cpu().numpy()
        history = X_sample.squeeze(0).cpu().numpy() 

        forecast_days = preds.shape[0]
        lookback_days = history.shape[0]

        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(history.shape[1])]

        if start_date is None:
            start_date = np.datetime64('today')

        past_dates = start_date - np.arange(lookback_days-1, -1, -1)
        future_dates = start_date + np.arange(1, forecast_days+1)
        full_dates = np.concatenate([past_dates, future_dates])

        _, axs = plt.subplots(len(feature_names), 1, figsize=(12, len(feature_names) * 4), sharex=True)

        if len(feature_names) == 1:
            axs = [axs]

        for i, ax in enumerate(axs):
            # Full ground truth
            full_truth = np.concatenate([history[:, i], y_true_future[:, i]])

            # Prediction starting from last history point
            preds_with_start = np.concatenate([[history[-1, i]], preds[:, i]])
            pred_dates_with_start = np.concatenate([[start_date], future_dates])

            raw_min_y = min(full_truth.min(), preds_with_start.min())
            padding = 0.1 * abs(raw_min_y)
            min_y = raw_min_y - padding

            ax.plot(full_dates, full_truth, color="#44c281", label="Ground Truth", linewidth=2)
            ax.fill_between(full_dates, min_y, full_truth, color="#9adbb9", alpha=0.5)

            ax.plot(pred_dates_with_start, preds_with_start, color="#3b6dc4", label="Prediction", linewidth=2)
            ax.fill_between(pred_dates_with_start, min_y, preds_with_start, color="#84a4db", alpha=0.4)

            ax.scatter(start_date, history[-1, i], color="#1D3557", s=10, zorder=5)

            # Add units to ylabel (assuming temperature is in °C)
            if feature_names[i] in ["temp", "tempmax", "tempmin"]:
                ax.set_ylabel(f"{feature_names[i]} (°C)", fontsize=12)
            else:
                ax.set_ylabel(feature_names[i], fontsize=12)
            
            ax.legend()

            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.suptitle(f"{self.model_name} 7 Day San Jose Weather Prediction", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()

    def forward(self, x):
        raise NotImplementedError("Each model must implement the forward() method.")