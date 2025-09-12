"""
Script to train a multi-symbol LSTM model to detect 'black swan' events or
high Risk:Reward trend beginnings from 5-minute candlestick data.
"""
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.metrics import AUC, Precision, Recall
import matplotlib.pyplot as plt
import random
from loguru import logger
from datetime import datetime

# --- Configuration ---
CONFIG = {
    "data": {
        "path": Path("data/processed/full_dataset_v3.parquet"),
        "seq_len": 50,
    },
    "model": {
        "lstm_units_1": 64,
        "lstm_units_2": 32,
        "dropout_1": 0.2,
        "dropout_2": 0.2,
        "prediction_threshold": 0.5,
    },
    "training": {
        "epochs": 50,
        "batch_size": 64,
        "early_stopping_patience": 5,
        "validation_split": 0.2,
    },
    "evaluation": {
        "trading_days_per_year": 252,
        "intervals_per_day": 288, # For 5-minute bars
    },
    "paths": {
        "model_save_dir": Path("models"),
        "scaler_path": Path("models/lstm_scaler.joblib"),
        "encoder_path": Path("models/lstm_encoder.joblib"),
        "model_path": Path("models/lstm_black_swan_multisymbol.keras"),
        "results_path": Path("models/lstm_training_results.json"),
        "plot_path": Path("models/training_history.png"),
    }
}

# Ensure model directory exists
CONFIG["paths"]["model_save_dir"].mkdir(parents=True, exist_ok=True)

# --- Logger Setup ---
logger.add("logs/lstm_training.log", rotation="10 MB")

# --- Reproducibility ---
def set_seeds(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    logger.info(f"Random seeds set to {seed}")

class SymbolDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of sequence data for a single symbol."""
    def __init__(self, symbol_df: pd.DataFrame, processed_features: pd.DataFrame, config: dict, for_validation=False):
        self.df = symbol_df # Keep for metadata access
        self.seq_len = config["data"]["seq_len"]
        self.batch_size = config["training"]["batch_size"]
        self.for_validation = for_validation

        # Pre-convert to NumPy for performance
        self.features_np = processed_features.loc[self.df.index].to_numpy()
        self.labels_np = self.df['label'].to_numpy()
        self.weights_np = self.df['trgt'].to_numpy()
        self.returns_np = self.df['ret'].to_numpy()
        self.sides_np = self.df['side'].to_numpy()

    def __len__(self):
        return int(np.ceil((len(self.df) - self.seq_len + 1) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.df) - self.seq_len + 1)

        batch_X, batch_y, batch_weights = [], [], []
        for i in range(start, end):
            batch_X.append(self.features_np[i:i + self.seq_len])
            batch_y.append(self.labels_np[i + self.seq_len - 1])
            batch_weights.append(self.weights_np[i + self.seq_len - 1])

        return np.array(batch_X), np.array(batch_y), np.array(batch_weights)
def load_data(file_path: Path) -> pd.DataFrame:
    """Loads data from a Parquet file."""
    logger.info(f"Loading data from {file_path}...")
    if not file_path.exists():
        logger.error(f"Data file not found at {file_path}")
        raise FileNotFoundError(f"Data file not found at {file_path}")
    df = pd.read_parquet(file_path)
    logger.info(f"Loaded data with shape: {df.shape}")
    return df.drop(columns=['t1'], errors='ignore')

class CombinedDataGenerator(tf.keras.utils.Sequence):
    """Combines multiple SymbolDataGenerators into a single generator."""
    def __init__(self, generators):
        self.generators = generators
        self.total_batches = sum(len(g) for g in self.generators)
        self.gen_map = []
        for i, gen in enumerate(self.generators):
            self.gen_map.extend([(i, j) for j in range(len(gen))])

    def __len__(self):
        return self.total_batches

    def __getitem__(self, idx):
        gen_idx, batch_idx = self.gen_map[idx]
        return self.generators[gen_idx][batch_idx]

def encode_and_scale_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Encodes categorical features and scales numerical features."""
    logger.info("Encoding symbols and scaling features...")
    feature_cols = [col for col in df.columns if col not in ['label', 'trgt', 'ret', 'side', 'symbol']]
    symbol_col = 'symbol'

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    symbol_encoded = encoder.fit_transform(df[[symbol_col]])
    symbol_encoded_df = pd.DataFrame(symbol_encoded, columns=encoder.get_feature_names_out([symbol_col]), index=df.index)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df[feature_cols])
    features_scaled_df = pd.DataFrame(features_scaled, columns=feature_cols, index=df.index)

    joblib.dump(scaler, config["paths"]["scaler_path"])
    joblib.dump(encoder, config["paths"]["encoder_path"])
    logger.info(f"Scaler and encoder saved to {config['paths']['model_save_dir']}")

    return pd.concat([features_scaled_df, symbol_encoded_df], axis=1)

def prepare_data(config: dict) -> tuple:
    """
    Performs time-based split per symbol and creates data generators.
    Returns a training generator, a validation generator, and a list of validation
    generators for later metric calculation.
    """
    df = load_data(config["data"]["path"])
    processed_features = encode_and_scale_features(df, config)

    train_gens, val_gens = [], []
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        split_point = int(len(symbol_df) * (1 - config["training"]["validation_split"]))

        train_df = symbol_df.iloc[:split_point]
        val_df = symbol_df.iloc[split_point:]

        if not train_df.empty:
            train_gens.append(SymbolDataGenerator(train_df, processed_features, config))
        if not val_df.empty:
            val_gens.append(SymbolDataGenerator(val_df, processed_features, config, for_validation=True))

    logger.info(f"Created {len(train_gens)} training generators and {len(val_gens)} validation generators.")
    return CombinedDataGenerator(train_gens), CombinedDataGenerator(val_gens), val_gens

def build_model(config: dict, input_shape: tuple) -> Sequential:
    """Builds and compiles the LSTM model."""
    model_config = config["model"]
    logger.info(f"Building LSTM model with input shape: {input_shape}")
    model = Sequential([
        LSTM(model_config["lstm_units_1"], return_sequences=True, input_shape=input_shape),
        Dropout(model_config["dropout_1"]),
        LSTM(model_config["lstm_units_2"]),
        Dropout(model_config["dropout_2"]),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[AUC(name="auc"), Precision(name="precision"), Recall(name="recall")])

    logger.info("Model built and compiled successfully.")
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    logger.info(f"Model Summary:\n{'\n'.join(stringlist)}")
    return model

def train_model(model: Sequential, train_gen, val_gen, config: dict) -> tf.keras.callbacks.History:
    """Trains the model using data generators with sample weights."""
    train_config = config["training"]
    logger.info("Starting model training with generators...")
    history = None
    try:
        early_stopping = EarlyStopping(monitor='val_loss', patience=train_config["early_stopping_patience"], restore_best_weights=True)
        lr_schedule = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=train_config["epochs"],
            callbacks=[early_stopping, lr_schedule],
            verbose=1
        )
        logger.info("Model training finished.")
    except tf.errors.OpError as e:
        logger.error(f"TensorFlow operation error during training: {e}")
        raise
    return history

def _calculate_sharpe(returns, periods):
    """Helper to calculate annualized Sharpe ratio."""
    mean_return = np.mean(returns)
    std_dev = np.std(returns)
    return (mean_return / std_dev) * np.sqrt(periods) if std_dev > 0 else 0.0

def _calculate_sortino(returns, periods):
    """Helper to calculate annualized Sortino ratio."""
    mean_return = np.mean(returns)
    downside_returns = returns[returns < 0]
    downside_std_dev = np.std(downside_returns)
    return (mean_return / downside_std_dev) * np.sqrt(periods) if downside_std_dev > 0 else 0.0

def find_optimal_threshold(y_true, y_pred_proba, returns, sides, config, metric='f1') -> float:
    """Finds the optimal prediction threshold to maximize a given metric (f1, sharpe, or sortino)."""
    best_score = -np.inf
    best_threshold = 0.5

    if metric == 'f1':
        for threshold in np.arange(0.05, 0.95, 0.05):
            y_pred = (y_pred_proba > threshold).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = threshold
    elif metric in ['sharpe', 'sortino']:
        eval_config = config["evaluation"]
        periods = eval_config["trading_days_per_year"] * eval_config["intervals_per_day"]
        for threshold in np.arange(0.05, 0.95, 0.05):
            y_pred = (y_pred_proba > threshold).astype(int)
            signals = y_pred.flatten() * sides
            strategy_returns = signals * returns
            if metric == 'sharpe':
                score = _calculate_sharpe(strategy_returns, periods)
            else: # sortino
                score = _calculate_sortino(strategy_returns, periods)
            if score > best_score:
                best_score = score
                best_threshold = threshold

    logger.info(f"Best score for metric '{metric}': {best_score:.4f} at threshold {best_threshold:.2f}")
    return best_threshold

def json_serializable_converter(o):
    """A robust JSON serializer for numpy and Path types."""
    if isinstance(o, (np.integer, np.int64)): return int(o)
    if isinstance(o, (np.floating, np.float32, np.float64)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, Path): return str(o)
    logger.warning(f"Cannot serialize type {type(o)}, converting to string.")
    return str(o)

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculates and logs classification metrics."""
    # Map multiclass labels {-1, 0, 1} to binary {0, 1} for metric calculation
    # We consider a hit on either profit-take or stop-loss barrier (1 or -1) as a positive event (1)
    y_true_binary = np.where(y_true == 0, 0, 1)

    accuracy = accuracy_score(y_true_binary, y_pred)
    precision = precision_score(y_true_binary, y_pred, zero_division=0)
    recall = recall_score(y_true_binary, y_pred, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred, zero_division=0)
    logger.info(f"Validation Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}

def calculate_financial_metrics(signals_df: pd.DataFrame, config: dict) -> dict:
    """Calculates and logs financial metrics."""
    eval_config = config["evaluation"]
    periods = eval_config["trading_days_per_year"] * eval_config["intervals_per_day"]

    signals_df['strategy_returns'] = signals_df['signal'] * signals_df['ret'] # Note: side is already incorporated in signal
    sharpe = _calculate_sharpe(signals_df['strategy_returns'], periods)
    sortino = _calculate_sortino(signals_df['strategy_returns'], periods)
    logger.info(f"Annualized Sharpe Ratio: {sharpe:.4f}, Annualized Sortino Ratio: {sortino:.4f}")
    return {'annualized_sharpe_ratio': sharpe, 'annualized_sortino_ratio': sortino}

def plot_training_history(history, config: dict):
    """Plots and saves the model's training history for new metrics."""
    if history is None: return
    metrics_to_plot = ['loss', 'auc', 'precision', 'recall']
    plt.figure(figsize=(12, 10))
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 2, i + 1)
        plt.plot(history.history[metric], label=f'Train {metric.capitalize()}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
        plt.title(f'Model {metric.capitalize()}')
        plt.legend()
    plt.tight_layout()
    plt.savefig(config["paths"]["plot_path"])
    plt.close()
    logger.info(f"Training history plot saved to {config['paths']['plot_path']}")

def save_artifacts(model, class_metrics: dict, fin_metrics: dict, history, config: dict):
    """Saves the model and evaluation results with error handling."""
    results = {
        'classification_metrics': class_metrics,
        'financial_metrics': fin_metrics,
        'training_history': history.history if history else None
    }
    try:
        with open(config["paths"]["results_path"], 'w') as f:
            json.dump(results, f, indent=4, default=json_serializable_converter)
        logger.info(f"Evaluation results saved to {config['paths']['results_path']}")
    except (TypeError, IOError) as e:
        logger.error(f"Failed to save results to JSON: {e}")

    try:
        model.save(config["paths"]["model_path"])
        logger.info(f"Model saved to {config['paths']['model_path']}")
    except (IOError, OSError) as e:
        logger.error(f"Failed to save model: {e}")

def train_and_evaluate(config: dict, train_gen, val_gen, val_gens_list):
    """Builds, trains, and evaluates the model using a robust, memory-efficient pipeline."""
    if not val_gens_list or not train_gen:
        logger.error("No training or validation data available. Skipping.")
        return

    first_batch_X, _, _ = val_gens_list[0][0]
    input_shape = (first_batch_X.shape[1], first_batch_X.shape[2])

    model = build_model(config, input_shape=input_shape)
    history = train_model(model, train_gen, val_gen, config)

    if history is None:
        logger.error("Skipping evaluation because model training failed.")
        return

    try:
        logger.info("Collecting predictions and ground truth in lockstep...")
        all_y_pred_proba, all_y_val, all_ret_val, all_side_val = [], [], [], []

        for i in range(len(val_gen)):
            X_batch, y_batch, _ = val_gen[i]
            y_pred_proba_batch = model.predict_on_batch(X_batch)

            # Map back to the original symbol generator to get metadata
            gen_idx, batch_idx_in_gen = val_gen.gen_map[i]
            symbol_gen = val_gen.generators[gen_idx]

            start = batch_idx_in_gen * symbol_gen.batch_size
            end = start + len(y_batch) # Use actual batch length

            metadata_indices = np.arange(start, end) + symbol_gen.seq_len - 1
            ret_batch = symbol_gen.returns_np[metadata_indices]
            side_batch = symbol_gen.sides_np[metadata_indices]

            all_y_pred_proba.append(y_pred_proba_batch)
            all_y_val.append(y_batch)
            all_ret_val.append(ret_batch)
            all_side_val.append(side_batch)

        y_pred_proba = np.concatenate(all_y_pred_proba)
        y_val = np.concatenate(all_y_val)
        ret_val = np.concatenate(all_ret_val)
        side_val = np.concatenate(all_side_val)

        # Find optimal threshold based on Sortino ratio
        optimal_threshold = find_optimal_threshold(y_val, y_pred_proba, ret_val, side_val, config, metric='sortino')
        y_pred = (y_pred_proba > optimal_threshold).astype(int)

        val_signals_df = pd.DataFrame({'ret': ret_val, 'side': side_val})
        val_signals_df['signal'] = y_pred.flatten() * val_signals_df['side']

        logger.info(f"Validation set true class distribution: {dict(zip(*np.unique(y_val, return_counts=True)))} ")
        logger.info(f"Generated signals distribution (1: Long, -1: Short, 0: Hold): \n{val_signals_df['signal'].value_counts()}")
        logger.info(f"Prediction probability percentiles (5, 25, 50, 75, 95): {np.percentile(y_pred_proba, [5, 25, 50, 75, 95])}")

        class_metrics = calculate_classification_metrics(y_val, y_pred)
        fin_metrics = calculate_financial_metrics(val_signals_df, config)
        plot_training_history(history, config)
        save_artifacts(model, class_metrics, fin_metrics, history, config)
    except tf.errors.OpError as e:
        logger.error(f"TensorFlow operation error during evaluation: {e}")
        raise

def main():
    """Main function to orchestrate the training pipeline."""
    # --- Artifact Versioning ---
    RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = CONFIG.copy()
    config["paths"]["model_path"] = config["paths"]["model_path"].with_name(f"{config['paths']['model_path'].stem}_{RUN_TIMESTAMP}.keras")
    config["paths"]["results_path"] = config["paths"]["results_path"].with_name(f"{config['paths']['results_path'].stem}_{RUN_TIMESTAMP}.json")
    config["paths"]["plot_path"] = config["paths"]["plot_path"].with_name(f"{config['paths']['plot_path'].stem}_{RUN_TIMESTAMP}.png")

    """Main function to orchestrate the training pipeline."""
    set_seeds(42)
    logger.info("Starting LSTM Black Swan detection model training...")
    try:
        train_generator, val_generator, val_gens_list = prepare_data(config)
        train_and_evaluate(config, train_generator, val_generator, val_gens_list)
        logger.info("Training pipeline finished successfully.")
    except FileNotFoundError:
        logger.error("Data file not found, aborting pipeline.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during the training pipeline: {e}")

if __name__ == "__main__":
    main()

