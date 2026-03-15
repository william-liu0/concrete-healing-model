#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================
# Global plotting config
# =====================
# Matches MVPHeal81 current settings (titles shown, size 25)
DEFAULT_FONT_SIZES = {
    'title': 25,
    'xlabel': 20,
    'ylabel': 20,
    'legend': 20,
    'tick': 15,
    'annotation': 20,
}

SHOW_TITLES = True

def set_plot_style(font_sizes: dict | None = None, show_titles: bool | None = None) -> None:
    """Set matplotlib font sizes and title visibility globally for all plots.

    Parameters
    ----------
    font_sizes : dict | None
        Keys: 'title', 'xlabel', 'ylabel', 'legend', 'tick', 'annotation'.
    show_titles : bool | None
        If provided, override whether titles are shown (controls title size).
    """
    global DEFAULT_FONT_SIZES, SHOW_TITLES
    if font_sizes is None:
        font_sizes = DEFAULT_FONT_SIZES
    else:
        merged = DEFAULT_FONT_SIZES.copy()
        merged.update(font_sizes)
        font_sizes = merged
        DEFAULT_FONT_SIZES = merged

    if show_titles is not None:
        SHOW_TITLES = bool(show_titles)

    plt.rcParams.update({
        'font.size': font_sizes['tick'],
        'axes.titlesize': font_sizes['title'] if SHOW_TITLES else 0,
        'axes.labelsize': font_sizes['xlabel'],
        'xtick.labelsize': font_sizes['tick'],
        'ytick.labelsize': font_sizes['tick'],
        'legend.fontsize': font_sizes['legend'],
        'figure.titlesize': font_sizes['title'] if SHOW_TITLES else 0,
    })

# Initialize plotting style once on import
set_plot_style()
from sklearn.preprocessing import StandardScaler
# Optional dependencies (guarded)
try:
    from imblearn.over_sampling import RandomOverSampler
    HAS_IMBLEARN = True
except Exception:
    HAS_IMBLEARN = False
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: SHAP not available. Install with: pip install shap")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
from datetime import datetime, timedelta

try:
    import tensorflow as tf
    # Set random seed for TensorFlow reproducibility
    tf.keras.utils.set_random_seed(42)
    HAS_TF = True
except Exception:
    HAS_TF = False
import os
import json

# Load SGB healing time data
df = pd.read_csv("../SGB Healing Time Data/sgb_healing_time_data.csv")

# Rename columns to lowercase for consistency
df.columns = df.columns.str.lower()

# Select features and target
feature_cols = ["sigma", "gamma", "beta"]
target_col = "healing_time"

# Convert all numeric columns to float
for col in feature_cols + [target_col]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows where healing_time is None/NaN (cases where threshold wasn't reached)
df = df.dropna(subset=[target_col])

print(f"Dataset shape: {df.shape}")
print(f"Features: {feature_cols}")
print(f"Target: {target_col}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nDataset statistics:")
print(df.describe())

def plot_histograms(df, feature_cols, target_col):
    # Create results folder if it doesn't exist
    results_folder = "Histogram Plots"
    os.makedirs(results_folder, exist_ok=True)
    
    # Plot histograms for each feature
    for label in feature_cols:
        plt.figure(figsize=(10, 6))
        plt.hist(df[label], bins=30, color="blue", alpha=0.7, density=True)
        plt.title(f"Distribution of {label}")
        plt.ylabel("Probability Density")
        plt.xlabel(label)
        plt.grid(True, alpha=0.3)
        
        # Save the plot as PNG
        filename = f"{results_folder}/{label}_histogram.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved {filename}")
        
        plt.close()  # Close the figure to free memory

    # Plot histogram for target variable
    plt.figure(figsize=(10, 6))
    plt.hist(df[target_col], bins=30, color="green", alpha=0.7, density=True)
    plt.title(f"Distribution of {target_col}")
    plt.ylabel("Probability Density")
    plt.xlabel(target_col)
    plt.grid(True, alpha=0.3)
    
    filename = f"{results_folder}/{target_col}_histogram.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()

plot_histograms(df, feature_cols, target_col)

# Split data into training, validation, and testing sets
train, valid, test = np.split(df.sample(frac=1, random_state=42), 
                               [int(0.6*len(df)), int(0.8*len(df))])

def scale_dataset(dataframe, feature_cols, target_col, fit_scaler=None):
    """
    Scale features for regression (no oversampling needed)
    
    Parameters:
    -----------
    dataframe : pd.DataFrame
        Input dataframe
    feature_cols : list
        List of feature column names
    target_col : str
        Target column name
    fit_scaler : StandardScaler or None
        If provided, use this scaler to transform (for validation/test sets)
        If None, fit a new scaler (for training set)
    
    Returns:
    --------
    X : np.array
        Scaled features
    y : np.array
        Target values
    scaler : StandardScaler
        Fitted scaler
    """
    X = dataframe[feature_cols].values
    y = dataframe[target_col].values

    if fit_scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        scaler = fit_scaler
        X = scaler.transform(X)

    return X, y, scaler

# Scale the datasets
X_train, y_train, scaler = scale_dataset(train, feature_cols, target_col)
X_valid, y_valid, _ = scale_dataset(valid, feature_cols, target_col, fit_scaler=scaler)
X_test, y_test, _ = scale_dataset(test, feature_cols, target_col, fit_scaler=scaler)

print(f"\nTraining set size: {len(y_train)}")
print(f"Validation set size: {len(y_valid)}")
print(f"Test set size: {len(y_test)}")
print(f"\nTarget statistics (training set):")
print(f"  Mean: {np.mean(y_train):.2f}")
print(f"  Std: {np.std(y_train):.2f}")
print(f"  Min: {np.min(y_train):.2f}")
print(f"  Max: {np.max(y_train):.2f}")

# Section 1: Gaussian Process Regression
def gaussian_process_model(X_train, y_train, X_test, y_test, decimal_places=3):
    # Create Gaussian Process folder if it doesn't exist
    os.makedirs("Gaussian Process", exist_ok=True)
    
    print("Training Gaussian Process Regressor...")
    start_time = time.time()
    
    # Define kernel: RBF (Radial Basis Function) kernel with constant kernel
    # C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    
    gp_model = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        alpha=1e-6,
        normalize_y=True
    )
    gp_model.fit(X_train, y_train)

    training_time = time.time() - start_time
    print(f"Gaussian Process training completed in {training_time:.2f} seconds")
    
    print("Making predictions...")
    y_pred, y_std = gp_model.predict(X_test, return_std=True)
    
    # Calculate regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nGaussian Process Regression Results:")
    print(f"  MAE: {mae:.{decimal_places}f}")
    print(f"  MSE: {mse:.{decimal_places}f}")
    print(f"  RMSE: {rmse:.{decimal_places}f}")
    print(f"  R²: {r2:.{decimal_places}f}")
    print(f"  Mean prediction std: {np.mean(y_std):.{decimal_places}f}")
    
    # Save regression report to file
    with open("Gaussian Process/gaussian_process_regression_report.txt", "w") as f:
        f.write("Gaussian Process Regression Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Model Parameters:\n")
        f.write(f"  Kernel: RBF (Radial Basis Function)\n")
        f.write(f"  Alpha (noise level): 1e-6\n")
        f.write(f"  Normalize y: True\n")
        f.write(f"  Restarts optimizer: 10\n")
        f.write(f"  Training time: {training_time:.2f} seconds\n\n")
        f.write(f"Performance Metrics:\n")
        f.write(f"  Mean Absolute Error (MAE): {mae:.{decimal_places}f}\n")
        f.write(f"  Mean Squared Error (MSE): {mse:.{decimal_places}f}\n")
        f.write(f"  Root Mean Squared Error (RMSE): {rmse:.{decimal_places}f}\n")
        f.write(f"  R² Score: {r2:.{decimal_places}f}\n")
        f.write(f"  Mean prediction uncertainty (std): {np.mean(y_std):.{decimal_places}f}\n")
    
    print("Regression report saved to: Gaussian Process/gaussian_process_regression_report.txt")
    
    # Save the trained model
    import joblib
    model_filename = "Gaussian Process/gaussian_process_model.pkl"
    joblib.dump(gp_model, model_filename)
    print(f"Trained model saved to: {model_filename}")

    return gp_model

# Section 2: Ridge Regression
def ridge_regression_model(X_train, y_train, X_test, y_test, alpha=1.0, decimal_places=3):
    # Create Ridge Regression folder if it doesn't exist
    os.makedirs("Ridge Regression", exist_ok=True)
    
    print(f"Training Ridge Regressor (alpha={alpha})...")
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)

    y_pred = ridge_model.predict(X_test)
    
    # Calculate regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nRidge Regression Results:")
    print(f"  Alpha (regularization): {alpha}")
    print(f"  MAE: {mae:.{decimal_places}f}")
    print(f"  MSE: {mse:.{decimal_places}f}")
    print(f"  RMSE: {rmse:.{decimal_places}f}")
    print(f"  R²: {r2:.{decimal_places}f}")
    
    # Save regression report to file
    with open("Ridge Regression/ridge_regression_report.txt", "w") as f:
        f.write("Ridge Regression Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Model Parameters:\n")
        f.write(f"  Alpha (regularization strength): {alpha}\n\n")
        f.write(f"Performance Metrics:\n")
        f.write(f"  Mean Absolute Error (MAE): {mae:.{decimal_places}f}\n")
        f.write(f"  Mean Squared Error (MSE): {mse:.{decimal_places}f}\n")
        f.write(f"  Root Mean Squared Error (RMSE): {rmse:.{decimal_places}f}\n")
        f.write(f"  R² Score: {r2:.{decimal_places}f}\n")
    
    print("Regression report saved to: Ridge Regression/ridge_regression_report.txt")
    
    # Save the trained model
    import joblib
    model_filename = "Ridge Regression/ridge_regression_model.pkl"
    joblib.dump(ridge_model, model_filename)
    print(f"Trained model saved to: {model_filename}")

    return ridge_model

# Section 3: Random Forest Regression
def random_forest_model(X_train, y_train, X_test, y_test, decimal_places=3):
    # Create Random Forest folder if it doesn't exist
    os.makedirs("Random Forest", exist_ok=True)
    
    print("Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    
    # Calculate regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nRandom Forest Regression Results:")
    print(f"  MAE: {mae:.{decimal_places}f}")
    print(f"  MSE: {mse:.{decimal_places}f}")
    print(f"  RMSE: {rmse:.{decimal_places}f}")
    print(f"  R²: {r2:.{decimal_places}f}")
    
    # Save regression report to file
    with open("Random Forest/random_forest_regression_report.txt", "w") as f:
        f.write("Random Forest Regression Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.{decimal_places}f}\n")
        f.write(f"Mean Squared Error (MSE): {mse:.{decimal_places}f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.{decimal_places}f}\n")
        f.write(f"R² Score: {r2:.{decimal_places}f}\n")
    
    print("Regression report saved to: Random Forest/random_forest_regression_report.txt")
    
    # Save the trained model
    import joblib
    model_filename = "Random Forest/random_forest_model.pkl"
    joblib.dump(rf_model, model_filename)
    print(f"Trained model saved to: {model_filename}")

    return rf_model

# Section 4: XGBoost (Gradient Boosted Trees)
def xgboost_model(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=6, learning_rate=0.1, decimal_places=3):
    if not HAS_XGBOOST:
        print("XGBoost not available. Skipping XGBoost model.")
        print("Install with: pip install xgboost")
        return None
    
    # Create XGBoost folder if it doesn't exist
    os.makedirs("XGBoost", exist_ok=True)

    print("Training XGBoost Regressor (Gradient Boosted Trees)...")
    print(f"  Parameters: n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}")
    start_time = time.time()
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
        verbosity=1
    )
    xgb_model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"XGBoost training completed in {training_time:.2f} seconds")
    
    print("Making predictions...")
    y_pred = xgb_model.predict(X_test)
    
    # Calculate regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nXGBoost Regression Results:")
    print(f"  MAE: {mae:.{decimal_places}f}")
    print(f"  MSE: {mse:.{decimal_places}f}")
    print(f"  RMSE: {rmse:.{decimal_places}f}")
    print(f"  R²: {r2:.{decimal_places}f}")
    
    # Get feature importance
    feature_importance = xgb_model.feature_importances_
    
    # Save regression report to file
    with open("XGBoost/xgboost_regression_report.txt", "w") as f:
        f.write("XGBoost (Gradient Boosted Trees) Regression Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Model Parameters:\n")
        f.write(f"  n_estimators: {n_estimators}\n")
        f.write(f"  max_depth: {max_depth}\n")
        f.write(f"  learning_rate: {learning_rate}\n")
        f.write(f"  objective: reg:squarederror\n")
        f.write(f"  Training Time: {training_time:.2f} seconds\n\n")
        f.write(f"Performance Metrics:\n")
        f.write(f"  Mean Absolute Error (MAE): {mae:.{decimal_places}f}\n")
        f.write(f"  Mean Squared Error (MSE): {mse:.{decimal_places}f}\n")
        f.write(f"  Root Mean Squared Error (RMSE): {rmse:.{decimal_places}f}\n")
        f.write(f"  R² Score: {r2:.{decimal_places}f}\n\n")
        f.write(f"Feature Importance:\n")
        for i, importance in enumerate(feature_importance):
            f.write(f"  Feature {i+1}: {importance:.{decimal_places}f}\n")
    
    print("Regression report saved to: XGBoost/xgboost_regression_report.txt")
    
    # Save the trained model
    import joblib
    model_filename = "XGBoost/xgboost_model.pkl"
    joblib.dump(xgb_model, model_filename)
    print(f"Trained model saved to: {model_filename}")

    return xgb_model

# Section 5: Neural Network Regression
def neural_network_model(X_train, y_train, X_test, y_test, dropout_prob, lr, batch_size, epochs, decimal_places=3):
    
    if not HAS_TF:
        print("TensorFlow not available. Skipping neural network model.")
        return None
    
    def plot_history(history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Add overall title for the entire figure
        fig.suptitle("Loss and MAE Over the Epochs", fontsize=16, fontweight='bold', y=0.98)
        
        ax1.plot(history.history["loss"], label="Training Loss")
        ax1.plot(history.history["val_loss"], label="Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Mean Squared Error (MSE)")
        ax1.set_title("Loss Over the Epochs", fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(history.history["mae"], label="Training MAE")
        ax2.plot(history.history["val_mae"], label="Validation MAE")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Mean Absolute Error (MAE)")
        ax2.set_title("MAE Over the Epochs", fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        
        # Create Neural Network folder if it doesn't exist
        os.makedirs("Neural Network", exist_ok=True)
        
        # Save the training history plot
        plot_filename = "Neural Network/loss_mae_history.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {plot_filename}")
        
        plt.close()
    
    def train_model(X_train, y_train, X_test, y_test, dropout_prob, lr, batch_size, epochs):
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # Normalize target variable for better training
        from sklearn.preprocessing import StandardScaler
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_mean = y_scaler.mean_[0]
        y_std = y_scaler.scale_[0]
        
        # Build 3-32-32-1 architecture (3 inputs -> 32 -> 32 -> 1 output)
        nn_model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.Dropout(dropout_prob),
            tf.keras.layers.Dense(32, activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.Dropout(dropout_prob),
            tf.keras.layers.Dense(1, activation="linear")  # Linear activation for regression
        ])
        
        nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                        loss="mse", 
                        metrics=["mae", "mse"])
        
        # Custom callback to show progress and time estimation
        class TimeEstimationCallback(tf.keras.callbacks.Callback):
            def __init__(self, total_epochs):
                super().__init__()
                self.total_epochs = total_epochs
                self.start_time = None
                
            def on_train_begin(self, logs=None):
                self.start_time = time.time()
                print(f"Starting training for {self.total_epochs} epochs...")
                
            def on_epoch_end(self, epoch, logs=None):
                current_time = time.time()
                if self.start_time:
                    elapsed = current_time - self.start_time
                    avg_time_per_epoch = elapsed / (epoch + 1)
                    remaining_epochs = self.total_epochs - (epoch + 1)
                    estimated_remaining = avg_time_per_epoch * remaining_epochs
                    
                    # Format time remaining
                    if estimated_remaining > 60:
                        time_str = f"{estimated_remaining/60:.1f} minutes"
                    else:
                        time_str = f"{estimated_remaining:.1f} seconds"
                    
                    print(f"Epoch {epoch+1}/{self.total_epochs} - "
                          f"Loss: {logs['loss']:.4f}, "
                          f"MAE: {logs['mae']:.4f}, "
                          f"Val Loss: {logs['val_loss']:.4f}, "
                          f"Val MAE: {logs['val_mae']:.4f}, "
                          f"ETA: {time_str}")
        
        # Early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        # Create callback instance
        time_callback = TimeEstimationCallback(epochs)
        
        history = nn_model.fit(X_train, y_train_scaled, epochs=epochs, batch_size=batch_size, 
                              validation_split=0.2, verbose=0, 
                              callbacks=[time_callback, early_stopping])

        return nn_model, history, y_scaler
    
    print("Training neural network regression model...")
    start_time = time.time()
    
    # Train single model with provided parameters
    nn_model, history, y_scaler = train_model(X_train, y_train, X_test, y_test, dropout_prob, lr, batch_size, epochs)
    
    training_time = time.time() - start_time
    print(f"Neural network training completed in {training_time:.2f} seconds")
    
    plot_history(history)
    
    # Predict on scaled data, then inverse transform
    y_pred_scaled = nn_model.predict(X_test, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
    
    # Calculate regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nNeural Network Regression Results:")
    print(f"  MAE: {mae:.{decimal_places}f}")
    print(f"  MSE: {mse:.{decimal_places}f}")
    print(f"  RMSE: {rmse:.{decimal_places}f}")
    print(f"  R²: {r2:.{decimal_places}f}")
    
    # Create Neural Network folder if it doesn't exist
    os.makedirs("Neural Network", exist_ok=True)
    
    # Save regression report to file
    with open("Neural Network/neural_network_regression_report.txt", "w") as f:
        f.write("Neural Network Regression Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Model Parameters:\n")
        f.write(f"- Architecture: 3-32-32-1 (3 inputs -> 32 -> 32 -> 1 output)\n")
        f.write(f"- Activation: ReLU (hidden layers), Linear (output)\n")
        f.write(f"- Dropout: {dropout_prob}\n")
        f.write(f"- Learning Rate: {lr}\n")
        f.write(f"- Batch Size: {batch_size}\n")
        f.write(f"- Epochs: {epochs}\n")
        f.write(f"- Training Time: {training_time:.2f} seconds\n\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.{decimal_places}f}\n")
        f.write(f"Mean Squared Error (MSE): {mse:.{decimal_places}f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.{decimal_places}f}\n")
        f.write(f"R² Score: {r2:.{decimal_places}f}\n")
    
    print("Regression report saved to: Neural Network/neural_network_regression_report.txt")
    
    # Save the trained model
    model_filename = "Neural Network/neural_network_model.h5"
    nn_model.save(model_filename)
    print(f"Trained model saved to: {model_filename}")
    
    return nn_model

def create_model_comparison_chart(min_score=None, max_score=None, buffer=0.05, decimal_places=4):
    """
    Create comparison bar charts for all models' regression performance metrics
    by reading from saved regression report files
    
    Parameters:
    - min_score: Minimum y-axis value (auto-calculated if None)
    - max_score: Maximum y-axis value (auto-calculated if None) 
    - buffer: Buffer space above max_score (default: 0.05)
    - decimal_places: Number of decimal places for display
    """
    # Create Model Comparison folder
    os.makedirs("Model Comparison", exist_ok=True)
    
    # Define the model folders and their regression report files
    model_folders = {
        'Gaussian Process': 'Gaussian Process/gaussian_process_regression_report.txt',
        'Ridge Regression': 'Ridge Regression/ridge_regression_report.txt',
        'Random Forest': 'Random Forest/random_forest_regression_report.txt',
        'XGBoost': 'XGBoost/xgboost_regression_report.txt',
        'Neural Network': 'Neural Network/neural_network_regression_report.txt'
    }
    
    # Dictionary to store model results
    model_results = {}
    
    # Read each model's regression report and extract metrics
    import re
    for model_name, file_path in model_folders.items():
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                # Extract R² score
                r2_match = re.search(r'R² Score:\s*([-]?\d+\.\d+)', content, re.IGNORECASE)
                r2 = float(r2_match.group(1)) if r2_match else None
                
                # Extract MAE
                mae_match = re.search(r'Mean Absolute Error \(MAE\):\s*(\d+\.\d+)', content, re.IGNORECASE)
                mae = float(mae_match.group(1)) if mae_match else None
                
                # Extract MSE
                mse_match = re.search(r'Mean Squared Error \(MSE\):\s*(\d+\.\d+)', content, re.IGNORECASE)
                mse = float(mse_match.group(1)) if mse_match else None
                
                # Extract RMSE
                rmse_match = re.search(r'Root Mean Squared Error \(RMSE\):\s*(\d+\.\d+)', content, re.IGNORECASE)
                rmse = float(rmse_match.group(1)) if rmse_match else None
                
                if r2 is not None:
                    model_results[model_name] = {
                        'r2': r2,
                        'mae': mae,
                        'mse': mse,
                        'rmse': rmse
                    }
                    print(f"✅ Loaded metrics for {model_name}: R² = {r2:.{decimal_places}f}")
                else:
                    print(f"⚠️  Could not extract R² from {model_name}")
                
            except Exception as e:
                print(f"❌ Error reading {model_name}: {e}")
                continue
        else:
            print(f"⚠️  Report file not found for {model_name}: {file_path}")
    
    if not model_results:
        print("❌ No model results found. Make sure to run at least one model first.")
        return
    
    # Extract model names and R² values
    model_names = list(model_results.keys())
    r2_values = [model_results[model]['r2'] for model in model_names]
    
    # Auto-calculate min and max if not provided
    if min_score is None:
        min_score = min(0.0, min(r2_values) - 0.1)  # Ensure we show below 0 if needed
    if max_score is None:
        max_score = max(1.0, max(r2_values) + 0.05)
    
    # Create bar chart for R² values
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use single blue color for all bars
    bars = ax.bar(model_names, r2_values, color='#1f77b4', alpha=0.8, zorder=3)
    
    # Customize the chart with 50% larger font sizes
    ax.set_xlabel('Models', fontsize=23)
    ax.set_ylabel('R² Score', fontsize=23)
    ax.set_title('Model Performance Comparison (R² Score)', fontsize=24)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(True, alpha=0.3, zorder=1, axis='y')
    
    # Set y-axis limits
    ax.set_ylim(min_score, max_score + buffer)
    
    # Add value labels on top of bars
    for bar, r2_val in zip(bars, r2_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + buffer/4,
               f'{r2_val:.{decimal_places}f}',
               ha='center', va='bottom', fontsize=17)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the chart
    chart_filename = "Model Comparison/model_performance_comparison_r2.png"
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ Comparison chart saved to: {chart_filename}")
    
    plt.close()
    
    # Create heatmap with all regression metrics
    print("\n" + "="*60)
    print("CREATING PERFORMANCE HEATMAP (All Metrics)")
    print("="*60)
    
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 8))
    
    # Prepare data for heatmap (R², MAE, MSE, RMSE)
    metrics = ['R²', 'MAE', 'MSE', 'RMSE']
    heatmap_data = []
    metric_labels = []
    
    # R² values (higher is better)
    r2_row = [model_results[model]['r2'] for model in model_names]
    heatmap_data.append(r2_row)
    metric_labels.append('R² Score')
    
    # For error metrics, we'll normalize them for visualization (inverse scale)
    # Or we can show them separately. Let's show R² prominently and errors in a second heatmap
    
    # Create heatmap for R² only (for clarity)
    heatmap_data_array = np.array([r2_row])
    
    # Normalize R² values for color mapping (0 to 1 scale)
    # Find min and max R²
    r2_min = min(r2_values)
    r2_max = max(r2_values)
    r2_range = r2_max - r2_min if r2_max != r2_min else 1.0
    
    # Use a diverging colormap
    im = ax_heatmap.imshow(heatmap_data_array, cmap='RdYlGn', aspect='auto', 
                           vmin=max(-1.0, r2_min - 0.1), vmax=min(1.0, r2_max + 0.1))
    
    # Set labels and title with 50% larger font sizes
    ax_heatmap.set_xticks(range(len(model_names)))
    ax_heatmap.set_yticks([0])
    ax_heatmap.set_xticklabels(model_names, rotation=45, ha='right', fontsize=18)
    ax_heatmap.set_yticklabels(['R² Score'], fontsize=18)
    ax_heatmap.set_xlabel('Models', fontsize=21)
    ax_heatmap.set_ylabel('Metric', fontsize=21)
    ax_heatmap.set_title('Model Performance Heatmap (R² Score)', fontsize=24)
    
    # Add colorbar with larger font
    cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.8)
    cbar.set_label('R² Score', fontsize=18)
    cbar.ax.tick_params(labelsize=18)
    
    # Add text annotations on heatmap cells
    for j in range(len(model_names)):
        value = r2_values[j]
        text_color = 'black' if value > 0.5 else 'white'
        ax_heatmap.text(j, 0, f'{value:.{decimal_places}f}', 
                       ha='center', va='center', fontsize=17, fontweight='bold',
                       color=text_color)
    
    plt.tight_layout()
    
    # Save the heatmap
    heatmap_filename = "Model Comparison/model_performance_heatmap_r2.png"
    plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Heatmap saved to: {heatmap_filename}")
    
    plt.close()
    
    # Save the data used for the chart
    data_filename = "Model Comparison/model_comparison_data.txt"
    with open(data_filename, 'w') as f:
        f.write("MODEL PERFORMANCE COMPARISON DATA (REGRESSION)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Model Performance Metrics:\n")
        f.write("-" * 30 + "\n")
        for model_name in model_names:
            f.write(f"\n{model_name}:\n")
            f.write(f"  R² Score: {model_results[model_name]['r2']:.{decimal_places}f}\n")
            if model_results[model_name]['mae'] is not None:
                f.write(f"  MAE: {model_results[model_name]['mae']:.{decimal_places}f}\n")
            if model_results[model_name]['mse'] is not None:
                f.write(f"  MSE: {model_results[model_name]['mse']:.{decimal_places}f}\n")
            if model_results[model_name]['rmse'] is not None:
                f.write(f"  RMSE: {model_results[model_name]['rmse']:.{decimal_places}f}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("Chart Details:\n")
        f.write("- Bar colors: Single blue color (#1f77b4) for all models\n")
        f.write("- R² = 1.0: Perfect predictions\n")
        f.write("- R² = 0.0: Predictions as good as mean baseline\n")
        f.write("- R² < 0.0: Predictions worse than mean baseline\n")
        f.write("- Heatmap visualization with RdYlGn color scheme\n")
    
    print(f"✅ Comparison data saved to: {data_filename}")
    
    # Also save as JSON for programmatic access
    json_filename = "Model Comparison/model_comparison_data.json"
    with open(json_filename, 'w') as f:
        json.dump(model_results, f, indent=2)
    
    print(f"✅ Comparison data (JSON) saved to: {json_filename}")
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY (R² Scores)")
    print("="*60)
    for model_name in model_names:
        r2 = model_results[model_name]['r2']
        print(f"{model_name:25s}: R² = {r2:.{decimal_places}f}")
    print("="*60)

def make_knn_predictions(knn_model, X_test, y_test, decimal_places=3):
    """
    Make KNN predictions for custom input parameters
    """
    # Create predictions folder if it doesn't exist
    os.makedirs("Predictions", exist_ok=True)
    
    print("\n" + "="*60)
    print("KNN PREDICTIONS FOR CUSTOM PARAMETERS")
    print("="*60)
    print("Enter your own parameter values for prediction:")
    
    predictions = []
    while True:
        try:
            # Get user input for parameters
            print(f"\nEnter parameters (or 'quit' to exit):")
            sigma_input = input("Sigma (σ): ").strip()
            if sigma_input.lower() == 'quit':
                break
                
            gamma_input = input("Gamma (γ): ").strip()
            if gamma_input.lower() == 'quit':
                break
                
            time_input = input("Time Step (t): ").strip()
            if time_input.lower() == 'quit':
                break
            
            # Convert to float
            sigma = float(sigma_input)
            gamma = float(gamma_input)
            time_step = float(time_input)
            
            # Create feature array and scale it (using the same scaling as training data)
            # Note: This assumes the same scaling was applied during training
            features = np.array([[sigma, gamma, time_step]])
            
            # Make prediction
            pred_label = knn_model.predict(features)[0]
            pred_prob = knn_model.predict_proba(features)[0]
            
            # Store prediction
            prediction = {
                'sigma': sigma,
                'gamma': gamma,
                'timeStep': time_step,
                'predicted_label': pred_label,
                'prediction_confidence': max(pred_prob),
                'healed_probability': pred_prob[1] if len(pred_prob) > 1 else pred_prob[0],
                'not_healed_probability': pred_prob[0] if len(pred_prob) > 1 else 1 - pred_prob[0]
            }
            
            predictions.append(prediction)
            
            # Display result
            print(f"\n🎯 PREDICTION RESULT:")
            print(f"  Parameters: σ={sigma:.{decimal_places}f}, γ={gamma:.{decimal_places}f}, t={time_step:.{decimal_places}f}")
            print(f"  Prediction: {'🔴 HEALED' if pred_label == 1 else '🔵 NOT HEALED'}")
            print(f"  Confidence: {max(pred_prob):.{decimal_places}f}")
            print(f"  Healed Probability: {pred_prob[1] if len(pred_prob) > 1 else pred_prob[0]:.{decimal_places}f}")
            print(f"  Not Healed Probability: {pred_prob[0] if len(pred_prob) > 1 else 1 - pred_prob[0]:.{decimal_places}f}")
            
        except ValueError:
            print("❌ Invalid input! Please enter numeric values.")
        except KeyboardInterrupt:
            print("\n\nExiting prediction mode...")
            break
    
    # Save all predictions to file
    if predictions:
        with open("Predictions/knn_custom_predictions.txt", "w") as f:
            f.write("KNN CUSTOM PARAMETER PREDICTIONS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Custom Parameter Predictions:\n")
            f.write("-" * 30 + "\n")
            for i, pred in enumerate(predictions):
                f.write(f"Case {i+1}:\n")
                f.write(f"  Parameters: σ={pred['sigma']:.{decimal_places}f}, γ={pred['gamma']:.{decimal_places}f}, t={pred['timeStep']:.{decimal_places}f}\n")
                f.write(f"  Prediction: {'HEALED' if pred['predicted_label'] == 1 else 'NOT HEALED'}\n")
                f.write(f"  Confidence: {pred['prediction_confidence']:.{decimal_places}f}\n")
                f.write(f"  Healed Probability: {pred['healed_probability']:.{decimal_places}f}\n")
                f.write(f"  Not Healed Probability: {pred['not_healed_probability']:.{decimal_places}f}\n\n")
        
        print(f"\n✅ All predictions saved to: Predictions/knn_custom_predictions.txt")
        print(f"📊 Total predictions made: {len(predictions)}")

def make_naive_bayes_predictions(nb_model, X_test, y_test, decimal_places=3):
    """
    Make Naive Bayes predictions for custom input parameters
    """
    # Create predictions folder if it doesn't exist
    os.makedirs("Predictions", exist_ok=True)
    
    print("\n" + "="*60)
    print("NAIVE BAYES PREDICTIONS FOR CUSTOM PARAMETERS")
    print("="*60)
    print("Enter your own parameter values for prediction:")
    
    predictions = []
    while True:
        try:
            # Get user input for parameters
            print(f"\nEnter parameters (or 'quit' to exit):")
            sigma_input = input("Sigma (σ): ").strip()
            if sigma_input.lower() == 'quit':
                break
                
            gamma_input = input("Gamma (γ): ").strip()
            if gamma_input.lower() == 'quit':
                break
                
            time_input = input("Time Step (t): ").strip()
            if time_input.lower() == 'quit':
                break
            
            # Convert to float
            sigma = float(sigma_input)
            gamma = float(gamma_input)
            time_step = float(time_input)
            
            # Create feature array
            features = np.array([[sigma, gamma, time_step]])
            
            # Make prediction
            pred_label = nb_model.predict(features)[0]
            pred_prob = nb_model.predict_proba(features)[0]
            
            # Store prediction
            prediction = {
                'sigma': sigma,
                'gamma': gamma,
                'timeStep': time_step,
                'predicted_label': pred_label,
                'prediction_confidence': max(pred_prob),
                'healed_probability': pred_prob[1] if len(pred_prob) > 1 else pred_prob[0],
                'not_healed_probability': pred_prob[0] if len(pred_prob) > 1 else 1 - pred_prob[0]
            }
            
            predictions.append(prediction)
            
            # Display result
            print(f"\n🎯 PREDICTION RESULT:")
            print(f"  Parameters: σ={sigma:.{decimal_places}f}, γ={gamma:.{decimal_places}f}, t={time_step:.{decimal_places}f}")
            print(f"  Prediction: {'🔴 HEALED' if pred_label == 1 else '🔵 NOT HEALED'}")
            print(f"  Confidence: {max(pred_prob):.{decimal_places}f}")
            print(f"  Healed Probability: {pred_prob[1] if len(pred_prob) > 1 else pred_prob[0]:.{decimal_places}f}")
            print(f"  Not Healed Probability: {pred_prob[0] if len(pred_prob) > 1 else 1 - pred_prob[0]:.{decimal_places}f}")
            
        except ValueError:
            print("❌ Invalid input! Please enter numeric values.")
        except KeyboardInterrupt:
            print("\n\nExiting prediction mode...")
            break
    
    # Save all predictions to file
    if predictions:
        with open("Predictions/naive_bayes_custom_predictions.txt", "w") as f:
            f.write("NAIVE BAYES CUSTOM PARAMETER PREDICTIONS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Custom Parameter Predictions:\n")
            f.write("-" * 30 + "\n")
            for i, pred in enumerate(predictions):
                f.write(f"Case {i+1}:\n")
                f.write(f"  Parameters: σ={pred['sigma']:.{decimal_places}f}, γ={pred['gamma']:.{decimal_places}f}, t={pred['timeStep']:.{decimal_places}f}\n")
                f.write(f"  Prediction: {'HEALED' if pred['predicted_label'] == 1 else 'NOT HEALED'}\n")
                f.write(f"  Confidence: {pred['prediction_confidence']:.{decimal_places}f}\n")
                f.write(f"  Healed Probability: {pred['healed_probability']:.{decimal_places}f}\n")
                f.write(f"  Not Healed Probability: {pred['not_healed_probability']:.{decimal_places}f}\n\n")
        
        print(f"\n✅ All predictions saved to: Predictions/naive_bayes_custom_predictions.txt")
        print(f"📊 Total predictions made: {len(predictions)}")

def make_logistic_regression_predictions(lr_model, X_test, y_test, decimal_places=3):
    """
    Make Logistic Regression predictions for custom input parameters
    """
    # Create predictions folder if it doesn't exist
    os.makedirs("Predictions", exist_ok=True)
    
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION PREDICTIONS FOR CUSTOM PARAMETERS")
    print("="*60)
    print("Enter your own parameter values for prediction:")
    
    predictions = []
    while True:
        try:
            # Get user input for parameters
            print(f"\nEnter parameters (or 'quit' to exit):")
            sigma_input = input("Sigma (σ): ").strip()
            if sigma_input.lower() == 'quit':
                break
                
            gamma_input = input("Gamma (γ): ").strip()
            if gamma_input.lower() == 'quit':
                break
                
            time_input = input("Time Step (t): ").strip()
            if time_input.lower() == 'quit':
                break
            
            # Convert to float
            sigma = float(sigma_input)
            gamma = float(gamma_input)
            time_step = float(time_input)
            
            # Create feature array
            features = np.array([[sigma, gamma, time_step]])
            
            # Make prediction
            pred_label = lr_model.predict(features)[0]
            pred_prob = lr_model.predict_proba(features)[0]
            
            # Store prediction
            prediction = {
                'sigma': sigma,
                'gamma': gamma,
                'timeStep': time_step,
                'predicted_label': pred_label,
                'prediction_confidence': max(pred_prob),
                'healed_probability': pred_prob[1] if len(pred_prob) > 1 else pred_prob[0],
                'not_healed_probability': pred_prob[0] if len(pred_prob) > 1 else 1 - pred_prob[0]
            }
            
            predictions.append(prediction)
            
            # Display result
            print(f"\n🎯 PREDICTION RESULT:")
            print(f"  Parameters: σ={sigma:.{decimal_places}f}, γ={gamma:.{decimal_places}f}, t={time_step:.{decimal_places}f}")
            print(f"  Prediction: {'🔴 HEALED' if pred_label == 1 else '🔵 NOT HEALED'}")
            print(f"  Confidence: {max(pred_prob):.{decimal_places}f}")
            print(f"  Healed Probability: {pred_prob[1] if len(pred_prob) > 1 else pred_prob[0]:.{decimal_places}f}")
            print(f"  Not Healed Probability: {pred_prob[0] if len(pred_prob) > 1 else 1 - pred_prob[0]:.{decimal_places}f}")
            
        except ValueError:
            print("❌ Invalid input! Please enter numeric values.")
        except KeyboardInterrupt:
            print("\n\nExiting prediction mode...")
            break
    
    # Save all predictions to file
    if predictions:
        with open("Predictions/logistic_regression_custom_predictions.txt", "w") as f:
            f.write("LOGISTIC REGRESSION CUSTOM PARAMETER PREDICTIONS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Custom Parameter Predictions:\n")
            f.write("-" * 30 + "\n")
            for i, pred in enumerate(predictions):
                f.write(f"Case {i+1}:\n")
                f.write(f"  Parameters: σ={pred['sigma']:.{decimal_places}f}, γ={pred['gamma']:.{decimal_places}f}, t={pred['timeStep']:.{decimal_places}f}\n")
                f.write(f"  Prediction: {'HEALED' if pred['predicted_label'] == 1 else 'NOT HEALED'}\n")
                f.write(f"  Confidence: {pred['prediction_confidence']:.{decimal_places}f}")
                f.write(f"  Healed Probability: {pred['healed_probability']:.{decimal_places}f}\n")
                f.write(f"  Not Healed Probability: {pred['not_healed_probability']:.{decimal_places}f}\n\n")
        
        print(f"\n✅ All predictions saved to: Predictions/logistic_regression_custom_predictions.txt")
        print(f"📊 Total predictions made: {len(predictions)}")

def make_svm_predictions(svm_model, X_test, y_test, decimal_places=3):
    """
    Make SVM predictions for custom input parameters
    """
    # Create predictions folder if it doesn't exist
    os.makedirs("Predictions", exist_ok=True)
    
    print("\n" + "="*60)
    print("SVM PREDICTIONS FOR CUSTOM PARAMETERS")
    print("="*60)
    print("Enter your own parameter values for prediction:")
    
    predictions = []
    while True:
        try:
            # Get user input for parameters
            print(f"\nEnter parameters (or 'quit' to exit):")
            sigma_input = input("Sigma (σ): ").strip()
            if sigma_input.lower() == 'quit':
                break
                
            gamma_input = input("Gamma (γ): ").strip()
            if gamma_input.lower() == 'quit':
                break
                
            time_input = input("Time Step (t): ").strip()
            if time_input.lower() == 'quit':
                break
            
            # Convert to float
            sigma = float(sigma_input)
            gamma = float(gamma_input)
            time_step = float(time_input)
            
            # Create feature array
            features = np.array([[sigma, gamma, time_step]])
            
            # Make prediction
            pred_label = svm_model.predict(features)[0]
            
            # SVM might not have predict_proba, so handle gracefully
            if hasattr(svm_model, 'predict_proba'):
                pred_prob = svm_model.predict_proba(features)[0]
                confidence = max(pred_prob)
                healed_prob = pred_prob[1] if len(pred_prob) > 1 else pred_prob[0]
                not_healed_prob = pred_prob[0] if len(pred_prob) > 1 else 1 - pred_prob[0]
            else:
                confidence = 1.0  # SVM without probability estimates
                healed_prob = 1.0 if pred_label == 1 else 0.0
                not_healed_prob = 1.0 if pred_label == 0 else 0.0
            
            # Store prediction
            prediction = {
                'sigma': sigma,
                'gamma': gamma,
                'timeStep': time_step,
                'predicted_label': pred_label,
                'prediction_confidence': confidence,
                'healed_probability': healed_prob,
                'not_healed_probability': not_healed_prob
            }
            
            predictions.append(prediction)
            
            # Display result
            print(f"\n🎯 PREDICTION RESULT:")
            print(f"  Parameters: σ={sigma:.{decimal_places}f}, γ={gamma:.{decimal_places}f}, t={time_step:.{decimal_places}f}")
            print(f"  Prediction: {'🔴 HEALED' if pred_label == 1 else '🔵 NOT HEALED'}")
            print(f"  Confidence: {confidence:.{decimal_places}f}")
            if hasattr(svm_model, 'predict_proba'):
                print(f"  Healed Probability: {healed_prob:.{decimal_places}f}")
                print(f"  Not Healed Probability: {not_healed_prob:.{decimal_places}f}")
            else:
                print(f"  Note: SVM without probability estimates")
            
        except ValueError:
            print("❌ Invalid input! Please enter numeric values.")
        except KeyboardInterrupt:
            print("\n\nExiting prediction mode...")
            break
    
    # Save all predictions to file
    if predictions:
        with open("Predictions/svm_custom_predictions.txt", "w") as f:
            f.write("SVM CUSTOM PARAMETER PREDICTIONS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Custom Parameter Predictions:\n")
            f.write("-" * 30 + "\n")
            for i, pred in enumerate(predictions):
                f.write(f"Case {i+1}:\n")
                f.write(f"  Parameters: σ={pred['sigma']:.{decimal_places}f}, γ={pred['gamma']:.{decimal_places}f}, t={pred['timeStep']:.{decimal_places}f}\n")
                f.write(f"  Prediction: {'HEALED' if pred['predicted_label'] == 1 else 'NOT HEALED'}\n")
                f.write(f"  Confidence: {pred['prediction_confidence']:.{decimal_places}f}\n")
                f.write(f"  Healed Probability: {pred['healed_probability']:.{decimal_places}f}\n")
                f.write(f"  Not Healed Probability: {pred['not_healed_probability']:.{decimal_places}f}\n\n")
        
        print(f"\n✅ All predictions saved to: Predictions/svm_custom_predictions.txt")
        print(f"📊 Total predictions made: {len(predictions)}")

def make_gaussian_process_predictions(gp_model, scaler, decimal_places=3):
    """
    Make Gaussian Process predictions for custom input parameters (regression)
    """
    os.makedirs("Predictions", exist_ok=True)
    
    print("\n" + "="*60)
    print("GAUSSIAN PROCESS PREDICTIONS FOR CUSTOM PARAMETERS")
    print("="*60)
    print("Enter your own parameter values for prediction:")
    
    predictions = []
    while True:
        try:
            print(f"\nEnter parameters (or 'quit' to exit):")
            sigma_input = input("Sigma (σ): ").strip()
            if sigma_input.lower() == 'quit':
                break
                
            gamma_input = input("Gamma (γ): ").strip()
            if gamma_input.lower() == 'quit':
                break
                
            beta_input = input("Beta (β, in radians): ").strip()
            if beta_input.lower() == 'quit':
                break
            
            sigma = float(sigma_input)
            gamma = float(gamma_input)
            beta = float(beta_input)
            
            # Scale features
            features = np.array([[sigma, gamma, beta]])
            features_scaled = scaler.transform(features)
            
            # Make prediction (returns prediction and uncertainty)
            pred_healing_time, pred_std = gp_model.predict(features_scaled, return_std=True)
            pred_healing_time = pred_healing_time[0]
            pred_std = pred_std[0]
            
            prediction = {
                'sigma': sigma,
                'gamma': gamma,
                'beta': beta,
                'predicted_healing_time': pred_healing_time,
                'uncertainty_std': pred_std
            }
            predictions.append(prediction)
            
            print(f"\n🎯 PREDICTION RESULT:")
            print(f"  Parameters: σ={sigma:.{decimal_places}f}, γ={gamma:.{decimal_places}f}, β={beta:.{decimal_places}f}")
            print(f"  Predicted Healing Time: {pred_healing_time:.{decimal_places}f} seconds")
            print(f"  Uncertainty (std): ±{pred_std:.{decimal_places}f} seconds")
            print(f"  ({pred_healing_time/3600:.{decimal_places}f} hours)")
            
        except ValueError:
            print("❌ Invalid input! Please enter numeric values.")
        except KeyboardInterrupt:
            print("\n\nExiting prediction mode...")
            break
    
    if predictions:
        with open("Predictions/gaussian_process_predictions.txt", "w") as f:
            f.write("GAUSSIAN PROCESS CUSTOM PARAMETER PREDICTIONS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for i, pred in enumerate(predictions):
                f.write(f"Case {i+1}:\n")
                f.write(f"  Parameters: σ={pred['sigma']:.{decimal_places}f}, γ={pred['gamma']:.{decimal_places}f}, β={pred['beta']:.{decimal_places}f}\n")
                f.write(f"  Predicted Healing Time: {pred['predicted_healing_time']:.{decimal_places}f} seconds ({pred['predicted_healing_time']/3600:.{decimal_places}f} hours)\n")
                f.write(f"  Uncertainty (std): ±{pred['uncertainty_std']:.{decimal_places}f} seconds\n\n")
        print(f"\n✅ All predictions saved to: Predictions/gaussian_process_predictions.txt")
        print(f"📊 Total predictions made: {len(predictions)}")

def make_ridge_regression_predictions(ridge_model, scaler, decimal_places=3):
    """Make Ridge Regression predictions for custom input parameters (regression)"""
    os.makedirs("Predictions", exist_ok=True)
    
    print("\n" + "="*60)
    print("RIDGE REGRESSION PREDICTIONS FOR CUSTOM PARAMETERS")
    print("="*60)
    print("Enter your own parameter values for prediction:")
    
    predictions = []
    while True:
        try:
            print(f"\nEnter parameters (or 'quit' to exit):")
            sigma_input = input("Sigma (σ): ").strip()
            if sigma_input.lower() == 'quit':
                break
            gamma_input = input("Gamma (γ): ").strip()
            if gamma_input.lower() == 'quit':
                break
            beta_input = input("Beta (β, in radians): ").strip()
            if beta_input.lower() == 'quit':
                break
            
            sigma = float(sigma_input)
            gamma = float(gamma_input)
            beta = float(beta_input)
            
            features = np.array([[sigma, gamma, beta]])
            features_scaled = scaler.transform(features)
            
            pred_healing_time = ridge_model.predict(features_scaled)[0]
            
            prediction = {
                'sigma': sigma,
                'gamma': gamma,
                'beta': beta,
                'predicted_healing_time': pred_healing_time
            }
            predictions.append(prediction)
            
            print(f"\n🎯 PREDICTION RESULT:")
            print(f"  Parameters: σ={sigma:.{decimal_places}f}, γ={gamma:.{decimal_places}f}, β={beta:.{decimal_places}f}")
            print(f"  Predicted Healing Time: {pred_healing_time:.{decimal_places}f} seconds ({pred_healing_time/3600:.{decimal_places}f} hours)")
            
        except ValueError:
            print("❌ Invalid input! Please enter numeric values.")
        except KeyboardInterrupt:
            print("\n\nExiting prediction mode...")
            break
    
    if predictions:
        with open("Predictions/ridge_regression_predictions.txt", "w") as f:
            f.write("RIDGE REGRESSION CUSTOM PARAMETER PREDICTIONS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for i, pred in enumerate(predictions):
                f.write(f"Case {i+1}:\n")
                f.write(f"  Parameters: σ={pred['sigma']:.{decimal_places}f}, γ={pred['gamma']:.{decimal_places}f}, β={pred['beta']:.{decimal_places}f}\n")
                f.write(f"  Predicted Healing Time: {pred['predicted_healing_time']:.{decimal_places}f} seconds ({pred['predicted_healing_time']/3600:.{decimal_places}f} hours)\n\n")
        print(f"\n✅ All predictions saved to: Predictions/ridge_regression_predictions.txt")
        print(f"📊 Total predictions made: {len(predictions)}")

def make_random_forest_predictions(rf_model, scaler, decimal_places=3):
    """Make Random Forest predictions for custom input parameters (regression)"""
    os.makedirs("Predictions", exist_ok=True)
    
    print("\n" + "="*60)
    print("RANDOM FOREST PREDICTIONS FOR CUSTOM PARAMETERS")
    print("="*60)
    print("Enter your own parameter values for prediction:")
    
    predictions = []
    while True:
        try:
            print(f"\nEnter parameters (or 'quit' to exit):")
            sigma_input = input("Sigma (σ): ").strip()
            if sigma_input.lower() == 'quit':
                break
            gamma_input = input("Gamma (γ): ").strip()
            if gamma_input.lower() == 'quit':
                break
            beta_input = input("Beta (β, in radians): ").strip()
            if beta_input.lower() == 'quit':
                break
            
            sigma = float(sigma_input)
            gamma = float(gamma_input)
            beta = float(beta_input)
            
            features = np.array([[sigma, gamma, beta]])
            features_scaled = scaler.transform(features)
            
            pred_healing_time = rf_model.predict(features_scaled)[0]
            
            prediction = {
                'sigma': sigma,
                'gamma': gamma,
                'beta': beta,
                'predicted_healing_time': pred_healing_time
            }
            predictions.append(prediction)
            
            print(f"\n🎯 PREDICTION RESULT:")
            print(f"  Parameters: σ={sigma:.{decimal_places}f}, γ={gamma:.{decimal_places}f}, β={beta:.{decimal_places}f}")
            print(f"  Predicted Healing Time: {pred_healing_time:.{decimal_places}f} seconds ({pred_healing_time/3600:.{decimal_places}f} hours)")
            
        except ValueError:
            print("❌ Invalid input! Please enter numeric values.")
        except KeyboardInterrupt:
            print("\n\nExiting prediction mode...")
            break
    
    if predictions:
        with open("Predictions/random_forest_predictions.txt", "w") as f:
            f.write("RANDOM FOREST CUSTOM PARAMETER PREDICTIONS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for i, pred in enumerate(predictions):
                f.write(f"Case {i+1}:\n")
                f.write(f"  Parameters: σ={pred['sigma']:.{decimal_places}f}, γ={pred['gamma']:.{decimal_places}f}, β={pred['beta']:.{decimal_places}f}\n")
                f.write(f"  Predicted Healing Time: {pred['predicted_healing_time']:.{decimal_places}f} seconds ({pred['predicted_healing_time']/3600:.{decimal_places}f} hours)\n\n")
        print(f"\n✅ All predictions saved to: Predictions/random_forest_predictions.txt")
        print(f"📊 Total predictions made: {len(predictions)}")

def make_xgboost_predictions(xgb_model, scaler, decimal_places=3):
    """Make XGBoost predictions for custom input parameters (regression)"""
    os.makedirs("Predictions", exist_ok=True)
    
    print("\n" + "="*60)
    print("XGBOOST PREDICTIONS FOR CUSTOM PARAMETERS")
    print("="*60)
    print("Enter your own parameter values for prediction:")
    
    predictions = []
    while True:
        try:
            print(f"\nEnter parameters (or 'quit' to exit):")
            sigma_input = input("Sigma (σ): ").strip()
            if sigma_input.lower() == 'quit':
                break
            gamma_input = input("Gamma (γ): ").strip()
            if gamma_input.lower() == 'quit':
                break
            beta_input = input("Beta (β, in radians): ").strip()
            if beta_input.lower() == 'quit':
                break
            
            sigma = float(sigma_input)
            gamma = float(gamma_input)
            beta = float(beta_input)
            
            features = np.array([[sigma, gamma, beta]])
            features_scaled = scaler.transform(features)
            
            pred_healing_time = xgb_model.predict(features_scaled)[0]
            
            prediction = {
                'sigma': sigma,
                'gamma': gamma,
                'beta': beta,
                'predicted_healing_time': pred_healing_time
            }
            predictions.append(prediction)
            
            print(f"\n🎯 PREDICTION RESULT:")
            print(f"  Parameters: σ={sigma:.{decimal_places}f}, γ={gamma:.{decimal_places}f}, β={beta:.{decimal_places}f}")
            print(f"  Predicted Healing Time: {pred_healing_time:.{decimal_places}f} seconds ({pred_healing_time/3600:.{decimal_places}f} hours)")
            
        except ValueError:
            print("❌ Invalid input! Please enter numeric values.")
        except KeyboardInterrupt:
            print("\n\nExiting prediction mode...")
            break
    
    if predictions:
        with open("Predictions/xgboost_predictions.txt", "w") as f:
            f.write("XGBOOST CUSTOM PARAMETER PREDICTIONS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for i, pred in enumerate(predictions):
                f.write(f"Case {i+1}:\n")
                f.write(f"  Parameters: σ={pred['sigma']:.{decimal_places}f}, γ={pred['gamma']:.{decimal_places}f}, β={pred['beta']:.{decimal_places}f}\n")
                f.write(f"  Predicted Healing Time: {pred['predicted_healing_time']:.{decimal_places}f} seconds ({pred['predicted_healing_time']/3600:.{decimal_places}f} hours)\n\n")
        print(f"\n✅ All predictions saved to: Predictions/xgboost_predictions.txt")
        print(f"📊 Total predictions made: {len(predictions)}")

def make_neural_network_regression_predictions(nn_model, scaler, y_scaler=None, decimal_places=3):
    """Make Neural Network regression predictions for custom input parameters"""
    os.makedirs("Predictions", exist_ok=True)
    
    print("\n" + "="*60)
    print("NEURAL NETWORK REGRESSION PREDICTIONS FOR CUSTOM PARAMETERS")
    print("="*60)
    print("Enter your own parameter values for prediction:")
    
    predictions = []
    while True:
        try:
            print(f"\nEnter parameters (or 'quit' to exit):")
            sigma_input = input("Sigma (σ): ").strip()
            if sigma_input.lower() == 'quit':
                break
            gamma_input = input("Gamma (γ): ").strip()
            if gamma_input.lower() == 'quit':
                break
            beta_input = input("Beta (β, in radians): ").strip()
            if beta_input.lower() == 'quit':
                break
            
            sigma = float(sigma_input)
            gamma = float(gamma_input)
            beta = float(beta_input)
            
            features = np.array([[sigma, gamma, beta]])
            features_scaled = scaler.transform(features)
            
            # Predict on scaled data
            pred_scaled = nn_model.predict(features_scaled, verbose=0)
            
            # Inverse transform if y_scaler is provided
            if y_scaler is not None:
                pred_healing_time = y_scaler.inverse_transform(pred_scaled)[0][0]
            else:
                pred_healing_time = pred_scaled[0][0]
            
            prediction = {
                'sigma': sigma,
                'gamma': gamma,
                'beta': beta,
                'predicted_healing_time': pred_healing_time
            }
            predictions.append(prediction)
            
            print(f"\n🎯 PREDICTION RESULT:")
            print(f"  Parameters: σ={sigma:.{decimal_places}f}, γ={gamma:.{decimal_places}f}, β={beta:.{decimal_places}f}")
            print(f"  Predicted Healing Time: {pred_healing_time:.{decimal_places}f} seconds ({pred_healing_time/3600:.{decimal_places}f} hours)")
            
        except ValueError:
            print("❌ Invalid input! Please enter numeric values.")
        except KeyboardInterrupt:
            print("\n\nExiting prediction mode...")
            break
    
    if predictions:
        with open("Predictions/neural_network_regression_predictions.txt", "w") as f:
            f.write("NEURAL NETWORK REGRESSION CUSTOM PARAMETER PREDICTIONS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for i, pred in enumerate(predictions):
                f.write(f"Case {i+1}:\n")
                f.write(f"  Parameters: σ={pred['sigma']:.{decimal_places}f}, γ={pred['gamma']:.{decimal_places}f}, β={pred['beta']:.{decimal_places}f}\n")
                f.write(f"  Predicted Healing Time: {pred['predicted_healing_time']:.{decimal_places}f} seconds ({pred['predicted_healing_time']/3600:.{decimal_places}f} hours)\n\n")
        print(f"\n✅ All predictions saved to: Predictions/neural_network_regression_predictions.txt")
        print(f"📊 Total predictions made: {len(predictions)}")

def load_saved_models():
    """
    Load all saved models from disk
    """
    models = {}
    import joblib
    
    # Try to load Gaussian Process model
    try:
        if os.path.exists("Gaussian Process/gaussian_process_model.pkl"):
            models['gaussian_process'] = joblib.load("Gaussian Process/gaussian_process_model.pkl")
            print("✅ Loaded saved Gaussian Process model")
    except:
        print("❌ Could not load Gaussian Process model")
    
    # Try to load Ridge Regression model
    try:
        if os.path.exists("Ridge Regression/ridge_regression_model.pkl"):
            models['ridge_regression'] = joblib.load("Ridge Regression/ridge_regression_model.pkl")
            print("✅ Loaded saved Ridge Regression model")
    except:
        print("❌ Could not load Ridge Regression model")
    
    # Try to load Random Forest model
    try:
        if os.path.exists("Random Forest/random_forest_model.pkl"):
            models['random_forest'] = joblib.load("Random Forest/random_forest_model.pkl")
            print("✅ Loaded saved Random Forest model")
    except:
        print("❌ Could not load Random Forest model")
    
    # Try to load XGBoost model
    try:
        if os.path.exists("XGBoost/xgboost_model.pkl"):
            models['xgboost'] = joblib.load("XGBoost/xgboost_model.pkl")
            print("✅ Loaded saved XGBoost model")
    except:
        print("❌ Could not load XGBoost model")
    
    # Try to load Neural Network model
    try:
        if not HAS_TF:
            print("⚠️  TensorFlow not available. Cannot load Neural Network model.")
        elif os.path.exists("Neural Network/neural_network_model.h5"):
            # Try loading with compile=False first (in case of custom objects)
            try:
                models['neural_network'] = tf.keras.models.load_model(
                    "Neural Network/neural_network_model.h5", 
                    compile=False
                )
                # Recompile with the same settings used during training
                models['neural_network'].compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss="mse",
                    metrics=["mae", "mse"]
                )
                print("✅ Loaded saved Neural Network model")
            except Exception as e1:
                # If compile=False fails, try with default loading
                try:
                    models['neural_network'] = tf.keras.models.load_model("Neural Network/neural_network_model.h5")
                    print("✅ Loaded saved Neural Network model")
                except Exception as e2:
                    print(f"❌ Could not load Neural Network model: {e2}")
                    print(f"   First attempt error: {e1}")
        else:
            print("⚠️  Neural Network model file not found")
    except Exception as e:
        print(f"❌ Could not load Neural Network model: {e}")
        import traceback
        traceback.print_exc()
    
    return models

def perform_shap_analysis(model, model_name, X_train, X_test, feature_names, scaler, y_scaler=None, sample_size=100):
    """
    Perform SHAP analysis on a trained model
    
    Parameters:
    -----------
    model : trained model
        The trained model to analyze
    model_name : str
        Name of the model
    X_train : np.array
        Training features
    X_test : np.array
        Test features
    feature_names : list
        List of feature names ['Sigma', 'Gamma', 'Beta']
    scaler : StandardScaler
        Feature scaler used during training
    y_scaler : StandardScaler or None
        Target scaler for neural network (if applicable)
    sample_size : int
        Number of samples to use for SHAP computation (for speed)
    """
    if not HAS_SHAP:
        print("❌ SHAP library not available. Please install with: pip install shap")
        return
    
    print(f"\n{'='*60}")
    print(f"PERFORMING SHAP ANALYSIS FOR {model_name.upper().replace('_', ' ')}")
    print(f"{'='*60}")
    
    # Create SHAP directory
    shap_dir = f"SHAP Analysis/{model_name.replace('_', ' ').title()}"
    os.makedirs(shap_dir, exist_ok=True)
    
    # Sample data for SHAP computation (for speed)
    # Use stratified sampling to ensure good coverage
    np.random.seed(42)
    if len(X_test) > sample_size:
        sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_test_sample = X_test[sample_indices]
    else:
        X_test_sample = X_test
        sample_indices = np.arange(len(X_test))
    
    print(f"Using {len(X_test_sample)} samples for SHAP computation...")
    
    # Create appropriate SHAP explainer based on model type
    try:
        if model_name == 'neural_network':
            # For neural networks, use KernelExplainer or DeepExplainer
            # Using KernelExplainer for compatibility
            def model_predict(X):
                # X is already scaled, but we need to handle it properly
                if y_scaler is not None:
                    pred_scaled = model.predict(X, verbose=0)
                    return y_scaler.inverse_transform(pred_scaled).flatten()
                else:
                    return model.predict(X, verbose=0).flatten()
            
            # Use a subset of training data as background
            background_size = min(50, len(X_train))
            background_indices = np.random.choice(len(X_train), background_size, replace=False)
            background = X_train[background_indices]
            
            explainer = shap.KernelExplainer(model_predict, background)
            shap_values = explainer.shap_values(X_test_sample)
            
        elif model_name in ['random_forest', 'xgboost']:
            # Tree-based models can use TreeExplainer (fastest)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_sample)
            
        elif model_name == 'gaussian_process':
            # For Gaussian Process, use KernelExplainer
            def model_predict(X):
                return model.predict(X, return_std=False)
            
            background_size = min(50, len(X_train))
            background_indices = np.random.choice(len(X_train), background_size, replace=False)
            background = X_train[background_indices]
            
            explainer = shap.KernelExplainer(model_predict, background)
            shap_values = explainer.shap_values(X_test_sample)
            
        elif model_name == 'ridge_regression':
            # For linear models, use LinearExplainer
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_test_sample)
            
        else:
            # Fallback to KernelExplainer
            def model_predict(X):
                return model.predict(X)
            
            background_size = min(50, len(X_train))
            background_indices = np.random.choice(len(X_train), background_size, replace=False)
            background = X_train[background_indices]
            
            explainer = shap.KernelExplainer(model_predict, background)
            shap_values = explainer.shap_values(X_test_sample)
        
        # Ensure shap_values is 2D
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(-1, 1)
        
        print("✅ SHAP values computed successfully")
        
        # Create mapping from feature names to LaTeX symbols
        latex_symbols = {
            'Beta': r'$\beta$',
            'beta': r'$\beta$',
            'Sigma': r'$\sigma$',
            'sigma': r'$\sigma$',
            'Gamma': r'$\gamma$',
            'gamma': r'$\gamma$'
        }
        
        # 1. Create horizontal bar chart of Mean Absolute SHAP Values
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create mapping from original feature names to indices
        feature_to_index = {name.lower(): i for i, name in enumerate(feature_names)}
        
        # Order: beta, sigma, gamma (from top to bottom)
        # Note: barh displays bottom to top, so we reverse the list to get beta at top
        desired_order = ['gamma', 'sigma', 'beta']  # Reversed so beta appears at top
        ordered_indices = [feature_to_index[name] for name in desired_order if name in feature_to_index]
        ordered_feature_names_latex = [latex_symbols[feature_names[i]] for i in ordered_indices]
        ordered_mean_abs_shap = mean_abs_shap[ordered_indices]
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(ordered_feature_names_latex, ordered_mean_abs_shap, color='#1f77b4', alpha=0.8, zorder=3)
        
        ax.set_xlabel('SHAP value (seconds)', fontsize=23)
        ax.set_ylabel('Variables', fontsize=23)
        ax.set_title(f'Mean Absolute SHAP Values - {model_name.replace("_", " ").title()}', fontsize=24)
        ax.grid(True, alpha=0.3, zorder=1, axis='x')
        
        # Use scientific notation for x-axis
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        ax.xaxis.get_offset_text().set_fontsize(14)
        
        # Make y-axis labels (LaTeX symbols) 30% bigger
        # Default y-axis label size is typically 18, so 30% bigger = 23.4, round to 23
        ax.tick_params(axis='y', labelsize=23)
        
        plt.tight_layout()
        bar_chart_filename = f"{shap_dir}/mean_absolute_shap_values.png"
        plt.savefig(bar_chart_filename, dpi=300, bbox_inches='tight')
        print(f"✅ Bar chart saved to: {bar_chart_filename}")
        plt.close()
        
        # Save SHAP values summary to text file
        summary_filename = f"{shap_dir}/shap_values_summary.txt"
        with open(summary_filename, 'w') as f:
            f.write("MEAN ABSOLUTE SHAP VALUES SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Model: {model_name.replace('_', ' ').title()}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Values are in seconds (same units as healing time).\n")
            f.write("Higher values indicate greater average impact on healing time prediction.\n\n")
            f.write("-" * 50 + "\n")
            f.write("Feature Rankings (sorted by impact, highest first):\n")
            f.write("-" * 50 + "\n")
            for i, (latex_name, shap_val) in enumerate(zip(ordered_feature_names_latex, ordered_mean_abs_shap), 1):
                # Write original feature name for text file
                original_name = feature_names[ordered_indices[i-1]]
                f.write(f"{i}. {original_name}: {shap_val:.6e} seconds\n")
            f.write("\n" + "=" * 50 + "\n")
            f.write("Raw Values (unsorted):\n")
            f.write("-" * 50 + "\n")
            for feature_name, shap_val in zip(feature_names, mean_abs_shap):
                f.write(f"{feature_name}: {shap_val:.6e} seconds\n")
        print(f"✅ SHAP values summary saved to: {summary_filename}")
        
        # 2. Create SHAP summary plot (feature impact distribution)
        # Get original feature values (inverse transform from scaled) for coloring
        X_test_sample_original = scaler.inverse_transform(X_test_sample)
        
        # Convert feature names to LaTeX symbols for summary plot
        feature_names_latex = [latex_symbols.get(name, name) for name in feature_names]
        
        # Create summary plot using SHAP's built-in function
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_sample_original, feature_names=feature_names_latex, 
                         show=False, plot_type="dot")
        
        # Customize the plot
        plt.title(f'SHAP Summary Plot - Feature Impact Distribution\n{model_name.replace("_", " ").title()}', 
                 fontsize=20, pad=20)
        plt.xlabel('SHAP value (seconds)', fontsize=18)
        plt.ylabel('Variables', fontsize=18)
        
        # Adjust tick label sizes
        ax = plt.gca()
        ax.tick_params(axis='x', labelsize=14)
        # Make y-axis labels (LaTeX symbols) 15% bigger
        # Default is 14, so 15% bigger = 16.1, round to 16
        ax.tick_params(axis='y', labelsize=16)
        
        # Fix x-axis tick frequency - use MaxNLocator to control number of ticks
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))  # Limit to 6 major ticks for cleaner look
        
        plt.tight_layout()
        summary_plot_filename = f"{shap_dir}/shap_summary_plot.png"
        plt.savefig(summary_plot_filename, dpi=300, bbox_inches='tight')
        print(f"✅ SHAP summary plot saved to: {summary_plot_filename}")
        plt.close()
        
        # Save SHAP values to CSV
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        shap_df.to_csv(f"{shap_dir}/shap_values.csv", index=False)
        print(f"✅ SHAP values saved to: {shap_dir}/shap_values.csv")
        
        # Print summary statistics
        print(f"\n{'='*60}")
        print("SHAP ANALYSIS SUMMARY")
        print(f"{'='*60}")
        for feature_name, mean_abs in zip(feature_names, mean_abs_shap):
            print(f"{feature_name}: Mean Absolute SHAP = {mean_abs:.6f}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"❌ Error during SHAP analysis: {e}")
        import traceback
        traceback.print_exc()

def make_neural_network_predictions(nn_model, X_test, y_test, decimal_places=3):
    """
    Make Neural Network predictions for custom input parameters
    """
    # Create predictions folder if it doesn't exist
    os.makedirs("Predictions", exist_ok=True)
    
    print("\n" + "="*60)
    print("NEURAL NETWORK PREDICTIONS FOR CUSTOM PARAMETERS")
    print("="*60)
    print("Enter your own parameter values for prediction:")
    
    predictions = []
    while True:
        try:
            # Get user input for parameters
            print(f"\nEnter parameters (or 'quit' to exit):")
            sigma_input = input("Sigma (σ): ").strip()
            if sigma_input.lower() == 'quit':
                break
                
            gamma_input = input("Gamma (γ): ").strip()
            if gamma_input.lower() == 'quit':
                break
                
            time_input = input("Time Step (t): ").strip()
            if time_input.lower() == 'quit':
                break
            
            # Convert to float
            sigma = float(sigma_input)
            gamma = float(gamma_input)
            time_step = float(time_input)
            
            # Create feature array
            features = np.array([[sigma, gamma, time_step]])
            
            # Make prediction
            pred_prob = nn_model.predict(features)[0][0]
            pred_label = 1 if pred_prob > 0.5 else 0
            
            # Store prediction
            prediction = {
                'sigma': sigma,
                'gamma': gamma,
                'timeStep': time_step,
                'predicted_label': pred_label,
                'prediction_confidence': max(pred_prob, 1-pred_prob),
                'healed_probability': pred_prob,
                'not_healed_probability': 1 - pred_prob
            }
            
            predictions.append(prediction)
            
            # Display result
            print(f"\n🎯 PREDICTION RESULT:")
            print(f"  Parameters: σ={sigma:.{decimal_places}f}, γ={gamma:.{decimal_places}f}, t={time_step:.{decimal_places}f}")
            print(f"  Prediction: {'🔴 HEALED' if pred_label == 1 else '🔵 NOT HEALED'}")
            print(f"  Confidence: {max(pred_prob, 1-pred_prob):.{decimal_places}f}")
            print(f"  Healed Probability: {pred_prob:.{decimal_places}f}")
            print(f"  Not Healed Probability: {1 - pred_prob:.{decimal_places}f}")
            
        except ValueError:
            print("❌ Invalid input! Please enter numeric values.")
        except KeyboardInterrupt:
            print("\n\nExiting prediction mode...")
            break
    
    # Save all predictions to file
    if predictions:
        with open("Predictions/neural_network_custom_predictions.txt", "w") as f:
            f.write("NEURAL NETWORK CUSTOM PARAMETER PREDICTIONS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Custom Parameter Predictions:\n")
            f.write("-" * 30 + "\n")
            for i, pred in enumerate(predictions):
                f.write(f"Case {i+1}:\n")
                f.write(f"  Parameters: σ={pred['sigma']:.{decimal_places}f}, γ={pred['gamma']:.{decimal_places}f}, t={pred['timeStep']:.{decimal_places}f}\n")
                f.write(f"  Prediction: {'HEALED' if pred['predicted_label'] == 1 else 'NOT HEALED'}\n")
                f.write(f"  Confidence: {pred['prediction_confidence']:.{decimal_places}f}\n")
                f.write(f"  Healed Probability: {pred['healed_probability']:.{decimal_places}f}\n")
                f.write(f"  Not Healed Probability: {pred['not_healed_probability']:.{decimal_places}f}\n\n")
        
        print(f"\n✅ All predictions saved to: Predictions/neural_network_custom_predictions.txt")
        print(f"📊 Total predictions made: {len(predictions)}")

def create_data_information_report(df, X_train, y_train, X_valid, y_valid, X_test, y_test, 
                                   feature_cols, target_col, decimal_places=3):
    """
    Create a comprehensive data information report for regression
    """
    # Create Data Information folder
    os.makedirs("Data Information", exist_ok=True)
    
    total_samples = len(df)
    
    # Target variable statistics
    target_stats = {
        'mean': df[target_col].mean(),
        'std': df[target_col].std(),
        'min': df[target_col].min(),
        'max': df[target_col].max(),
        'median': df[target_col].median()
    }
    
    # Training, validation, and test set target statistics
    train_target_stats = {
        'mean': np.mean(y_train),
        'std': np.std(y_train),
        'min': np.min(y_train),
        'max': np.max(y_train),
        'median': np.median(y_train)
    }
    
    valid_target_stats = {
        'mean': np.mean(y_valid),
        'std': np.std(y_valid),
        'min': np.min(y_valid),
        'max': np.max(y_valid),
        'median': np.median(y_valid)
    }
    
    test_target_stats = {
        'mean': np.mean(y_test),
        'std': np.std(y_test),
        'min': np.min(y_test),
        'max': np.max(y_test),
        'median': np.median(y_test)
    }
    
    # Feature statistics
    feature_stats = {}
    for col in feature_cols:
        feature_stats[col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median()
        }
    
    # Data split percentages
    train_percent = 0.6
    valid_percent = 0.2
    test_percent = 0.2
    
    # Save comprehensive report
    with open("Data Information/data_information_report.txt", "w") as f:
        f.write("COMPREHENSIVE DATA INFORMATION REPORT (REGRESSION)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("1. DATASET OVERVIEW\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total samples: {total_samples:,}\n")
        f.write(f"Features: {len(feature_cols)} ({', '.join(feature_cols)})\n")
        f.write(f"Target variable: {target_col} (healing time in seconds)\n")
        f.write(f"Data source: sgb_healing_time_data.csv\n")
        f.write(f"Task type: Regression\n\n")
        
        f.write("2. TARGET VARIABLE STATISTICS (FULL DATASET)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean: {target_stats['mean']:.{decimal_places}f} seconds\n")
        f.write(f"Std: {target_stats['std']:.{decimal_places}f} seconds\n")
        f.write(f"Min: {target_stats['min']:.{decimal_places}f} seconds\n")
        f.write(f"Max: {target_stats['max']:.{decimal_places}f} seconds\n")
        f.write(f"Median: {target_stats['median']:.{decimal_places}f} seconds\n\n")
        
        f.write("3. DATA SPLIT INFORMATION\n")
        f.write("-" * 30 + "\n")
        f.write(f"Training set: {train_percent*100:.0f}% ({len(X_train):,} samples)\n")
        f.write(f"Validation set: {valid_percent*100:.0f}% ({len(X_valid):,} samples)\n")
        f.write(f"Test set: {test_percent*100:.0f}% ({len(X_test):,} samples)\n\n")
        
        f.write("4. TRAINING SET TARGET STATISTICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mean: {train_target_stats['mean']:.{decimal_places}f} seconds\n")
        f.write(f"Std: {train_target_stats['std']:.{decimal_places}f} seconds\n")
        f.write(f"Min: {train_target_stats['min']:.{decimal_places}f} seconds\n")
        f.write(f"Max: {train_target_stats['max']:.{decimal_places}f} seconds\n")
        f.write(f"Median: {train_target_stats['median']:.{decimal_places}f} seconds\n\n")
        
        f.write("5. VALIDATION SET TARGET STATISTICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mean: {valid_target_stats['mean']:.{decimal_places}f} seconds\n")
        f.write(f"Std: {valid_target_stats['std']:.{decimal_places}f} seconds\n")
        f.write(f"Min: {valid_target_stats['min']:.{decimal_places}f} seconds\n")
        f.write(f"Max: {valid_target_stats['max']:.{decimal_places}f} seconds\n")
        f.write(f"Median: {valid_target_stats['median']:.{decimal_places}f} seconds\n\n")
        
        f.write("6. TEST SET TARGET STATISTICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mean: {test_target_stats['mean']:.{decimal_places}f} seconds\n")
        f.write(f"Std: {test_target_stats['std']:.{decimal_places}f} seconds\n")
        f.write(f"Min: {test_target_stats['min']:.{decimal_places}f} seconds\n")
        f.write(f"Max: {test_target_stats['max']:.{decimal_places}f} seconds\n")
        f.write(f"Median: {test_target_stats['median']:.{decimal_places}f} seconds\n\n")
        
        f.write("7. FEATURE STATISTICS\n")
        f.write("-" * 30 + "\n")
        for feature, stats in feature_stats.items():
            f.write(f"{feature}:\n")
            f.write(f"  - Mean: {stats['mean']:.{decimal_places}f}\n")
            f.write(f"  - Std: {stats['std']:.{decimal_places}f}\n")
            f.write(f"  - Min: {stats['min']:.{decimal_places}f}\n")
            f.write(f"  - Max: {stats['max']:.{decimal_places}f}\n")
            f.write(f"  - Median: {stats['median']:.{decimal_places}f}\n\n")
        
        f.write("8. PREPROCESSING DETAILS\n")
        f.write("-" * 30 + "\n")
        f.write("Standardization: StandardScaler applied to all features\n")
        f.write("Oversampling: Not applicable (regression task)\n")
        f.write("Validation and test sets: No preprocessing applied (preserves real distribution)\n\n")
        
        f.write("9. MODEL PARAMETERS\n")
        f.write("-" * 30 + "\n")
        f.write("Gaussian Process Regressor: RBF kernel, alpha=1e-6\n")
        f.write("Ridge Regression: alpha=1.0 (L2 regularization)\n")
        f.write("Random Forest Regressor: n_estimators=100\n")
        f.write("XGBoost Regressor: Gradient Boosted Trees, n_estimators=100, max_depth=6\n")
        f.write("Neural Network: 3-32-32-1 architecture, ReLU activation, dropout=0.2, lr=0.001, batch_size=32, epochs=100\n\n")
        
        f.write("10. TECHNICAL DETAILS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Random seed: 42 (for reproducibility)\n")
        f.write(f"Data type: Float64\n")
        f.write(f"Missing values: {df.isnull().sum().sum()}\n")
        f.write(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.{decimal_places}f} MB\n")
        
    print("Data information report saved to: Data Information/data_information_report.txt")
    
    # Also save as JSON for programmatic access
    json_filename = "Data Information/data_information_data.json"
    data_summary = {
        'dataset_overview': {
            'total_samples': total_samples,
            'features': len(feature_cols),
            'feature_names': feature_cols,
            'target_variable': target_col,
            'task_type': 'regression'
        },
        'target_statistics': {
            'full_dataset': target_stats,
            'training_set': train_target_stats,
            'validation_set': valid_target_stats,
            'test_set': test_target_stats
        },
        'data_splits': {
            'training': {'samples': len(X_train), 'percentage': train_percent*100},
            'validation': {'samples': len(X_valid), 'percentage': valid_percent*100},
            'test': {'samples': len(X_test), 'percentage': test_percent*100}
        },
        'feature_statistics': feature_stats
    }
    
    with open(json_filename, 'w') as f:
        json.dump(data_summary, f, indent=2)
    
    print(f"Data information (JSON) saved to: {json_filename}")

def debug_classification_reports():
    """
    Debug function to show the raw content of classification report files
    """
    print("\n" + "="*60)
    print("DEBUGGING CLASSIFICATION REPORT FILES")
    print("="*60)
    
    model_folders = {
        'KNN': 'KNN/knn_classification_report.txt',
        'Naive Bayes': 'Naive Bayes/naive_bayes_classification_report.txt',
        'Logistic Regression': 'Logistic Regression/logistic_regression_classification_report.txt',
        'SVM': 'SVM/svm_classification_report.txt',
        'Neural Network': 'Neural Network/neural_network_classification_report.txt'
    }
    
    for model_name, file_path in model_folders.items():
        if os.path.exists(file_path):
            print(f"\n--- {model_name} ---")
            print(f"File: {file_path}")
            print("-" * 40)
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    print(content[:500] + "..." if len(content) > 500 else content)
            except Exception as e:
                print(f"Error reading file: {e}")
        else:
            print(f"\n--- {model_name} ---")
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    # ========================================================================
    # TOGGLE SWITCHES - Set to True/False to enable/disable functionality
    # ========================================================================
    
    # Data Visualization
    PLOT_HISTOGRAMS = True  # Plot histograms for features and target
    
    # Model Training
    TRAIN_GAUSSIAN_PROCESS = True
    TRAIN_RIDGE_REGRESSION = True
    TRAIN_RANDOM_FOREST = True
    TRAIN_XGBOOST = True
    TRAIN_NEURAL_NETWORK = True
    
    # Reports and Analysis
    CREATE_DATA_INFORMATION_REPORT = True  # Create comprehensive data report
    CREATE_MODEL_COMPARISON_CHART = True  # Create R² comparison chart
    
    # Interactive Sections
    ENABLE_PREDICTIONS = True  # Enable interactive predictions section
    ENABLE_SHAP_ANALYSIS = True  # Enable SHAP analysis section
    
    # Configuration variable for decimal places in regression reports
    decimal_places = 6  # Number of decimal places to show in performance metrics
    
    # ========================================================================
    # EXECUTION - Code below runs based on toggle switches above
    # ========================================================================
    
    # Plot histograms
    if PLOT_HISTOGRAMS:
        print("\n" + "="*60)
        print("PLOTTING HISTOGRAMS")
        print("="*60)
        plot_histograms(df, feature_cols, target_col)
    
    # Train regression models
    if TRAIN_GAUSSIAN_PROCESS:
        print("\n" + "="*60)
        print("TRAINING GAUSSIAN PROCESS REGRESSOR")
        print("="*60)
        gaussian_process_model_trained = gaussian_process_model(X_train, y_train, X_test, y_test, decimal_places)
    
    if TRAIN_RIDGE_REGRESSION:
        print("\n" + "="*60)
        print("TRAINING RIDGE REGRESSION")
        print("="*60)
        ridge_regression_model_trained = ridge_regression_model(X_train, y_train, X_test, y_test, alpha=1.0, decimal_places=decimal_places)
    
    if TRAIN_RANDOM_FOREST:
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST REGRESSOR")
        print("="*60)
        random_forest_model_trained = random_forest_model(X_train, y_train, X_test, y_test, decimal_places)
    
    if TRAIN_XGBOOST:
        print("\n" + "="*60)
        print("TRAINING XGBOOST REGRESSOR")
        print("="*60)
        xgboost_model_trained = xgboost_model(X_train, y_train, X_test, y_test, 
                                             n_estimators=100, max_depth=6, learning_rate=0.1, 
                                             decimal_places=decimal_places)
    
    if TRAIN_NEURAL_NETWORK:
        print("\n" + "="*60)
        print("TRAINING NEURAL NETWORK REGRESSOR")
        print("="*60)
        # Architecture: 3-32-32-1 (3 inputs -> 32 -> 32 -> 1 output)
        # Parameters: dropout=0.2, learning_rate=0.001, batch_size=32, epochs=100
        neural_network_model_trained = neural_network_model(X_train, y_train, X_test, y_test, 
                                                           dropout_prob=0.2, lr=0.001, batch_size=32, epochs=100, 
                                                           decimal_places=decimal_places)
    
    # Create comprehensive data information report
    if CREATE_DATA_INFORMATION_REPORT:
        print("\n" + "="*60)
        print("CREATING DATA INFORMATION REPORT")
        print("="*60)
        create_data_information_report(df, X_train, y_train, X_valid, y_valid, X_test, y_test, 
                                       feature_cols, target_col, decimal_places)
    
    # Create comparison chart by reading from saved files
    if CREATE_MODEL_COMPARISON_CHART:
        print("\n" + "="*60)
        print("CREATING MODEL COMPARISON CHART (R² Scores)")
        print("="*60)
        # Create chart with auto-calculated y-axis range
        create_model_comparison_chart(min_score=None, max_score=None, buffer=0.05, decimal_places=decimal_places)

    # Make predictions with saved models
    if ENABLE_PREDICTIONS:
        print("\n" + "="*60)
        print("LOADING SAVED MODELS FOR PREDICTIONS")
        print("="*60)
        
        # Load all saved models
        saved_models = load_saved_models()
        
        if not saved_models:
            print("❌ No saved models found. Please train at least one model first.")
            print("💡 Models are automatically saved when you train them.")
        else:
            print(f"\n✅ Loaded {len(saved_models)} model(s) for predictions")
            
            # Interactive model selection for predictions
            while True:
                print("\n" + "="*50)
                print("AVAILABLE MODELS FOR PREDICTIONS:")
                print("="*50)
                for i, (model_name, model) in enumerate(saved_models.items(), 1):
                    print(f"{i}. {model_name.replace('_', ' ').title()}")
                print("0. Exit predictions")
                
                try:
                    choice = input("\nSelect a model (0-{}): ".format(len(saved_models))).strip()
                    
                    if choice == '0':
                        print("Exiting prediction mode...")
                        break
                    
                    choice = int(choice)
                    if 1 <= choice <= len(saved_models):
                        model_names = list(saved_models.keys())
                        selected_model_name = model_names[choice - 1]
                        selected_model = saved_models[selected_model_name]
                        
                        print(f"\n🎯 Using {selected_model_name.replace('_', ' ').title()} for predictions")
                        
                        # Call appropriate prediction function
                        if selected_model_name == 'gaussian_process':
                            make_gaussian_process_predictions(selected_model, scaler, decimal_places)
                        elif selected_model_name == 'ridge_regression':
                            make_ridge_regression_predictions(selected_model, scaler, decimal_places)
                        elif selected_model_name == 'random_forest':
                            make_random_forest_predictions(selected_model, scaler, decimal_places)
                        elif selected_model_name == 'xgboost':
                            make_xgboost_predictions(selected_model, scaler, decimal_places)
                        elif selected_model_name == 'neural_network':
                            # For neural network, we need to check if y_scaler was saved
                            # For now, we'll try without it (model should handle scaling internally)
                            make_neural_network_regression_predictions(selected_model, scaler, y_scaler=None, decimal_places=decimal_places)
                        # Legacy classification models (kept for backward compatibility)
                        elif selected_model_name == 'knn':
                            print("⚠️  KNN model is for classification. Please use regression models.")
                        elif selected_model_name == 'naive_bayes':
                            print("⚠️  Naive Bayes model is for classification. Please use regression models.")
                        elif selected_model_name == 'logistic_regression':
                            print("⚠️  Logistic Regression model is for classification. Please use regression models.")
                        elif selected_model_name == 'svm':
                            print("⚠️  SVM model is for classification. Please use regression models.")
                        else:
                            print(f"❌ Unknown model type: {selected_model_name}")
                        
                    else:
                        print("❌ Invalid choice. Please select a valid option.")
                        
                except ValueError:
                    print("❌ Please enter a valid number.")
                except KeyboardInterrupt:
                    print("\n\nExiting prediction mode...")
                    break
    
    # SHAP Analysis Section
    if ENABLE_SHAP_ANALYSIS:
        print("\n" + "="*60)
        print("SHAP ANALYSIS")
        print("="*60)
        
        if not HAS_SHAP:
            print("❌ SHAP library not available. Please install with: pip install shap")
        else:
            # Load models for SHAP analysis (use same models as predictions)
            shap_models = load_saved_models()
            
            if not shap_models:
                print("❌ No saved models found. Please train at least one model first.")
            else:
                print(f"\n✅ Loaded {len(shap_models)} model(s) for SHAP analysis")
                
                # Interactive model selection for SHAP analysis
                while True:
                    print("\n" + "="*50)
                    print("AVAILABLE MODELS FOR SHAP ANALYSIS:")
                    print("="*50)
                    for i, (model_name, model) in enumerate(shap_models.items(), 1):
                        print(f"{i}. {model_name.replace('_', ' ').title()}")
                    print("0. Exit SHAP analysis")
                    
                    try:
                        choice = input("\nSelect a model for SHAP analysis (0-{}): ".format(len(shap_models))).strip()
                        
                        if choice == '0':
                            print("Exiting SHAP analysis mode...")
                            break
                        
                        choice = int(choice)
                        if 1 <= choice <= len(shap_models):
                            model_names = list(shap_models.keys())
                            selected_model_name = model_names[choice - 1]
                            selected_model = shap_models[selected_model_name]
                            
                            print(f"\n🎯 Performing SHAP analysis on {selected_model_name.replace('_', ' ').title()}")
                            
                            # Get y_scaler for neural network if available
                            y_scaler_shap = None
                            if selected_model_name == 'neural_network':
                                # Try to load y_scaler if it was saved
                                # For now, we'll compute it from training data
                                from sklearn.preprocessing import StandardScaler
                                y_scaler_shap = StandardScaler()
                                y_train_scaled = y_scaler_shap.fit_transform(y_train.reshape(-1, 1))
                            
                            # Perform SHAP analysis (capitalize feature names for display)
                            feature_names_display = [name.capitalize() for name in feature_cols]
                            perform_shap_analysis(
                                model=selected_model,
                                model_name=selected_model_name,
                                X_train=X_train,
                                X_test=X_test,
                                feature_names=feature_names_display,
                                scaler=scaler,
                                y_scaler=y_scaler_shap,
                                sample_size=100
                            )
                            
                        else:
                            print("❌ Invalid choice. Please select a valid option.")
                            
                    except ValueError:
                        print("❌ Please enter a valid number.")
                    except KeyboardInterrupt:
                        print("\n\nExiting SHAP analysis mode...")
                        break
    
    print("\n" + "="*60)
    print("SCRIPT EXECUTION COMPLETE")
    print("="*60)

