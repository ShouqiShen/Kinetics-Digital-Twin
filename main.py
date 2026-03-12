import os
import numpy as np
import tensorflow as tf
from src.data_loader import KineticDataLoader
from models.arch import build_hpinn_model

# --- Configuration ---
TEST_MOL = "C10"  # Target molecule for Leave-One-Molecule-Out (LOMO) validation
DATA_PATH = 'data/sample_data.csv' if os.path.exists('data/sample_data.csv') else 'data/Master_Kinetics_Dataset.csv'
WEIGHTS_DIR = 'models/weights'

def train_stages(model, X_train, y_train, validation_data, sample_weights):
    """
    Executes the 4-stage physics-informed training strategy.
    """
    def set_trainable(prefixes, status):
        for layer in model.layers:
            if any(layer.name.startswith(p) for p in prefixes):
                layer.trainable = status

    def compile_and_fit(lr, epochs, msg):
        print(f"\n>>> {msg} (LR={lr})")
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')
        model.fit(X_train, y_train, sample_weight=sample_weights, 
                  validation_data=validation_data, epochs=epochs, 
                  batch_size=64, verbose=1,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

    # Stage 1: Focus on Physics (Theta branch)
    set_trainable(["delta", "iso", "dyn_corr"], False)
    compile_and_fit(1e-3, 80, "Stage 1: Training Physics Branch")

    # Stage 2: Train Residuals (Delta branch)
    set_trainable(["theta", "struct"], False)
    set_trainable(["delta"], True)
    compile_and_fit(5e-4, 80, "Stage 2: Training Delta Residuals")

    # Stage 3: Fine-tune Offsets (Isothermal/Dynamic corrections)
    set_trainable(["delta"], False)
    set_trainable(["iso_dense", "dyn_corr"], True)
    compile_and_fit(5e-4, 80, "Stage 3: Training Correction Layers")

    # Stage 4: Joint Fine-tuning
    for layer in model.layers: layer.trainable = True
    compile_and_fit(1e-4, 40, "Stage 4: Global Fine-tuning")

if __name__ == "__main__":
    # Initialize Data
    loader = KineticDataLoader(DATA_PATH)
    data = loader.prepare_tensors()
    
    # LOMO Split
    df = loader.df
    train_idx = df[df['Molecule'] != TEST_MOL].index
    test_idx  = df[df['Molecule'] == TEST_MOL].index

    X_train = [inputs[train_idx] for inputs in data["inputs"]]
    X_test  = [inputs[test_idx] for inputs in data["inputs"]]
    y_train, y_test = data["labels"][train_idx], data["labels"][test_idx]

    # Sample Weighting: Mitigate high-conversion diffusion noise
    alpha_train = df.loc[train_idx, 'Alpha'].values
    weights = np.ones_like(alpha_train)
    weights *= 1.0 / (1.0 + np.exp(25.0 * (alpha_train - 0.92))) + 0.15

    # Build and Train
    model = build_hpinn_model(loader.max_smiles_len, loader.vocab_size, loader.max_atoms)
    train_stages(model, X_train, y_train, (X_test, y_test), weights)

    # Save Results
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    model.save_weights(f"{WEIGHTS_DIR}/hpinn_{TEST_MOL}.h5")
    print(f"Success: Weights saved for {TEST_MOL}")
