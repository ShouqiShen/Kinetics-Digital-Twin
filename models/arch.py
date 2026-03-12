import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from spektral.layers import GCNConv, GlobalSumPool
from src.physics_engine import phys_ln_rate

def build_hpinn_model(max_smiles_len, vocab_size, max_atoms):
    # --- Inputs ---
    in_smiles = Input(shape=(max_smiles_len,), name="smiles_tokens")
    in_nodes  = Input(shape=(max_atoms, 3), name="node_features")
    in_adjs   = Input(shape=(max_atoms, max_atoms), name="adjacency")
    in_ecfp   = Input(shape=(2048,), name="ecfp")
    in_state  = Input(shape=(2,), name="state_scaled")
    in_T      = Input(shape=(1,), name="Temp_K")
    in_alpha  = Input(shape=(1,), name="Alpha")
    in_mode   = Input(shape=(1,), name="is_dynamic")
    in_beta   = Input(shape=(1,), name="beta_scaled")
    in_tiso   = Input(shape=(1,), name="tiso_scaled")

    # --- Structure Encoder ---
    x1 = Embedding(vocab_size, 32, mask_zero=True)(in_smiles)
    x1 = Conv1D(64, 3, activation='relu', padding='same')(x1)
    x1 = GlobalMaxPooling1D()(x1)

    x2 = GCNConv(64, activation='relu')([in_nodes, in_adjs])
    x2 = GlobalSumPool()(x2)

    x3 = Dense(256, activation='relu')(in_ecfp)
    x3 = Dense(128, activation='relu')(x3)

    struct = Concatenate(name="struct_concat")([x1, x2, x3])
    struct = Dense(256, activation='relu')(struct)
    struct = Dropout(0.2)(struct) # Important for UQ
    struct = Dense(128, activation='relu')(struct)

    # --- Physics Branch (Theta Head) ---
    theta_in = Concatenate()([struct, in_mode, in_beta, in_tiso])
    theta_h  = Dense(128, activation='relu')(theta_in)
    theta_raw = Dense(7, name="theta_raw")(theta_h)
    lnr_phys = Lambda(lambda t: phys_ln_rate(t[0], t[1], t[2]), name="ln_rate_phys")([in_T, in_alpha, theta_raw])

    # --- Hybrid Branch (Delta Head) ---
    res_in = Concatenate()([struct, in_state, in_mode, in_beta, in_tiso])
    h = Dense(128, activation='relu')(res_in)
    h = Dense(64, activation='relu')(h)
    delta = Dense(1, name="delta")(h)

    lnr_hyb = Add(name="ln_rate_hybrid")([lnr_phys, delta])

    # --- Output & Regularization ---
    # ... (Keep your iso_offset and dyn_corr logic here)
    
    model = Model(
        inputs=[in_smiles, in_nodes, in_adjs, in_ecfp, in_state, in_T, in_alpha, in_mode, in_beta, in_tiso],
        outputs=lnr_hyb # Simplify for brevity, apply offsets as in original
    )
    
    return model
