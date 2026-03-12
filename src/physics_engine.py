import tensorflow as tf

# Constants
R = 8.314462618  # J/mol/K

def phys_ln_rate(T, alpha, theta_raw):
    """
    Physics-informed Mastercurve equation.
    T: Temperature in Kelvin
    alpha: Conversion degree (0-1)
    theta_raw: Raw kinetic parameters from neural network
    """
    logA, Eraw, mraw, nraw, craw, ac0raw, acTraw = tf.split(theta_raw, 7, axis=-1)

    # Physical constraints using activations
    A   = tf.exp(logA)
    Ea  = tf.nn.softplus(Eraw)
    m   = tf.nn.softplus(mraw)
    n   = tf.nn.softplus(nraw)
    c   = tf.nn.softplus(craw)
    ac0 = tf.sigmoid(ac0raw)
    acT = 3e-3 * tf.tanh(acTraw)

    a = tf.clip_by_value(alpha, 1e-6, 1.0 - 1e-6)

    # Kinetic equation: Arrhenius * Autocatalytic Model / Diffusion Gating
    num  = A * tf.exp(-Ea/(R*T)) * tf.pow(a, m) * tf.pow(1.0-a, n)
    gate = 1.0 + tf.exp(c*(a - ac0 + acT*T))
    rate = num / gate
    rate = tf.clip_by_value(rate, 1e-30, 1e30)
    
    return tf.math.log(rate)
