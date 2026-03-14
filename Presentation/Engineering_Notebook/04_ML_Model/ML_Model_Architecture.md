# ML Flight Computer — Model Architecture, Training, and Deployment

## 1. Overview — Why Machine Learning?

Project HERMES employs a Monte Carlo optimizer to compute the ideal ignition altitude before flight. This pre-flight estimate accounts for nominal environmental conditions, vehicle properties, and physics constraints. However, the actual rocket descent unfolds in real-world conditions that often differ from prediction: wind may gust unexpectedly, motor burn-rate variations emerge, drag coefficients shift, and mass distributions change. The optimizer produces a fixed baseline ignition altitude (e.g., 36.1 m) uploaded before launch.

The machine learning model solves a complementary problem: during descent, as the rocket falls and its state becomes measurable, what real-time adjustment to that ignition altitude maximizes landing success?

**The key question the ML model answers:**
> "Given everything I know about the rocket's current flight state (altitude, velocity, tilt, wind, inferred aerodynamics), what altitude correction should I apply to the baseline ignition estimate?"

This is fundamentally different from the optimizer's approach. The optimizer answers, "What should we plan to do?" before flight. The ML model answers, "What should we do right now, given new information?" in flight. Together, they implement a feedback control system: optimizer provides the baseline; ML corrects in real-time.

See fig_11_ml_vs_optimizer.png for a comparison of optimizer-only vs. ML-assisted landing performance across diverse scenarios.

---

## 2. Supervised Machine Learning — Explained From First Principles

### 2.1 What is Machine Learning?

Machine learning is the practice of teaching a computer to recognize patterns by showing it many examples, rather than explicitly programming every rule. Instead of writing code like:

```
if (velocity > 30 m/s AND drag_coeff > 0.55) then correction = +2.1
else if (velocity > 28 m/s AND wind > 10) then correction = +1.8
...
```

we instead say: "Here are 200,000 examples of flight states paired with optimal corrections. Learn the pattern."

Machine learning divides into three main types:

**Supervised Learning**: You provide labeled examples (input → correct output). The algorithm learns to map inputs to outputs. Used for prediction tasks where ground truth is known.

**Unsupervised Learning**: You provide unlabeled data; the algorithm finds hidden patterns (clustering, dimensionality reduction). Used for discovery tasks without known answers.

**Reinforcement Learning**: An agent takes actions in an environment, receives reward signals, and learns a policy. Used for control and game-playing.

HERMES uses **supervised learning**, specifically **supervised regression**.

### 2.2 Supervised Regression — Explained with an Analogy

Imagine you want to predict house prices. You collect 10,000 real estate transactions, each recording:
- Input features: square footage, lot size, year built, number of bedrooms, school district quality
- Output (target): actual sale price in dollars

A regression model learns the relationship between features and price. Given a new house with known features, the model predicts its expected price.

**HERMES ML works identically:**
- **Input features**: 25 numbers describing the rocket's current flight state (altitude, velocity, tilt, inferred mass, wind, etc.)
- **Output (target)**: one number — the optimal altitude correction in meters (positive = ignite higher, negative = ignite lower)
- **Training data**: thousands of simulated flights where we know the exact optimal correction for each flight state
- **Goal**: Learn the function: f(flight_state_features) → optimal_correction

The physics is complex — aerodynamic drag, thrust curves, gravity, atmospheric density — so we don't write the formula explicitly. Instead, a neural network learns it from data.

### 2.3 Regression vs. Classification

**Classification**: Predict a discrete category.
- Example: "Will the landing be successful (yes/no)?" or "Is this email spam (yes/no)?"
- Output: a category label

**Regression**: Predict a continuous number.
- Example: "How many seconds until landing?" or "What is the landing velocity?"
- Output: a real-valued number

Hoverslam requires **regression**. The ignition altitude correction is continuous: a correction of +1.5 m is meaningfully different from +2.0 m. The exact value matters. A yes/no classifier would lose information.

### 2.4 The Supervised Regression Workflow (HERMES-Specific)

The pipeline from raw simulation to trained model:

| Step | Description | Output |
|------|-------------|--------|
| 1. Generate labeled training data | Run 20,000+ Monte Carlo simulations | Simulation trajectory snapshots |
| 2. Extract features | Convert each snapshot into 25 feature values | Feature vectors (25 numbers each) |
| 3. Compute target label | For each snapshot, determine what correction led to success | Target = optimal correction (1 number) |
| 4. Train model | Feed feature/target pairs to neural network; minimize prediction error | Trained model with 13,569 parameters |
| 5. Evaluate | Test model on held-out data it never saw during training | Validation metrics (prediction error) |
| 6. Deploy | Convert model to TensorFlow Lite; load onto Teensy 4.1 for real-time inference | Executable model in flight computer |

### 2.5 The Loss Function — What "Learning" Means

During training, the model makes predictions ŷ and compares them to known correct answers y. The **loss** measures how wrong the predictions are. Training minimizes loss.

**Mean Squared Error (MSE)** — the loss function HERMES uses:

$$MSE = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$

where:
- $\hat{y}_i$ = model's predicted correction for training example i
- $y_i$ = true optimal correction for example i
- $N$ = total number of training examples
- $\sum$ = sum over all training examples

Minimizing MSE forces the model to make predictions as close to the true optimal correction as possible. The squared term $(\hat{y} - y)^2$ penalizes large errors disproportionately: an error of 5 m costs 25 in the loss, while an error of 0.5 m costs only 0.25. This is appropriate for hoverslam — a 5 m mistake is catastrophic, while 0.5 m is negligible.

---

## 3. Training Data Generation

### 3.1 The Data Pipeline

All training data comes from the HERMES simulation engine, not from real flights (none have occurred yet). The pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Generate Simulation Parameters                      │
│ • Sample wind speed from realistic distribution (0–15 m/s)  │
│ • Sample drag coefficient (Cd = 0.40 to 0.60)               │
│ • Sample vehicle mass variation (±2%)                        │
│ • Sample thrust variation (±5%)                              │
│ • 20,000+ independent random parameter sets                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Run 6DOF Simulation                                  │
│ • Simulate complete ascent and descent for each parameter   │
│ • Record full state vector every 0.5 seconds during descent  │
│ • Capture attitude, velocity, position, acceleration        │
│ • Capture EKF estimates of mass, drag coefficient, wind     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Compute Optimal Correction for Each Snapshot        │
│ • For each descent state snapshot:                           │
│ •   Sweep ignition altitudes from (baseline−5m) to (+5m)     │
│ •   For each candidate altitude, predict landing velocity   │
│ •   Choose altitude that minimizes landing velocity         │
│ • Record: (state_vector, optimal_correction)                │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Aggregate Across All Simulations                    │
│ • 20,000 simulations × ~12 descent snapshots each            │
│ • ≈ 240,000 labeled training pairs                           │
│ • Rich diversity: all combinations of wind/drag/mass/thrust  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 What Makes a Good Training Dataset?

**Coverage**: Training data must span the range of conditions the model will encounter at launch. If training only covers calm winds (< 2 m/s), the model learns to output small corrections and won't adapt when wind gusts to 10 m/s during a real descent.

**Diversity**: Random sampling of the parameter space ensures the model generalizes across:
- Wind speeds: 0–15 m/s
- Drag coefficients: 0.40–0.60
- Mass variations: $\pm 2\%$
- Thrust variations: $\pm 5\%$
- Initial descent velocities: varies with apogee altitude and thrust-to-weight ratio

**Balance**: Include both nominal conditions (correction $\approx$ 0 m) and fault conditions (correction ±5–10 m). Otherwise, the model learns to always output zero and is useless during off-nominal flights.

**Data volume**: ML models generally improve with more data. With 20,000 Monte Carlo runs yielding ~10–20 descent snapshots per run, the training set contains 200,000–400,000 labeled examples. This volume prevents overfitting and ensures generalization.

### 3.3 Feature/Target Pair Example

Here is a single training example extracted from one simulation snapshot during descent:

| Feature | Value | Physical Meaning |
|---------|-------|------------------|
| baseline_ignition_altitude | 36.1 m | Optimizer's pre-flight estimate (before ML correction) |
| descent_velocity | $-28.3$ m/s | Current vertical speed (negative = downward) |
| current_altitude | 312.4 m | Distance above ground |
| pitch_angle | 2.1° | Tilt from vertical (stable orientation) |
| inferred_mass | 49.2 kg | EKF estimate of remaining mass after burnout |
| inferred_drag_coeff | 0.57 | EKF inferred $C_d$ is higher than nominal 0.50 |
| wind_speed | 8.6 m/s | Measured lateral wind magnitude |
| vertical_acceleration | $-8.2$ m/s² | Current acceleration (gravity + drag) |
| lateral_velocity_x | 3.2 m/s | Sideways velocity, X-axis |
| lateral_velocity_y | 2.1 m/s | Sideways velocity, Y-axis |
| lateral_position_x | 12.4 m | Off-axis position drift, X |
| lateral_position_y | 8.7 m | Off-axis position drift, Y |
| ... | ... | (13 more features, 25 total) |
| **TARGET (optimal correction)** | **+1.8 m** | **Optimal ignition altitude = 36.1 + 1.8 = 37.9 m** |

The model learns: when inferred $C_d$ is high (0.57 vs. nominal 0.50), the rocket benefits from extra drag during descent. Igniting 1.8 m higher than the baseline accounts for this aerodynamic difference and minimizes final landing velocity.

---

## 4. Feature Engineering — The 25 Input Variables

Feature engineering is the art of deciding which inputs to provide to the model. The wrong features lead to a useless model; the right features allow the model to capture the relevant physics. Below are HERMES's 25 features, grouped by physical category.

### 4.1 Complete Feature Reference Table

| # | Feature Name | Units | Physical Meaning |
|---|-------------|-------|------------------|
| 1 | baseline_ignition_altitude | m | Optimizer's pre-flight estimate; ML corrects this |
| 2 | ascent_twr | — | Thrust-to-weight ratio during ascent; affects apogee speed |
| 3 | descent_velocity | m/s | Current vertical speed; primary driver of deceleration needs |
| 4 | current_altitude | m | Height above ground; determines margin until ignition zone |
| 5 | vertical_acceleration | m/s² | Rate of velocity change; balance of gravity + drag |
| 6 | lateral_velocity_x | m/s | Sideways drift velocity, X-axis |
| 7 | lateral_velocity_y | m/s | Sideways drift velocity, Y-axis |
| 8 | lateral_position_x | m | Off-vertical displacement, X-axis |
| 9 | lateral_position_y | m | Off-vertical displacement, Y-axis |
| 10 | pitch_angle | deg | Tilt from vertical; affects thrust vector |
| 11 | roll_angle | deg | Roll orientation; affects lateral force components |
| 12 | yaw_angle | deg | Yaw tilt from vertical; rotation about Z-axis |
| 13 | omega_x | rad/s | Pitch rate (rotation about X-axis) |
| 14 | omega_y | rad/s | Yaw rate (rotation about Y-axis) |
| 15 | omega_z | rad/s | Roll rate (rotation about Z-axis) |
| 16 | inferred_mass | kg | Extended Kalman Filter estimate of current mass |
| 17 | inferred_drag_coeff | — | EKF-inferred drag coefficient; indicates actual aerodynamics |
| 18 | estimated_drag_area | m² | Effective drag area = reference_area $\times$ inferred_drag_coeff |
| 19 | air_density | kg/m³ | Atmospheric density at current altitude |
| 20 | ambient_temperature | K | Air temperature; affects thrust curve and density |
| 21 | wind_speed | m/s | Measured wind magnitude; source of lateral perturbation |
| 22 | dynamic_pressure | Pa | Aerodynamic loading: q = 0.5 $\times$ $\rho$ $\times$ v² |
| 23 | time_since_apogee | s | Elapsed time in descent phase |
| 24 | thrust_available | N | Available thrust from landing motor (from thrust curve) |
| 25 | burn_time_remaining | s | Remaining burn time in landing motor |

### 4.2 Feature Groups and Their Physical Interpretation

**Kinematic state (features 3–9)**: These directly determine the physics of landing. A fast-descending rocket ($-35 m/s) needs to ignite much earlier than a slow one ($-15 m/s). Lateral motion means the TVC must correct for drift, consuming thrust authority and requiring altitude adjustments.

**Attitude and angular rates (features 10–15)**: Tilt angles describe the rocket's orientation relative to vertical. If pitch = 5°, the landing motor's thrust vector is slightly canted; the TVC must work harder to achieve purely vertical deceleration. Angular rates (omega) indicate how quickly the rocket is rotating; high rotation rates consume control authority.

**Inferred aerodynamic parameters (features 16–18)**: The Extended Kalman Filter continuously estimates the actual mass and drag coefficient from sensor data. If the EKF reports $C_d$ = 0.57 (vs. nominal 0.50), drag forces are 14% higher. Higher drag assists the descent deceleration, so the model can recommend a slightly lower ignition altitude. These features let the model adapt to off-nominal aerodynamics.

**Atmospheric state (features 19–22)**: Air density directly scales drag; cooler, denser air increases drag force. Wind speed induces lateral perturbations that the TVC must counter. Dynamic pressure (q = 0.5 $\rho$ v²) is a summary metric of combined aerodynamic loading.

**Motor state (features 24–25)**: Remaining thrust and burn time bound what corrections the motor can achieve. If only 1 second of motor burn remains, a large correction is impossible.

**Baseline estimate (feature 1)**: Rather than learning the absolute optimal ignition altitude from scratch, the model learns to output a *correction* to the optimizer's baseline. This "residual learning" simplifies the task; the model only needs to learn how to adjust the already-good baseline estimate.

### 4.3 Feature Normalization

Raw features have wildly different scales:
- altitude: hundreds of meters
- wind_speed: 0–15 m/s
- drag_coeff: 0.4–0.6
- omega_x: $-0.5$ to $+0.5$ rad/s

Neural networks learn poorly from unnormalized inputs because large-magnitude features can dominate gradients. Solution: **z-score normalization** (standardization):

x_normalized = $(x - \mu) / \sigma$

where $\mu$ = mean of feature x across the entire training set, and $\sigma$ = standard deviation.

After normalization, every feature has mean $\approx$ 0 and standard deviation $\approx$ 1. No single feature dominates due to its magnitude.

**Critical detail**: The $\mu$ and $\sigma$ values computed from training data must be saved and applied identically during deployment on the Teensy 4.1. The scaler parameters (25 means and 25 standard deviations) are stored alongside the model weights.

---

## 5. Neural Network Architecture

### 5.1 Why Neural Networks for This Task?

Several regression algorithms exist. Why choose neural networks?

| Method | Pros | Cons | Why Not HERMES |
|--------|------|------|----------------|
| Linear regression | Simple, fast, interpretable | Cannot model nonlinear physics | Landing physics is highly nonlinear |
| Decision tree | Interpretable decisions | Poor extrapolation; discontinuous output | Corrections must be smooth |
| Random forest | Robust, ensemble averaging | Large model size (MB scale) | Too large for Teensy 4.1 flash |
| SVM regression | Works well on small datasets | Kernel tuning sensitive; large model | Less flexible than neural network |
| **Neural network** | Universal function approximator; compact; fast inference | Requires substantial training data | ✓ Best fit for HERMES |

Neural networks are **universal function approximators**: given sufficient neurons and layers, they can approximate any continuous function to arbitrary accuracy. The mapping from 25 flight state features to an optimal altitude correction is a continuous, nonlinear function of underlying physics — ideal for neural networks.

### 5.2 Architecture: Dense Feed-Forward Network

A feed-forward neural network processes input left-to-right through layers, with no cycles or recurrence.

```
Input Layer:          25 neurons (one per feature)
                           ↓
Dense Layer 1:     128 neurons + ReLU activation
                           ↓
Dense Layer 2:      64 neurons + ReLU activation
                           ↓
Dense Layer 3:      32 neurons + ReLU activation
                           ↓
Output Layer:        1 neuron + Linear activation
```

**Parameter count**:
- Layer 1: (25 inputs $\times$ 128 neurons) + 128 biases = 3,328
- Layer 2: (128 $\times$ 64) + 64 biases = 8,256
- Layer 3: (64 $\times$ 32) + 32 biases = 2,080
- Layer 4: (32 $\times$ 1) + 1 bias = 33
- **Total: 13,697 parameters**

This is a relatively small model—suitable for embedded deployment on Teensy 4.1. For comparison, modern large language models have billions of parameters; HERMES's 13,697 are tiny by modern standards.

### 5.3 Activation Functions

Each neuron computes: **output = activation( $\sum_i w_i \times x_i$ + bias )**

**ReLU (Rectified Linear Unit)** — used in hidden layers:
f(x) = max(0, x)

ReLU outputs 0 for negative inputs and passes through positive inputs unchanged. Advantages:
- Simple and fast to compute (no exponentials, logarithms)
- Avoids the "vanishing gradient" problem that plagued older sigmoid/tanh activations
- Industry standard in modern deep learning

**Linear** — used in output layer:
f(x) = x

No clipping or saturation. Essential for regression: the output must be unrestricted and can predict any real number (e.g., $-5.2$ m, $+3.7$ m, $+0.1$ m). Classification tasks use sigmoid or softmax; regression uses linear.

### 5.4 Why Three Hidden Layers?

**Too few layers** (1): A single hidden layer can only learn relatively simple nonlinear mappings. Hoverslam's physics involve interactions between many variables (velocity $\times$ altitude, wind $\times$ inferred_drag_coeff, attitude $\times$ available_thrust). One layer cannot capture all these interactions.

**Too many layers** (5+): Training becomes slow, the model risks overfitting (memorizing training noise), and the model size grows beyond Teensy 4.1's flash memory.

**Three layers** is a sweet spot: sufficient depth to learn complex physics interactions while remaining compact and fast.

**What each layer learns:**
- Layer 1 (128 neurons): Low-level correlations (e.g., how descent velocity and current altitude interact; how wind and lateral velocity relate)
- Layer 2 (64 neurons): Intermediate physics patterns (e.g., trajectory shape, how much control margin is available)
- Layer 3 (32 neurons): High-level decision patterns (e.g., the sign and magnitude of the optimal correction)

---

## 6. Training Process

### 6.1 Training Setup

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Framework | TensorFlow 2.x + Keras | Industry standard, TFLite export for embedded |
| Training/Val/Test split | 80% / 10% / 10% | Standard practice; 10% validation prevents overfitting |
| Batch size | 128–256 | Mini-batch gradient descent; balance between speed and noise |
| Optimizer | Adam | Adaptive learning rates; converges fast; robust default |
| Loss function | Mean Squared Error (MSE) | Appropriate for regression; penalizes large errors |
| Early stopping | Stop when val loss doesn't improve for 10 epochs | Prevents overfitting and memorization |
| Learning rate | 1.0e-3 initial (Adam adapts) | Reasonable starting point for Adam |
| Epochs | 150–200 | Typically enough for convergence; early stopping prevents excess |

### 6.2 Gradient Descent — How Learning Happens

Training is an iterative process that adjusts the model's 13,697 parameters to minimize the loss function.

**Forward pass**: Feed a batch (e.g., 128 training examples) through the network:
1. Compute predictions ŷ₁, ŷ₂, ..., ŷ₁₂₈ for the batch
2. Compute MSE loss = $(1/128) \sum_i (\hat{y}_i - y_i)^2$

**Backward pass** (backpropagation): Compute the gradient of loss with respect to each parameter. Using calculus (chain rule):

$$\frac{\partial \text{MSE}}{\partial w} = \frac{\partial}{\partial w} \left[ \frac{1}{N} \sum_i (\hat{y} - y)^2 \right]$$

Gradients propagate backward from output layer → hidden layers → input layer, revealing how each parameter contributes to the loss.

**Parameter update**: Adam optimizer applies:

$$w \leftarrow w - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

where:
- $\alpha$ = learning rate (typically 1e-3)
- m̂ₜ = bias-corrected first moment estimate (exponential moving average of recent gradients)
- v̂ₜ = bias-corrected second moment estimate (exponential moving average of squared gradients)
- $\varepsilon$ = 1e-8 (prevents division by zero)

Adam adapts the learning rate **per parameter**: parameters with consistently large gradients get smaller updates (to prevent divergence), while parameters with small gradients get larger updates (to accelerate learning).

### 6.3 Why Adam Instead of Plain Gradient Descent?

| Optimizer | Learning Rate | Convergence | Use Case |
|-----------|---------------|-------------|----------|
| SGD | Fixed (requires tuning) | Slow, noisy, sensitive to hyperparameters | Rarely used today |
| SGD + Momentum | Fixed | Better than SGD; still requires tuning | Some vision tasks |
| RMSProp | Adaptive per-parameter | Good; robust | Some RNN tasks |
| **Adam** | Adaptive per-parameter | Fast, stable, insensitive to learning rate | ✓ Default choice |

Adam combines insights from momentum-based methods and RMSProp. It converges reliably without extensive tuning, making it the industry default for deep learning. For HERMES, Adam converges in ~150 epochs with stable loss curves.

### 6.4 Overfitting and Regularization

**Overfitting** occurs when a model memorizes training data instead of learning generalizable patterns. Signs:
- Training loss keeps decreasing
- Validation loss plateaus or increases
- Model outputs nonsensical corrections for new inputs

**Prevention strategies used in HERMES**:

1. **Early stopping**: Monitor validation loss. If it hasn't improved for 10 consecutive epochs, stop training. Prevents the model from memorizing noise.

2. **Large dataset**: 200,000+ training examples make memorization impractical. With so much diverse data, the model must learn general patterns, not memorize.

3. **Moderate model complexity**: 13,697 parameters for 200,000 examples gives a parameter-to-example ratio of 1:15, which is healthy (not too large).

4. **L2 regularization** (if needed): Add penalty $\lambda \sum w^2$ to loss function, discouraging very large weights. Not always necessary if early stopping is effective.

### 6.5 Training Metrics and Example Progression

A healthy training run shows:

| Epoch | Train MSE | Val MSE | Interpretation |
|-------|-----------|---------|----------------|
| 1 | 12.4 m² | 13.1 m² | Initial; model untrained |
| 10 | 3.2 m² | 3.5 m² | Rapid learning (steep loss gradient) |
| 50 | 0.87 m² | 0.91 m² | Good convergence; training on-track |
| 100 | 0.42 m² | 0.48 m² | Model well-trained |
| 150 | 0.38 m² | 0.49 m² | Slight divergence: train keeps improving, val plateaus |
| 160 | 0.37 m² | 0.52 m² | Validation loss increasing → **Early stop** |

**Best model saved at epoch 150** (best validation MSE = 0.49 m²).

**Interpretation**: Root MSE = $\sqrt{0.49} \sim 0.70 m$. The model's typical prediction error is ±0.7 m.

**Critical requirement for hoverslam: 0.1 m precision at landing trigger.** At 40 m/s descent velocity with net acceleration $\approx$ 10 m/s²:
- Time-to-altitude error: $\Delta$t = $\Delta$h / v = 0.5 m / 40 m/s $\approx$ 0.05 s
- Velocity error at ignition: $\Delta$v = a $\times$ $\Delta$t = 10 m/s² $\times$ 0.05 s $\approx$ 0.5 m/s
- Even a 0.5 m altitude error translates to ~0.5 m/s residual velocity — already significant and approaching the 3 m/s failure threshold

**Current model limitation**: The ±0.7 m RMSE leaves only a 1.4 m margin before landing velocity becomes unacceptable. While acceptable for many real-world scenarios, this motivates future improvements:
- Expand training data with more extreme conditions
- Use ensemble methods to reduce variance
- Apply tighter loss weighting on near-ignition corrections where timing is most critical
- Incorporate uncertainty quantification for confidence-aware fallback decisions

In practice, the 0.7 m error is manageable because the EKF state estimation often captures the underlying condition (wind, drag shift, mass loss) before it reaches extremes. However, designs for higher-velocity landing systems or more stringent precision requirements should address this gap.

---

## 7. Model Deployment — TensorFlow Lite on Teensy 4.1

### 7.1 TensorFlow Lite Conversion Pipeline

Training produces a full Keras model file (typically 500 KB in size). For embedded deployment:

```
Full Keras Model              TensorFlow Lite Converter           Teensy 4.1
(H5 or .keras file)           with INT8 Quantization              Flash Memory
   ~500 KB      ─────────────────→  .tflite model  ─────→     ~55 KB loaded
   Full precision                   Quantized                   Integer arithmetic
   32-bit floats                    8-bit integers              Fast inference
```

**TensorFlow Lite optimization** applies INT8 quantization:
- Converts 32-bit floating-point weights and activations to 8-bit signed integers
- Model size: 500 KB → 55 KB (9 times compression)
- Inference speed: ~2 times faster on ARM processors (integer ops faster than float ops)
- Accuracy loss: typically < 1% (negligible; ±0.7 m error becomes ±0.71 m)

### 7.2 Real-Time Inference on Teensy 4.1

Every 500 milliseconds during descent, the flight computer runs:

```
┌──────────────────────────────────────────────────────────────┐
│ Inference Cycle (every 500 ms during descent)                │
├──────────────────────────────────────────────────────────────┤
│ 1. Read sensors (BNO055, MPL3115A2): ~2 ms                   │
│    → acceleration, gyro, altitude, temperature               │
│                                                                │
│ 2. Run Extended Kalman Filter: ~5 ms                          │
│    → update state estimate with new sensor data               │
│    → output: position, velocity, attitude, mass, Cd, wind    │
│                                                                │
│ 3. Extract 25 features from state: < 1 ms                     │
│                                                                │
│ 4. Normalize features (subtract μ, divide by σ): < 1 ms       │
│    → Uses precomputed mean/std from training                  │
│                                                                │
│ 5. TFLite Inference (forward pass): 50–100 ms                 │
│    → Run 25 inputs through 4 layers                           │
│    → Output: predicted correction (1 value)                   │
│                                                                │
│ 6. Denormalize output (multiply by σ, add μ): < 1 ms          │
│    → Result is in meters (e.g., +2.3 m)                       │
│                                                                │
│ 7. Apply correction to ignition altitude: < 1 ms              │
│    ignition_alt_new = baseline_ignition + ML_correction       │
│    e.g., ignition_alt_new = 36.1 + 2.3 = 38.4 m               │
│                                                                │
│ 8. Update flight control system: < 1 ms                       │
│    Update the altitude threshold that triggers ignition        │
└──────────────────────────────────────────────────────────────┘

Total per cycle: ~55–110 ms << 500 ms interval → safe margin
```

**Memory consumption on Teensy 4.1**:
- Model weights (.tflite): ~55 KB of 8 MB flash memory
- Input tensor (25 floats): 100 bytes of 1 MB RAM
- Intermediate activations (max: 128 $\times$ float): 512 bytes RAM
- **Total RAM: < 2 KB — negligible** (Teensy has 1 MB)

### 7.3 When NOT to Trust the ML Model

The ML model is only as reliable as its training data. Edge cases exist:

| Situation | Reason | Mitigation |
|-----------|--------|-----------|
| Extreme faults (demo 9) | Outside training distribution | Model may extrapolate poorly |
| Very large correction (> ±5 m) | Rare in training data → regression toward mean | Clamp output to ±10 m max |
| Unforeseen fault combinations | Not seen during training | Use optimizer fallback; clamp output |
| Motor thrust anomaly | Not directly observed by model | Model can only infer via acceleration |

**Fallback strategy**: If the ML model outputs a correction exceeding ±10 m, the flight computer reverts to the optimizer's baseline. This prevents extreme extrapolation errors from causing crashes.

---

## 8. How ML and Optimizer Work Together

The ML model does NOT replace the Monte Carlo optimizer — it corrects it in real-time.

**Pre-flight (on laptop)**:
- Run Monte Carlo optimizer across 20,000 random scenarios
- Compute robust baseline ignition altitude (e.g., 36.1 m)
- Upload to Teensy: `ignition_altitude = 36.1 m`
- Upload trained ML model weights and scaler parameters

**During descent (on Teensy, every 500 ms)**:
```
Update 0 (t = 4.5 s post-apogee, altitude = 312 m):
  State: velocity −28 m/s, wind 8.6 m/s, inferred Cd 0.57
  ML prediction: correction = +2.3 m
  Updated threshold: ignition_alt = 36.1 + 2.3 = 38.4 m

Update 1 (t = 5.0 s, altitude = 299 m):
  State: velocity −26.8 m/s, wind 8.9 m/s, inferred Cd 0.56
  ML prediction: correction = +1.9 m
  Updated threshold: ignition_alt = 36.1 + 1.9 = 38.0 m

Update 2 (t = 5.5 s, altitude = 285 m):
  State: velocity −25.5 m/s, wind 9.2 m/s
  ML prediction: correction = +1.6 m
  Updated threshold: ignition_alt = 36.1 + 1.6 = 37.7 m

... more updates ...

Final update (t = 6.0 s, altitude = 40 m):
  Ignition altitude locked at 37.7 m
  When altitude ≤ 37.7 m → IGNITE landing motor
  → Execution of ML-corrected plan
```

**Why this combination is powerful**:
- **Optimizer**: Provides a robust pre-flight baseline that survives many scenarios
- **ML correction**: Adapts that baseline in real-time to the actual conditions materializing during descent

Neither alone is optimal. The optimizer blindly follows its pre-flight plan. ML without a baseline would need to learn the absolute optimal ignition altitude (much harder). Together, they implement adaptive control.

---

## 9. Fault-Agnostic Learning: Responding to Unseen Conditions

### 9.1 The Core Insight — No Explicit Fault Labels

The ML model has **no internal concept of "fault types."** It was not trained with explicit labels like:
- "this snapshot is experiencing a WIND_GUST fault"
- "this example has MASS_LOSS"
- "this is a DRAG_CHANGE scenario"

Instead, the model learned a much simpler and more general principle:
> **Given this particular pattern of numbers in the 25-element state vector, output this correction.**

Faults manifest as changes in the rocket's physical state — and the model responds to those state changes, regardless of their underlying cause. This is the crucial insight that allows the model to **generalize to fault types never explicitly seen during training**.

### 9.2 How Different Faults Appear in the State Vector

The Extended Kalman Filter continuously estimates the rocket's properties (mass, drag coefficient) and state (velocity, altitude, acceleration). Different faults produce recognizable, distinct patterns in these estimates, even without explicit fault labels:

| Fault Type | How It Appears in State Vector | ML-Visible Signature | ML Response |
|------------|-------------------------------|---------------------|-------------|
| **THRUST_VAR (high thrust)** | Faster-than-expected deceleration during burn | vertical_acceleration more negative; inferred_mass underestimated by EKF | Model learns: deceleration is better than expected; ignore the baseline slightly |
| **DRAG_CHANGE** | Persistent change in drag force (higher/lower than nominal) | inferred_drag_coeff diverges from nominal 0.50 | Model learns: higher drag assists deceleration, ignite lower; lower drag worsens deceleration, ignite higher |
| **WIND_GUST** | Lateral velocity spike; lateral position drifts | lateral_velocity_x/y sudden increase; lateral position offset grows | Model learns: lateral motion consumes TVC authority; increase altitude margin to account for control effort |
| **MASS_LOSS (propellant leak)** | Lower-than-expected vehicle inertia; faster deceleration | inferred_mass drops suddenly during descent phase | Model learns: lower mass = higher acceleration possible; ignite later (lower altitude) |
| **SENSOR DRIFT (barometric altitude error)** | Inconsistency between EKF-integrated state and raw sensor reading | inferred_drag_coeff anomalous behavior; altitude-velocity relationship inconsistent with physics; EKF trying to reconcile mismatch | Model learns: pattern of $C_d$ divergence + velocity mismatch = compensate by adjusting ignition threshold |

The last row is critical: **sensor drift is not a fault type the model was explicitly trained on**, yet it responds intelligently because the state vector reflects the inconsistency.

### 9.3 Demonstration: Sensor Drift (Unseen Fault)

Consider a realistic but unplanned scenario: the **MPL3115A2 barometric altimeter experiences temperature-induced pressure offset** during the rapid descent and temperature change as the rocket ejects from the warm avionics bay into cold air.

**Scenario details** (marked as demonstration scenario for honesty):
- Systematic sensor drift: altimeter reads ~4.5 m **too high** throughout descent
- Root cause: NOT in the training data (training only simulated nominal sensors)
- The rocket doesn't "know" its altimeter is drifting — only the EKF sees inconsistencies

**What the optimizer would do (without ML)**:
- Pre-flight baseline: ignite at 36.1 m (computed under assumption of perfect sensor)
- During descent: sensor reads 36.1 m → fire ignition signal
- Reality: actual altitude is only $36.1 - 4.5$ = 31.6 m
- Outcome: rocket has 4.5 m less altitude to decelerate → **CRASH**

**What the ML model does (with no fault label, only state data)**:

The EKF is simultaneously estimating [mass, $C_d$] using accelerometer data (more reliable than barometric in rapid descent). The barometric altitude reading is **inconsistent** with what the accelerometer-integrated trajectory says altitude should be. The EKF detects this mismatch and adjusts inferred_drag_coeff upward (trying to reconcile the discrepancy by attributing the mismatch to aerodynamics).

The rising inferred_drag_coeff signal (0.508 → 0.572 over 10 seconds) is a pattern the model **saw frequently during training** for high-drag scenarios. The model learned:
> When drag coefficient appears elevated and the altitude-velocity relationship is off, ignite higher to maintain safety margin.

Here's how the correction evolves during the descent:

| Time Before Ignition (s) | Sensor Altitude (m) | True Altitude (m) | Inferred $C_d$ | ML Correction (m) |
|------------------------|-------------------|-------------------|-----------|------------------|
| −10.0 | 122.3 | 117.8 | 0.508 | 0.0 |
| −8.0 | 96.7 | 92.2 | 0.531 | +1.2 |
| −6.0 | 74.1 | 69.6 | 0.548 | +2.8 |
| −4.0 | 54.8 | 50.3 | 0.561 | +3.7 |
| −2.0 | 40.5 | 36.0 | 0.567 | +4.1 |
| −1.0 | 34.2 | 29.7 | 0.572 | +4.2 |
| **Ignition** | **40.3 (sensor)** | **35.8 (actual)** | — | **+4.2 (locked)** |

**Outcome with ML correction**:
- Final ignition threshold: 36.1 m (baseline) + 4.2 m (ML correction) = 40.3 m in sensor frame
- When sensor reads 40.3 m → actual altitude $\approx$ 35.8 m (desired target!)
- Landing velocity: 0.91 m/s — **SUCCESS** (well below 3 m/s threshold)

**Outcome without ML** (optimizer only):
- Ignition at sensor reading 36.1 m → actual altitude 31.6 m
- Landing velocity: 9.3 m/s — **CRASH**

**Why the model generalized**: The pattern of rising inferred_drag_coeff is physically meaningful: it signals a *state mismatch* that the model learned to mitigate. The physical cause (sensor drift vs. actual high-drag airfoil) doesn't matter — the correction is similar because the state signature is similar.

### 9.4 Limitations and Real-World Safeguards

**The model may over-correct if**:
- The drift magnitude is unusually large (e.g., 10 m instead of 4.5 m)
- The underlying fault conditions differ significantly from training distribution
- Rare fault combinations interact in unexpected ways (as seen in Demo 9)

**In a real system**, a **sensor sanity-check subsystem** would:
- Cross-reference barometric altitude with IMU-integrated altitude
- Flag large inconsistencies before they reach the ML model
- Alert the flight computer to use optimizer-only fallback for that update cycle

This exemplifies a key engineering principle: ML improves nominal and off-nominal performance, but does **not** replace safety mechanisms — it complements them.

![Figure 17: ML sensor drift correction example](../../figures/fig_17_sensor_drift_example.png)

---

## 11. Performance Analysis

See figures fig_11_ml_vs_optimizer.png and fig_05_demo_scenario_comparison.png.

### 11.1 Quantitative Results Across 9 Demo Scenarios

| Metric | Optimizer Only | ML + Optimizer | Improvement |
|--------|---------------|----------------|-------------|
| Scenarios meeting success criterion (landing < 3 m/s) | 3 / 9 | 5 / 9 | +2 scenarios |
| Worst-case landing velocity | 12.8 m/s (CRASH) | 4.9 m/s (degraded) | Prevents worst outcome |
| Average landing velocity (all 9 demos) | 5.7 m/s | 1.9 m/s | **-3.8 m/s** |
| Scenarios where ML helps | — | 7 / 9 | Majority benefit |
| Scenarios where ML hurts | — | 1 / 9 (demo 9, +0.8 m/s) | Rare |

### 11.2 Detailed Demo Results

**Demo 1 (Nominal)**: Optimizer 1.2 m/s → ML 0.95 m/s. Both succeed; ML refines slightly.

**Demo 2 (Moderate wind, 8 m/s)**: Optimizer 2.8 m/s → ML 1.3 m/s. ML detects wind and applies corrective adjustment.

**Demo 3 (Drag increase, $C_d$ +0.07)**: Optimizer 3.1 m/s → ML 1.8 m/s. ML infers higher drag via EKF; ignites lower (less deceleration needed due to higher drag).

**Demo 6 (Severe fault: wind + drag + mass)**: Optimizer 4.1 m/s → ML 2.2 m/s. Complex fault combination; ML applies composite correction.

**Demo 7 (Severe fault: wind + drag + mass + low thrust)**: Optimizer CRASH (12.8 m/s) → ML 4.9 m/s (degraded but survivable). ML detects the extreme scenario and applies maximum correction to mitigate; optimizer's baseline is insufficient.

**Demo 9 (Extreme: wind 12 m/s + $C_d$ +0.10 + mass $-$2% + thrust $-$5%)**: Optimizer 4.1 m/s → ML 4.9 m/s (+0.8 m/s worse). Possible explanations:
1. Extreme fault combination is outside the training distribution; model extrapolates poorly
2. Regression toward mean: model learned to output large corrections for "severe" faults, but this extreme case required unusual precision
3. Antagonistic fault combination (e.g., high wind but lower mass, competing effects) confused the model

This motivates future work: expand training distribution to cover more extreme scenarios.

### 11.3 Why Doesn't ML Always Win?

In demo 9, ML actually makes things slightly worse (0.8 m/s penalty). Why?

1. **Out-of-distribution extrapolation**: The training set's most extreme scenarios are severe but not as extreme as demo 9. The model was not shown enough examples of this particular fault combination. Neural networks extrapolate poorly outside their training range — they were not designed for it.

2. **Regression toward mean**: Neural networks learn to output the statistically typical answer for their input region. For "severe fault" conditions, the typical answer is "apply a large correction." But demo 9 is so extreme that it needs a different strategy.

3. **Nonlinear interactions**: Hoverslam physics involves many nonlinear interactions. The model may have learned to respond to, say, high wind, but demo 9 combines high wind with a specific mass/thrust/drag combination that wasn't well-represented in training.

**Lesson**: ML is powerful but not magic. It requires good training data coverage. Extreme scenarios beyond the training distribution are still risky — hence the ±10 m clamp and fallback to the optimizer.

---

## 12. Future ML Development

Several directions could improve the ML model:

| Improvement | Description | Expected Impact |
|-------------|-------------|-----------------|
| **Expand training distribution** | Include more extreme scenarios; push Monte Carlo to ±10% parameter variations instead of ±5% | Better out-of-distribution performance; reduce demo 9 penalty |
| **Uncertainty quantification** | Bayesian neural network or ensemble model; output predicted correction + confidence interval | Know when to trust ML vs. fallback |
| **Transfer learning** | Train on simulation; fine-tune with real flight data when available | Massive accuracy jump; close sim-to-real gap |
| **Sequence modeling (LSTM)** | Instead of snapshot (current state only), input full recent descent trajectory; model temporal dynamics | Better anticipation of trajectory evolution |
| **Curriculum learning** | Train on easy scenarios (nominal), progressively add harder ones | More stable training; better extreme-case performance |
| **Real-time adaptation** | Online learning: update weights slightly during descent based on EKF observations | Ultimate adaptability (very advanced) |
| **Ensemble model** | Train multiple neural networks on slightly different data splits; average predictions | Reduce overfitting; improve robustness |

Most promising near-term: **expand training distribution** and **add uncertainty quantification**. Together, they'd give the flight computer confidence intervals, allowing smart fallback to the optimizer when the ML model is uncertain.

---

## 13. Summary and Key Hyperparameters

### Model and Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Architecture** | Dense: 25 → 128 → 64 → 32 → 1 | 3 hidden layers; moderate size for Teensy |
| **Activation (hidden)** | ReLU | Fast; avoids vanishing gradient |
| **Activation (output)** | Linear | Regression; unrestricted output |
| **Total parameters** | 13,697 | Compact; < 2 KB RAM at inference |
| **Loss function** | Mean Squared Error | Standard for regression |
| **Optimizer** | Adam | Adaptive; converges reliably |
| **Batch size** | 128–256 | Balances speed and noise |
| **Learning rate** | 1.0e-3 initial | Adam adapts per-parameter |
| **Early stopping** | 10 epochs no improvement | Prevents overfitting |
| **Training data** | ~240,000 examples | 20,000 MC sims $\times$ 12 snapshots each |
| **Feature count** | 25 | Balances expressiveness and complexity |
| **Feature normalization** | Z-score (mean 0, std 1) | Prevents magnitude dominance |
| **Validation MSE** | ~0.49 m² | ±0.7 m typical prediction error |
| **Deployment** | TensorFlow Lite, INT8 quantization | 55 KB model; fast inference on ARM |

### Operational Integration

| System | Interface | Latency |
|--------|-----------|---------|
| Monte Carlo Optimizer | Provides baseline ignition altitude | Pre-flight (offline) |
| Extended Kalman Filter | Provides estimated state (25 features) | Every 500 ms |
| Flight Control System | Receives updated ignition altitude | Every 500 ms |
| Teensy 4.1 | Runs TFLite inference | 50–100 ms per cycle |

---

## References and See Also

- **Core Simulation Engine**: `/04_ML_Model/../Simulation_Engine.md` — 6DOF physics, EKF, aerodynamics
- **Monte Carlo Optimizer**: `/04_ML_Model/../Optimizer_Design.md` — pre-flight baseline computation
- **System Architecture**: fig_01_system_architecture.png — overall HERMES architecture
- **Comparison Results**: fig_11_ml_vs_optimizer.png, fig_05_demo_scenario_comparison.png — performance across scenarios
- **Extended Kalman Filter**: State estimation details used to populate features 16–22

---

**Document Version**: 1.0
**Last Updated**: 2026-03-12
**Status**: Engineering Notebook — Science Fair Documentation
