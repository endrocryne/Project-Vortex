# Normalized Fault Intensity — Formula Reference

Document for `calculate_fault_intensity()` in [faults.py](../faults.py).

---

## Why something fancier than multiplication?

Simple linear ramps and flat multiplications lose discrimination at the extremes — anything severe enough saturates to the same value and can no longer be ranked.  The three mathematical tools used here (Hill functions, exponential urgency integrals, and exponential saturation curves) each come from a physical first-principle and collectively preserve full dynamic range across every parameter.

---

## Final Formula

$$\boxed{I = P \cdot B \cdot C}$$

Substituting definitions gives a single expression:

$$
I = P\,B\left[ C_0 + (1-C_0)\left(w_tT_n + w_dD_n + w_cT_nD_n\right)\right]
= P\,B\left[ C_0 + (1-C_0)\Bigl(w_t\frac{e^{\alpha\tau}-1}{e^{\alpha}-1} + w_d(1-e^{-\lambda\delta}) + w_c\frac{e^{\alpha\tau}-1}{e^{\alpha}-1}(1-e^{-\lambda\delta})\Bigr)\right]
$$

where $\tau=t_{\text{trigger}}/T_{\text{descent}}$ and $\delta=\text{duration}/T_{\text{descent}}$.

| Symbol | Domain | Meaning |
|--------|--------|---------|
| $P$ | $[0, 1]$ | Probability the fault occurs |
| $B$ | $[0, 1)$ | Base severity — how badly the fault degrades the vehicle |
| $C$ | $[C_0, 1]$ | Context modifier — amplifies $B$ based on *when* and *how long* |

---

## Component 1 — Base intensity $B$ : Hill / cooperative sigmoid

Each fault type uses a **Hill function** (also called the cooperative Michaelis-Menten equation, originating in enzyme kinetics and receptor-occupancy theory):

$$B(x;\, K,\, n) = \frac{x^n}{x^n + K^n}$$

| Property | Behaviour |
|----------|-----------|
| $B(0) = 0$ | Zero-magnitude fault has zero impact |
| $B(K) = 0.5$ | $K$ is the **half-saturation point** (input that yields 50% severity) |
| $B \to 1$ as $x \to \infty$ | Asymptotically saturates at full severity |
| $n > 1$ | Sigmoidal / "cooperative" — slow onset below $K$, steep rise around $K$, plateau above |
| $n < 1$ | Concave — rapid initial rise, then diminishing returns |

Using a Hill function rather than a linear ramp or hard clip captures the real physical non-linearity: a 5% thrust reduction barely matters, a 20% reduction is a serious concern, and a 60% reduction is effectively catastrophic — an inherently S-shaped response.

### Per-fault-type parameters

| Fault | Input $x$ | Half-sat $K$ | Cooperativity $n$ | Notes |
|-------|-----------|---|---|---|
| Mass loss | $m_{\text{lost}} / m_{\text{ref}}$ | 0.15 | 1.4 | S-curve; ~15% mass loss is the inflection |
| Thrust reduction | $\delta = 1 - T_{\text{mult}}$ | 0.20 | 2.0 | Strong cooperative response; steepest around 20% loss |
| Thrust increase | $\delta = T_{\text{mult}} - 1$ | 0.25 | 0.7 | Concave — capped at 0.4 (controllable concern) |
| Drag decrease | $\delta = 1 - C_{D,\text{mult}}$ | 0.25 | 1.8 | Cooperative; reduced drag is critical for propulsive landing |
| Drag increase | $\delta = C_{D,\text{mult}} - 1$ | 0.30 | 0.8 | Concave — capped at 0.6 |
| Wind gust | $v / 15\text{ m/s}$ | 0.40 | 0.75 | Concave; first m/s of gust has steepest proportional effect |

---

## Component 2 — Timing criticality $T_n$ : exponential urgency integral

The correction authority available to the guidance system at time $t$ equals the remaining corrective impulse:

$$J_{\text{remaining}}(t) = \int_t^{T_{\text{land}}} F_{\max}\, d\tilde{t}$$

As $t \to T_{\text{land}}$, this integral shrinks to zero and the *marginal* danger of firing one second later grows at an exponential rate.  Integrating an urgency function $u(\tau) \propto e^{\alpha\tau}$ over $[0, \tau]$ and normalising yields:

$$T_n(\tau) = \frac{e^{\alpha\tau} - 1}{e^{\alpha} - 1}, \qquad \tau = \frac{t_{\text{trigger}}}{T_{\text{descent}}} \in [0,1]$$

with $\alpha = 3.0$, fitted so that a fault at 80% of descent is roughly 3× more critical than one at 40%.

| Descent elapsed $\tau$ | Linear model | $T_n$ (exponential, $\alpha=3$) |
|---|---|---|
| 0 % | 0.00 | 0.000 |
| 20 % | 0.20 | 0.049 |
| 40 % | 0.40 | 0.152 |
| 60 % | 0.60 | 0.348 |
| 80 % | 0.80 | 0.627 |
| 100 % | 1.00 | 1.000 |

The last 20% of descent is disproportionately dangerous — a fact the exponential captures and a linear model fundamentally cannot.

---

## Component 3 — Duration severity $D_n$ : exponential saturation

Integrating a constant thrust deficit $\Delta F$ over fault duration $\Delta t$ yields a velocity error $\Delta v = \Delta F \cdot \Delta t / m$.  However, each additional second becomes marginally less damaging once the trajectory is already badly off-nominal — the guidance system's residual can grow no worse than some catastrophic bound.  This diminishing return is modelled by exponential saturation, the natural integral of a decaying response:

$$D_n(\tau) = 1 - e^{-\lambda\tau}, \qquad \tau = \frac{\Delta t_{\text{fault}}}{T_{\text{descent}}}$$

with $\lambda = 3.0$.  A permanent fault ($\Delta t \to \infty$) gives $D_n = 1$ exactly.

| Duration as fraction of descent | Linear model | $D_n$ (exp. saturation, $\lambda=3$) |
|---|---|---|
| 0 % | 0.00 | 0.000 |
| 10 % (1 s in 10 s) | 0.10 | 0.259 |
| 25 % | 0.25 | 0.528 |
| 50 % | 0.50 | 0.777 |
| 100 % | 1.00 | 0.950 |
| Permanent ($\infty$) | — | 1.000 |

---

## Component 4 — Context modifier $C$ : bilinear interaction

Rather than adding timing and duration independently, the context modifier includes a **bilinear interaction term** $T_n \cdot D_n$ that captures the super-additive danger of a fault that is *simultaneously* late and persistent:

$$C = C_0 + (1 - C_0)\,\bigl(w_t\, T_n + w_d\, D_n + w_c\, T_n D_n\bigr)$$

| Constant | Value | Role |
|----------|-------|------|
| $C_0$ | 0.25 | Floor — even best-timed, briefest fault still registers at 25% |
| $w_t$ | 0.35 | Timing's share of the variable context range |
| $w_d$ | 0.35 | Duration's share |
| $w_c$ | 0.30 | Bilinear interaction weight ($w_t + w_d + w_c = 1$) |

The bilinear term $w_c\,T_n D_n$ is the key innovation over a simple weighted average.  Without it, a fault that is very late *or* very long gets nearly full context amplification regardless of the other factor.  With it, the maximum amplification only fully activates when **both** are high — the physically correct combination where there is neither time nor control authority for recovery.

**Range:** At $T_n = D_n = 1$: $C = 0.25 + 0.75(0.35 + 0.35 + 0.30) = 1.0$. At $T_n = D_n = 0$: $C = 0.25$.

---

## Boundary properties

| Property | Status | Reason |
|----------|--------|--------|
| $I \in [0, 1]$ | Guaranteed | Every factor is in $[0, 1]$, no clamping needed |
| $B = 0 \Rightarrow I = 0$ | Guaranteed | $B$ gates everything |
| $P = 0 \Rightarrow I = 0$ | Guaranteed | $P$ gates everything |
| Monotonically non-decreasing | Guaranteed | Every sub-function is monotone increasing in its input |
| $\sup(I) = 1$ (limit only) | Guaranteed | The Hill function is asymptotic: $B \to 1$ as magnitude $\to \infty$ and $T_n \to 1$ only at exactly $\tau = 1$ — so $I = 1$ is never achieved at finite inputs.  This is physically correct: no finite fault guarantees failure with probability exactly 1. |

---

## Multi-fault combination

Individual intensities $I_i$ are combined using **probabilistic OR** (inclusion–exclusion):

$$I_{\text{combined}} = 1 - \prod_i (1 - I_i)$$

This gives sub-linear accumulation (two 0.3-intensity faults yield ~0.51, not 0.6), stays in $[0, 1]$, and is commutative.

---

## Worked examples

Using defaults: $T_{\text{descent}} = 10\text{ s}$, $h_{\text{ref}} = 1000\text{ m}$, $m_{\text{ref}} = 60\text{ kg}$.

| Scenario | $B$ | $T_n$ | $D_n$ | $C$ | $I$ |
|---|---|---|---|---|---|
| Mass 5 kg, t+2 s after apogee, permanent | 0.19 | 0.027 | 1.00 | 0.276 | 0.053 |
| Mass 5 kg, t+10 s, permanent | 0.19 | 1.000 | 1.00 | 1.000 | 0.190 |
| Thrust 80%, alt 500 m, 2 s | 0.20 | 0.50 | 0.26 | 0.511 | 0.102 |
| Thrust 40%, alt 200 m, 5 s | 0.80 | 0.80 | 0.78 | 0.878 | 0.702 |
| Total thrust loss, alt 10 m, permanent | 0.96 | 0.99 | 1.00 | 0.985 | 0.945 |
| Wind 2 m/s, t+0.5 s, 1 s | 0.43 | 0.003 | 0.09 | 0.280 | 0.121 |
| Wind 15 m/s, ground level, permanent | 0.67 | 1.000 | 1.00 | 1.000 | 0.665 |
| Thrust nominal (no fault) | 0.00 | — | — | — | 0.000 |
| Mass 20 kg but $P = 0$ | 1.00 | — | — | — | 0.000 |
