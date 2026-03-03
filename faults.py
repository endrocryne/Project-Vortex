"""
Fault Injection System for Suicide Burn Simulation
Supports dynamic fault injection during simulation runtime
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import copy


class FaultType(Enum):
    """Enumeration of supported fault types"""
    MASS_LOSS = "mass_loss"
    THRUST_VAR = "thrust_var"
    DRAG_CHANGE = "drag_change"
    WIND_GUST = "wind_gust"


class TriggerMode(Enum):
    """Fault trigger modes"""
    ABSOLUTE_TIME = "absolute_time"
    TIME_SINCE_APOGEE = "time_since_apogee"
    ALTITUDE_THRESHOLD = "altitude_threshold"
    MANUAL = "manual"


class FaultTarget(Enum):
    """Where the fault affects"""
    SIMULATED_STATE = "simulated_state"  # Affects actual dynamics
    SENSOR_ONLY = "sensor_only"  # Only affects sensor readings


@dataclass
class FaultConfig:
    """Configuration for a single fault"""
    fault_type: FaultType
    trigger_mode: TriggerMode
    trigger_value: float  # time (s) or altitude (m) depending on mode
    magnitude: float  # Multiplier or absolute value depending on fault type
    duration: float  # Duration in seconds (0 = permanent)
    target: FaultTarget = FaultTarget.SIMULATED_STATE
    randomize_magnitude: bool = False
    magnitude_range: tuple = (0.8, 1.2)  # Min/max for randomization
    probability: float = 1.0  # Probability of occurring (0-1)
    active: bool = False  # Runtime flag
    start_time: float = 0.0  # When fault was activated
    applied_magnitude: float = 0.0  # Actual magnitude after randomization
    fault_id: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'fault_type': self.fault_type.value,
            'trigger_mode': self.trigger_mode.value,
            'trigger_value': self.trigger_value,
            'magnitude': self.magnitude,
            'duration': self.duration,
            'target': self.target.value,
            'randomize_magnitude': self.randomize_magnitude,
            'magnitude_range': list(self.magnitude_range),
            'probability': self.probability,
            'fault_id': self.fault_id
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'FaultConfig':
        """Create from dictionary"""
        return FaultConfig(
            fault_type=FaultType(d['fault_type']),
            trigger_mode=TriggerMode(d['trigger_mode']),
            trigger_value=d['trigger_value'],
            magnitude=d['magnitude'],
            duration=d['duration'],
            target=FaultTarget(d.get('target', 'simulated_state')),
            randomize_magnitude=d.get('randomize_magnitude', False),
            magnitude_range=tuple(d.get('magnitude_range', [0.8, 1.2])),
            probability=d.get('probability', 1.0),
            fault_id=d.get('fault_id', 0)
        )


@dataclass
class FaultGroup:
    """Group of faults that execute concurrently or sequentially"""
    faults: List[FaultConfig] = field(default_factory=list)
    concurrent: bool = True  # If True, all faults trigger together
    group_id: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'faults': [f.to_dict() for f in self.faults],
            'concurrent': self.concurrent,
            'group_id': self.group_id
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'FaultGroup':
        """Create from dictionary"""
        return FaultGroup(
            faults=[FaultConfig.from_dict(f) for f in d['faults']],
            concurrent=d.get('concurrent', True),
            group_id=d.get('group_id', 0)
        )


class FaultInjectionManager:
    """
    Manages fault injection during simulation
    Handles triggering, application, and deactivation of faults
    """
    
    def __init__(self, fault_groups: Optional[List[FaultGroup]] = None):
        """
        Initialize fault manager
        
        Args:
            fault_groups: List of fault groups to manage
        """
        self.fault_groups = fault_groups or []
        self.all_faults: List[FaultConfig] = []
        self.active_faults: List[FaultConfig] = []
        self.fault_history: List[Dict[str, Any]] = []
        self.apogee_time: Optional[float] = None
        self.apogee_detected: bool = False
        self.has_ascended: bool = False
        
        # Flatten all faults for easy access
        self._flatten_faults()
        
        # State modifications (accumulated)
        self.mass_delta = 0.0
        self.thrust_multiplier = 1.0
        self.drag_multiplier = 1.0
        self.wind_speed_delta = 0.0
        self.wind_direction_delta = 0.0
        
    def _flatten_faults(self):
        """Flatten all faults from groups into single list"""
        self.all_faults = []
        for group in self.fault_groups:
            self.all_faults.extend(group.faults)
    
    def add_fault_group(self, group: FaultGroup):
        """Add a fault group"""
        self.fault_groups.append(group)
        self._flatten_faults()
    
    def clear_faults(self):
        """Clear all faults and reset state"""
        self.fault_groups = []
        self.all_faults = []
        self.active_faults = []
        self.fault_history = []
        self.reset()
        
    def reset(self):
        """Reset apogee detection state"""
        self.apogee_time = None
        self.apogee_detected = False
        self.has_ascended = False
        self.mass_delta = 0.0
        self.thrust_multiplier = 1.0
        self.drag_multiplier = 1.0
        self.wind_speed_delta = 0.0
        self.wind_direction_delta = 0.0
        # Re-flatten to ensure any group modifications are picked up
        self._flatten_faults()
    
    def detect_apogee(self, current_time: float, vz: float, altitude: float):
        """
        Detect apogee based on vertical velocity and flight history
        
        Args:
            current_time: Current simulation time
            vz: Vertical velocity
            altitude: Current altitude
        """
        if not self.apogee_detected:
            # Track if we have started ascending
            # Require both velocity and a bit of altitude gain to be sure
            if vz > 2.0 and altitude > 10.0:
                self.has_ascended = True
            
            # Detect apogee if:
            # 1. We were ascending and now we are not (vz <= 0)
            # OR
            # 2. We are starting high up (initial condition) and not ascending
            if vz <= 0:
                # Extra check: avoid triggering at start if on pad
                if self.has_ascended:
                    self.apogee_time = current_time
                    self.apogee_detected = True
                elif current_time < 0.1 and altitude > 10.0:
                    # Case: starting already at altitude (descent-only sim)
                    self.apogee_time = current_time
                    self.apogee_detected = True
    
    def check_triggers(self, current_time: float, altitude: float, vz: float) -> List[FaultConfig]:
        """
        Check if any faults should be triggered
        
        Args:
            current_time: Current simulation time (s)
            altitude: Current altitude (m)
            vz: Vertical velocity (for apogee detection)
            
        Returns:
            List of newly triggered faults
        """
        # Detect apogee
        self.detect_apogee(current_time, vz, altitude)
        
        newly_triggered = []
        
        for fault in self.all_faults:
            if fault.active:
                continue  # Already active
            
            triggered = False
            
            # Check trigger condition
            if fault.trigger_mode == TriggerMode.ABSOLUTE_TIME:
                triggered = current_time >= fault.trigger_value
            
            elif fault.trigger_mode == TriggerMode.TIME_SINCE_APOGEE:
                if self.apogee_detected and self.apogee_time is not None:
                    time_since_apogee = current_time - self.apogee_time
                    triggered = time_since_apogee >= fault.trigger_value
            
            elif fault.trigger_mode == TriggerMode.ALTITUDE_THRESHOLD:
                triggered = altitude <= fault.trigger_value
            
            elif fault.trigger_mode == TriggerMode.MANUAL:
                # Manual faults don't auto-trigger
                triggered = False
            
            if triggered:
                # Check probability
                if np.random.random() <= fault.probability:
                    self._activate_fault(fault, current_time)
                    newly_triggered.append(fault)
        
        return newly_triggered
    
    def _activate_fault(self, fault: FaultConfig, current_time: float):
        """Activate a fault"""
        fault.active = True
        fault.start_time = current_time
        
        # Randomize magnitude if requested
        if fault.randomize_magnitude:
            fault.applied_magnitude = np.random.uniform(
                fault.magnitude_range[0], 
                fault.magnitude_range[1]
            )
        else:
            fault.applied_magnitude = fault.magnitude
        
        self.active_faults.append(fault)
        
        # Log activation
        self.fault_history.append({
            'time': current_time,
            'action': 'activate',
            'fault_type': fault.fault_type.value,
            'magnitude': fault.applied_magnitude,
            'duration': fault.duration,
            'fault_id': fault.fault_id
        })
    
    def manually_trigger_fault(self, fault_id: int, current_time: float):
        """Manually trigger a specific fault by ID"""
        for fault in self.all_faults:
            if fault.fault_id == fault_id and not fault.active:
                self._activate_fault(fault, current_time)
                break
    
    def check_deactivations(self, current_time: float) -> List[FaultConfig]:
        """
        Check if any active faults should be deactivated
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of deactivated faults
        """
        deactivated = []
        
        for fault in self.active_faults[:]:  # Copy list to allow removal
            if fault.duration > 0:  # 0 means permanent
                elapsed = current_time - fault.start_time
                if elapsed >= fault.duration:
                    self._deactivate_fault(fault, current_time)
                    deactivated.append(fault)
        
        return deactivated
    
    def _deactivate_fault(self, fault: FaultConfig, current_time: float):
        """Deactivate a fault"""
        fault.active = False
        self.active_faults.remove(fault)
        
        # Log deactivation
        self.fault_history.append({
            'time': current_time,
            'action': 'deactivate',
            'fault_type': fault.fault_type.value,
            'fault_id': fault.fault_id
        })
    
    def apply_faults_to_state(self, state: np.ndarray, current_time: float) -> np.ndarray:
        """
        Apply active faults to simulation state
        
        Args:
            state: Current state vector [x,y,z,vx,vy,vz,qw,qx,qy,qz,ωx,ωy,ωz,mass]
            current_time: Current simulation time
            
        Returns:
            Modified state vector
        """
        modified_state = state.copy()
        
        # Reset accumulated modifications
        self.mass_delta = 0.0
        
        for fault in self.active_faults:
            if fault.target != FaultTarget.SIMULATED_STATE:
                continue
            
            if fault.fault_type == FaultType.MASS_LOSS:
                # Apply mass loss (negative magnitude means loss)
                self.mass_delta += fault.applied_magnitude
                modified_state[13] = max(1.0, modified_state[13] + fault.applied_magnitude)
        
        return modified_state
    
    def get_thrust_multiplier(self) -> float:
        """Get current thrust multiplier from active faults"""
        multiplier = 1.0
        for fault in self.active_faults:
            if fault.fault_type == FaultType.THRUST_VAR:
                if fault.target == FaultTarget.SIMULATED_STATE:
                    multiplier *= fault.applied_magnitude
        return multiplier
    
    def get_drag_multiplier(self) -> float:
        """Get current drag multiplier from active faults"""
        multiplier = 1.0
        for fault in self.active_faults:
            if fault.fault_type == FaultType.DRAG_CHANGE:
                if fault.target == FaultTarget.SIMULATED_STATE:
                    multiplier *= fault.applied_magnitude
        return multiplier
    
    def get_wind_modification(self) -> tuple:
        """
        Get wind modifications from active faults
        
        Returns:
            (speed_delta, direction_delta) in m/s and degrees
        """
        speed_delta = 0.0
        direction_delta = 0.0
        
        for fault in self.active_faults:
            if fault.fault_type == FaultType.WIND_GUST:
                if fault.target == FaultTarget.SIMULATED_STATE:
                    # Magnitude represents wind speed change
                    speed_delta += fault.applied_magnitude
        
        return speed_delta, direction_delta
    
    def apply_sensor_noise(self, sensor_readings: Dict[str, float]) -> Dict[str, float]:
        """
        Apply sensor-only faults to sensor readings
        
        Args:
            sensor_readings: Dictionary of sensor values
            
        Returns:
            Modified sensor readings
        """
        modified = sensor_readings.copy()
        
        for fault in self.active_faults:
            if fault.target != FaultTarget.SENSOR_ONLY:
                continue
            
            # Apply sensor-specific modifications
            if fault.fault_type == FaultType.MASS_LOSS:
                # Would affect inferred mass from accelerometer
                if 'inferred_mass' in modified:
                    modified['inferred_mass'] *= fault.applied_magnitude
            
            # Add more sensor-specific logic as needed
        
        return modified
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get current fault injection status"""
        return {
            'total_faults': len(self.all_faults),
            'active_faults': len(self.active_faults),
            'apogee_detected': self.apogee_detected,
            'apogee_time': self.apogee_time,
            'active_fault_types': [f.fault_type.value for f in self.active_faults],
            'fault_history_count': len(self.fault_history)
        }
    
    def get_fault_log(self) -> List[Dict[str, Any]]:
        """Get complete fault history log"""
        return self.fault_history.copy()
    
    def to_config_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return {
            'fault_groups': [g.to_dict() for g in self.fault_groups]
        }
    
    @staticmethod
    def from_config_dict(d: Dict[str, Any]) -> 'FaultInjectionManager':
        """Create manager from configuration dictionary"""
        groups = [FaultGroup.from_dict(g) for g in d.get('fault_groups', [])]
        return FaultInjectionManager(groups)


def _hill(x: float, K: float, n: float) -> float:
    """
    Hill / Michaelis-Menten function: f(x; K, n) = x^n / (x^n + K^n)
    
    Properties:
    - f(0) = 0,  f(K) = 0.5,  f(∞) → 1
    - n < 1 : concave (diminishing returns)
    - n = 1 : Michaelis-Menten (hyperbolic)
    - n > 1 : sigmoidal / cooperative (slow start, then rapid rise, then plateau)
    
    K is the half-saturation point (input yielding output = 0.5).
    """
    if x <= 0.0:
        return 0.0
    xn = x ** n
    return xn / (xn + K ** n)


def _exp_timing(frac: float, alpha: float) -> float:
    """
    Exponentially growing criticality on [0, 1] → [0, 1].
    
    T_n(τ) = (e^(α·τ) - 1) / (e^α - 1)
    
    Derived from integrating an urgency function u(τ) ∝ e^(α·τ) over [0, τ],
    then normalizing so T_n(0) = 0 and T_n(1) = 1.
    
    Physical meaning: the recovery margin available to the control system shrinks
    exponentially as the vehicle approaches touchdown. A fault that fires at 90%
    of the descent is far more dangerous than its position in time suggests
    linearly, because very little corrective impulse (∫F dt) remains.
    """
    if alpha == 0.0:
        return frac
    ea = np.exp(alpha)
    return (np.exp(alpha * frac) - 1.0) / (ea - 1.0)


def _exp_saturation(frac: float, lam: float) -> float:
    """
    Exponential saturation on [0, ∞) → [0, 1).
    
    D_n(τ) = 1 - e^(-λ·τ)
    
    Models the accumulated velocity impulse deficit inflicted by a thrust or drag
    fault of relative duration τ = dur / T_descent.  Integrating a constant thrust
    deficit ΔF over time τ yields Δv = ΔF·τ/m, but the marginal severity of each
    additional second diminishes once the vehicle's trajectory is already badly
    perturbed — hence the exponential saturation captures this physics better than
    a linear ramp.
    
    A permanent fault (τ → ∞) naturally gives D_n → 1.
    """
    return 1.0 - np.exp(-lam * frac)


def calculate_fault_intensity(fault: FaultConfig,
                              typical_descent_time: float = 10.0,
                              typical_altitude: float = 1000.0,
                              reference_mass: float = 60.0) -> float:
    """
    Calculate normalized fault intensity I ∈ [0, 1] representing the expected
    impact of a fault on landing safety.

    ═══════════════════════════════════════════════════════════════════════
    FINAL FORMULA
    ═══════════════════════════════════════════════════════════════════════

        I = P · B · C

    where:

        B  = base intensity from fault magnitude            ∈ [0, 1]
        C  = context modifier (timing × duration coupling)  ∈ [C₀, 1]
        P  = probability of fault occurring                 ∈ [0, 1]

    ───────────────────────────────────────────────────────────────────────
    BASE INTENSITY  B  — Hill / cooperative-sigmoid functions
    ───────────────────────────────────────────────────────────────────────
    Each fault type uses a Hill function with tuned half-saturation K and
    cooperativity exponent n:

        B_hill(x; K, n) = x^n / (x^n + K^n)

    n > 1 produces an S-curve (slow onset, rapid rise, plateau); n < 1
    gives concave diminishing-return behaviour.  This is the same form used
    in enzyme kinetics and receptor-occupancy models.

    ───────────────────────────────────────────────────────────────────────
    TIMING CRITICALITY  T_n  — exponential urgency integral
    ───────────────────────────────────────────────────────────────────────
    The control system's ability to correct a fault is proportional to the
    remaining corrective impulse ∫_{t}^{T} F_max dt.  As t → T this margin
    shrinks, and the marginal danger of firing one second later grows
    exponentially.  Normalizing the integrated urgency function gives:

        T_n(τ) = (e^{ατ} − 1) / (e^α − 1),   τ = t_trigger / T_descent

    with α = 3.0 (fitted so that a fault at 80% descent is ~3× more
    critical than a fault at 40% descent in normalised units).

    ───────────────────────────────────────────────────────────────────────
    DURATION SEVERITY  D_n  — accumulated impulse deficit (exponential sat.)
    ───────────────────────────────────────────────────────────────────────
    Integrating a constant thrust deficit ΔF over fault duration τ yields a
    velocity error Δv = ΔF·τ/m.  The marginal severity of each additional
    second diminishes once the trajectory is already badly perturbed, making
    exponential saturation the physically correct model:

        D_n(τ) = 1 − e^{−λτ},   τ = dur / T_descent,   λ = 3.0

    A permanent fault (dur = 0 ⟹ τ → ∞) gives D_n = 1 exactly.

    ───────────────────────────────────────────────────────────────────────
    CONTEXT MODIFIER  C  — bilinear interaction (Cobb-Douglas extended)
    ───────────────────────────────────────────────────────────────────────
        C = C₀ + (1 − C₀) · (w_t·T_n + w_d·D_n + w_c·T_n·D_n)

    The bilinear interaction term w_c·T_n·D_n captures the super-additive
    severity of a fault that is BOTH late AND persistent — a situation where
    there is neither time nor opportunity for recovery.  Without this term
    the model would treat late+long faults as merely additive rather than
    multiplicatively dangerous.

    Constants:  C₀ = 0.25,  w_t = 0.35,  w_d = 0.35,  w_c = 0.30
    (weights sum to 1.0; the interaction term borrows equally from each).

    Maximum at T_n = D_n = 1:  C = 0.25 + 0.75·(0.35 + 0.35 + 0.30) = 1.0
    Minimum at T_n = D_n = 0:  C = 0.25

    ───────────────────────────────────────────────────────────────────────
    BOUNDARY PROPERTIES (all guaranteed by construction)
    ───────────────────────────────────────────────────────────────────────
    • I ∈ [0, 1]  — no clamping required
    • B = 0  ⟹  I = 0  (zero-magnitude fault has zero impact)
    • P = 0  ⟹  I = 0  (impossible fault contributes nothing)
    • sup(I) = 1  — the supremum is 1, approached as P→1, |magnitude|→∞,
                     timing→ touchdown, and duration→∞.  It is never exactly
                     reached at finite inputs because the Hill function is
                     asymptotic.  This is physically correct: even a catastrophic
                     fault does not guarantee failure with probability exactly 1.
    • Monotonically non-decreasing in P, |magnitude|, t_trigger, duration

    Args:
        fault: FaultConfig object
        typical_descent_time: Expected descent duration in seconds
        typical_altitude: Expected starting altitude in metres
        reference_mass: Reference vehicle mass in kg

    Returns:
        Normalized intensity value I ∈ [0, 1]
    """

    # ── Shape constants ───────────────────────────────────────────────────
    C_FLOOR  = 0.25   # context floor (even best-timed, briefest fault registers)
    W_T      = 0.35   # timing weight in context
    W_D      = 0.35   # duration weight in context
    W_C      = 0.30   # bilinear interaction weight
    ALPHA    = 3.0    # exponential steepness of timing criticality
    LAMBDA   = 3.0    # exponential saturation rate for duration

    # ════════════════════════════════════════════════════════════════════
    # STEP 1 — Base intensity B using Hill functions
    # ════════════════════════════════════════════════════════════════════
    base_intensity = 0.0

    if fault.fault_type == FaultType.MASS_LOSS:
        # Hill(x; K=0.15, n=1.4): half-saturation at 15% mass loss.
        # n > 1 gives cooperative (S-curve) onset — small losses matter less,
        # losses above ~15% of reference mass grow rapidly in severity.
        mass_frac = abs(fault.magnitude) / reference_mass
        base_intensity = _hill(mass_frac, K=0.15, n=1.4)

    elif fault.fault_type == FaultType.THRUST_VAR:
        thrust_mult = fault.magnitude
        if thrust_mult < 1.0:
            # Thrust REDUCTION — highly critical for suicide burn.
            # Hill(δ; K=0.20, n=2.0): strong cooperative response.
            # At δ=0.1 (10% loss): B ≈ 0.20
            # At δ=0.2 (20% loss): B ≈ 0.50
            # At δ=0.4 (40% loss): B ≈ 0.80
            # At δ=0.6 (60% loss): B ≈ 0.90
            deviation = 1.0 - thrust_mult
            base_intensity = _hill(deviation, K=0.20, n=2.0)
        else:
            # Thrust INCREASE — minor concern (fuel waste, control load).
            # Concave Hill (n < 1): sharp onset, quickly diminishing returns.
            deviation = thrust_mult - 1.0
            base_intensity = min(0.4, _hill(deviation, K=0.25, n=0.7))

    elif fault.fault_type == FaultType.DRAG_CHANGE:
        drag_mult = fault.magnitude
        if drag_mult < 1.0:
            # Drag DECREASE — vehicle cannot slow down; critical for propulsive landing.
            # Hill(δ; K=0.25, n=1.8): cooperative, saturates around 60% drag loss.
            deviation = 1.0 - drag_mult
            base_intensity = _hill(deviation, K=0.25, n=1.8)
        else:
            # Drag INCREASE — excessive deceleration; less severe.
            # Concave Hill: significant at small increases, flattens quickly.
            deviation = drag_mult - 1.0
            base_intensity = min(0.6, _hill(deviation, K=0.30, n=0.8))

    elif fault.fault_type == FaultType.WIND_GUST:
        # Wind force on the vehicle scales as v² (aerodynamic drag), but
        # pilot/controller difficulty is closer to v^1.5.  We use
        # Hill(v / v_ref; K=0.4, n=0.75) which gives sub-linear concave
        # behaviour: the first few m/s impose the steepest fractional difficulty.
        # Half-saturation at v = 0.4 * 15 = 6 m/s.
        wind_norm = abs(fault.magnitude) / 15.0   # 15 m/s = reference max gust
        base_intensity = min(1.0, _hill(wind_norm, K=0.40, n=0.75))

    # ════════════════════════════════════════════════════════════════════
    # STEP 2 — Timing criticality T_n via exponential urgency integral
    # ════════════════════════════════════════════════════════════════════
    # Raw timing fraction τ ∈ [0, 1]:  0 = top of descent, 1 = touchdown
    timing_frac = 0.6  # default: slightly late (conservative for MANUAL)

    if fault.trigger_mode == TriggerMode.ABSOLUTE_TIME:
        timing_frac = min(1.0, fault.trigger_value / typical_descent_time)

    elif fault.trigger_mode == TriggerMode.TIME_SINCE_APOGEE:
        timing_frac = min(1.0, fault.trigger_value / typical_descent_time)

    elif fault.trigger_mode == TriggerMode.ALTITUDE_THRESHOLD:
        # Lower altitude → higher timing fraction (closer to touchdown)
        alt_frac = min(1.0, fault.trigger_value / typical_altitude)
        timing_frac = 1.0 - alt_frac

    elif fault.trigger_mode == TriggerMode.MANUAL:
        timing_frac = 0.75   # assume mid-to-late descent conservatively

    # Apply exponential criticality growth
    T_n = _exp_timing(timing_frac, ALPHA)

    # ════════════════════════════════════════════════════════════════════
    # STEP 3 — Duration severity D_n via exponential saturation
    # ════════════════════════════════════════════════════════════════════
    if fault.duration == 0.0:
        # Permanent fault: τ → ∞, so 1 - e^{-λ·∞} = 1
        D_n = 1.0
    else:
        dur_frac = fault.duration / typical_descent_time
        D_n = _exp_saturation(dur_frac, LAMBDA)

    # ════════════════════════════════════════════════════════════════════
    # STEP 4 — Context modifier C with bilinear interaction term
    # ════════════════════════════════════════════════════════════════════
    # C = C₀ + (1 − C₀) · [w_t·T_n  +  w_d·D_n  +  w_c·T_n·D_n]
    # The T_n·D_n term is the key addition: a fault must be BOTH late AND
    # persistent to trigger the maximum context amplification.
    context_variable = W_T * T_n + W_D * D_n + W_C * T_n * D_n
    context = C_FLOOR + (1.0 - C_FLOOR) * context_variable

    # ════════════════════════════════════════════════════════════════════
    # STEP 5 — Final intensity  I = P · B · C
    # ════════════════════════════════════════════════════════════════════
    return fault.probability * base_intensity * context


def calculate_combined_fault_intensity(faults: List[FaultConfig],
                                       typical_descent_time: float = 10.0,
                                       typical_altitude: float = 1000.0,
                                       reference_mass: float = 60.0) -> float:
    """
    Calculate combined intensity for multiple faults.
    
    For multiple faults, intensities combine sub-linearly (not purely additive)
    to represent that multiple faults may have overlapping or saturating effects.
    
    Formula: I_combined = 1 - prod(1 - I_i)
    
    This is the inclusion-exclusion (probabilistic-OR) combination where each
    individual I_i is produced by calculate_fault_intensity (now guaranteed
    ∈ [0, 1] by construction).
    
    Properties:
    - Multiple small faults accumulate realistically
    - Result stays in [0, 1] range by construction
    - Order of faults doesn't matter (commutative)
    - Adding a fault can only increase the combined intensity
    
    Args:
        faults: List of FaultConfig objects
        typical_descent_time: Expected descent duration
        typical_altitude: Expected starting altitude
        reference_mass: Reference vehicle mass
        
    Returns:
        Combined normalized intensity [0, 1]
    """
    if not faults:
        return 0.0
    
    # Calculate individual intensities
    intensities = [
        calculate_fault_intensity(f, typical_descent_time, typical_altitude, reference_mass)
        for f in faults
    ]
    
    # Combine using probability multiplication (1 - product of complements)
    combined = 1.0
    for intensity in intensities:
        combined *= (1.0 - intensity)
    
    return 1.0 - combined


def create_default_fault_configs() -> List[FaultGroup]:
    """Create default fault configurations for testing"""
    # Example: Mass loss at 50% through descent
    mass_loss = FaultConfig(
        fault_type=FaultType.MASS_LOSS,
        trigger_mode=TriggerMode.TIME_SINCE_APOGEE,
        trigger_value=2.0,
        magnitude=-5.0,  # Lose 5 kg
        duration=0.0,  # Permanent
        probability=0.0,  # Disabled by default
        fault_id=1
    )
    
    # Example: Thrust variation
    thrust_var = FaultConfig(
        fault_type=FaultType.THRUST_VAR,
        trigger_mode=TriggerMode.ALTITUDE_THRESHOLD,
        trigger_value=500.0,
        magnitude=0.8,  # 80% thrust
        duration=1.0,  # 1 second
        probability=0.0,
        fault_id=2
    )
    
    # Example: Drag increase
    drag_change = FaultConfig(
        fault_type=FaultType.DRAG_CHANGE,
        trigger_mode=TriggerMode.ABSOLUTE_TIME,
        trigger_value=10.0,
        magnitude=1.5,  # 150% drag
        duration=2.0,
        probability=0.0,
        fault_id=3
    )
    
    # Example: Wind gust
    wind_gust = FaultConfig(
        fault_type=FaultType.WIND_GUST,
        trigger_mode=TriggerMode.ALTITUDE_THRESHOLD,
        trigger_value=200.0,
        magnitude=10.0,  # +10 m/s wind
        duration=3.0,
        probability=0.0,
        fault_id=4
    )
    
    # Create groups (all individual for now)
    groups = [
        FaultGroup(faults=[mass_loss], concurrent=False, group_id=1),
        FaultGroup(faults=[thrust_var], concurrent=False, group_id=2),
        FaultGroup(faults=[drag_change], concurrent=False, group_id=3),
        FaultGroup(faults=[wind_gust], concurrent=False, group_id=4),
    ]
    
    return groups
