import numpy as np

class StateEstimator:
    """
    Real-time parameter estimator for Mass and Drag Coefficient.
    Uses an Extended Kalman Filter (EKF) approach.
    """
    def __init__(self, initial_mass, initial_cd, reference_area, dt=0.01):
        self.dt = dt
        
        # State vector: [mass, Cd]
        self.x = np.array([initial_mass, initial_cd])
        
        # Covariance matrix P
        # Initial uncertainty
        self.P = np.diag([1.0, 0.01]) 
        
        # Process noise covariance Q
        # Allow mass to walk slightly (burn noise) or jump (faults -> needs high process noise or adaptive)
        # Allow Cd to walk slightly
        self.Q = np.diag([0.01, 0.001]) 
        
        # Measurement noise covariance R
        # Accelerometer noise variance
        self.R = 0.1 
        
        # Reference area (from rocket configuration)
        self.A_ref = reference_area
        
    def predict(self, mass_flow_rate=0.0):
        """
        Predict step of EKF.
        mass(k+1) = mass(k) - m_dot * dt
        Cd(k+1) = Cd(k)
        """
        # State prediction
        self.x[0] -= mass_flow_rate * self.dt
        
        # Covariance prediction
        # Jacobian F is Identity for this simple kinematic model
        # P = F P F.T + Q -> P = P + Q
        self.P += self.Q
        
    def update(self, accel_sens_z, velocity_z, altitude, rho, thrust_z):
        """
        Update step of EKF using accelerometer measurement.
        
        y = a_sens_z (measured)
        h(x) = (Thrust - Drag) / mass
             = (Thrust - 0.5 * rho * v^2 * Cd * A * sign(v)) / mass
             
        Note: We define Drag force opposing motion. 
        If moving DOWN (v < 0), Drag is UP (+z).
        F_drag_z = 0.5 * rho * v^2 * Cd * A   (positive)
        
        Equation of motion (Z-axis):
        m * a_inertial = -mg + T + D
        a_sens = a_inertial + g = (T + D) / m
        
        So h(x) = (T + 0.5 * rho * v^2 * Cd * A) / m   (assuming T is +z, D is +z for falling object)
        
        Wait, careful with signs.
        If falling (v < 0), Drag is +Z.
        If rising (v > 0), Drag is -Z.
        Let's use v_squared_signed = v * abs(v).
        D = -0.5 * rho * v * abs(v) * Cd * A
        
        h(x) = (T + D) / m
        """
        m = self.x[0]
        cd = self.x[1]
        
        # Dynamic pressure term (Force/Cd)
        q_dyn = 0.5 * rho * abs(velocity_z) * velocity_z * self.A_ref
        
        # Drag force (opposes velocity)
        # If v is negative (falling), drag is positive.
        # F_d = -0.5 * rho * v * |v| * Cd * A
        # Let q_factor = -0.5 * rho * v * |v| * A
        q_factor = -0.5 * rho * velocity_z * abs(velocity_z) * self.A_ref
        
        drag_force = q_factor * cd
        
        # Predicted Measurement
        h = (thrust_z + drag_force) / m
        
        # Measurement Residual
        y_residual = accel_sens_z - h
        
        # Jacobian H = dh/dx
        # h = (T + K*Cd)/m
        # dh/dm = -(T + K*Cd)/m^2 = -h/m = -a_pred / m
        # dh/dCd = K/m
        
        H = np.array([
            -h / m,       # dh/dm
            q_factor / m  # dh/dCd
        ])
        
        # Kalman Gain
        # S = H P H.T + R
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T / S
        
        # State Update
        self.x += K * y_residual
        
        # Covariance Update
        # P = (I - K H) P
        I = np.eye(2)
        self.P = (I - np.outer(K, H)) @ self.P
        
        # Safety Clamps
        self.x[0] = max(1.0, self.x[0]) # Mass > 1kg
        self.x[1] = max(0.1, min(self.x[1], 2.0)) # Cd in reasonable range [0.1, 2.0]
        
        return self.x
