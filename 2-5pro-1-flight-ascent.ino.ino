/********************************************************************************
 * Vortex 1 - On-Board Flight Computer Code
 * Last Updated: 09/25/2025 (MM/DD/YYYY) 5:21 PM EST
 * Author(s): Mishra Rockets
 * Compiler History:
  * 9/24/2025 (6): Arduino IDE
  * 9/24/2025 (2): PlatformIO for Code-OSS
  * 9/24/2025 (3): PlatformIO for VSCode
  * 9/24/2025 (5): PlatformIO for JB_Google/CLion
  * 9/25/2025 (1): PlatformIO for JB_Google/CLion
 *
 * TODO:
 * 1.  Verify and change all pin definitions in the "HARDWARE PINS" section.
 * 2.  Tune the PID gains (KP, KI, KD)
 * 3.  Tune the Kalman Filter gains (Q_ACCEL, R_ALTITUDE).
 * 4.  Verify the mapping of the IMU axes to rocket's pitch and yaw.
 * 5.  Conduct extensive ground testing (static fires, component testing) before attempting a flight.
 *
 ********************************************************************************/

// LIBRARIES
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <Servo.h>
#include <Adafruit_BNO055.h>
#include <Adafruit_MPL3115A2.h>
#include <Adafruit_HDC302x.h>
#include <Adafruit_INA219.h>
#include <Adafruit_GPS.h>
#include <RH_RF95.h>
#include <Adafruit_Sensor.h>
#include <Watchdog_t4.h> // Specific Watchdog library for Teensy 4.1

//================================================================================
// HARDWARE PINS & CONFIGURATION
//================================================================================
const int PIN_SERVO_PITCH = 9;
const int PIN_SERVO_YAW = 10;
const int PIN_MOSFET_IGNITER = 20;
const int PIN_MOSFET_PARACHUTE = 21;
const int PIN_LED_STATUS = LED_BUILTIN;
const int PIN_ARMING_SWITCH = 22;
const int PIN_SD_CS = 10;
const int RFM95_CS = 5;
const int RFM95_RST = 6;
const int RFM95_INT = 7;
#define RF95_FREQ 915.0

// PID CONTROLLER GAINS
const double KP = 2.5; const double KI = 0.1; const double KD = 0.8;

// KALMAN FILTER TUNING GAINS
// This is now a 3-state filter (pos, vel, accel)
const float SIGMA_ACCEL_NOISE = 0.5; // Process noise. How much we expect the acceleration to change unpredictably between steps.
const float R_ALTITUDE = 10.0;       // Measurement noise. How much we trust the barometer reading.

// CONSTANTS & FLIGHT PARAMETERS
const float SEA_LEVEL_PRESSURE_HPA = 1013.25;
const int TVC_UPDATE_RATE_HZ = 100;
const unsigned long TVC_UPDATE_INTERVAL_US = 1000000 / TVC_UPDATE_RATE_HZ;
const int STATE_ESTIMATION_RATE_HZ = 50;
const unsigned long STATE_ESTIMATION_INTERVAL_US = 1000000 / STATE_ESTIMATION_RATE_HZ;
const float MIN_BATT_VOLTAGE = 7.0;

// FLIGHT EVENT CRITERIA
const float LIFTOFF_ACCEL_G = 1.2;
const float LIFTOFF_ALTITUDE_M = 5.0;
const float BURNOUT_ACCEL_G = 0.5;
const unsigned long BURNOUT_DETECT_MS = 200;
const unsigned long APOGEE_CONFIRMATION_MS = 250;
const float TOUCHDOWN_ACCEL_G = 5.0;
const float TOUCHDOWN_ALTITUDE_M = 20.0;

// SERVO CONFIGURATION
const int SERVO_PITCH_CENTER_US = 1500;
const int SERVO_YAW_CENTER_US = 1500;
const int SERVO_MAX_ANGLE = 20;

//================================================================================
// GLOBAL OBJECTS & VARIABLES
//================================================================================
WDT_T4<WDT1> wdt; 
enum FlightState {
  INITIALIZING, PRE_FLIGHT_CHECKS, ARMED, LAUNCH_COUNTDOWN, IGNITED,
  ASCENT_TVC, COAST, PARACHUTE_DEPLOY, DESCENT, TOUCHDOWN,
  RECOVERY_MODE, ABORTED, FAILURE_STATE
};
volatile FlightState currentState = INITIALIZING;

Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);
Adafruit_MPL3115A2 baro = Adafruit_MPL3115A2();
Adafruit_HDC302x hdc = Adafruit_HDC302x();
Adafruit_INA219 ina219 = Adafruit_INA219();
Adafruit_GPS GPS(&Serial1);
RH_RF95 rf95(RFM95_CS, RFM95_INT);
Servo servoPitch, servoYaw;
File dataFile;

float altitude = 0.0, velocity = 0.0, acceleration = 0.0, rawBaroAltitude = 0.0, maxAltitude = 0.0, initialAltitude = 0.0;
imu::Vector<3> euler, linearAccel;

// State struct for the 3-state (position, velocity, acceleration) Kalman Filter
struct Kalman { float x[3]; float P[3][3]; float r; float sigma_a; } kalman_state;

double pidPitchOutput, pidYawOutput;
double pitchError, yawError, lastPitchError = 0, lastYawError = 0;
double pitchIntegral = 0, yawIntegral = 0;
double setpointPitch = 0.0, setpointYaw = 0.0;

unsigned long stateTimer = 0, lastTvcUpdate = 0, lastActionTimer = 0, lastStateEstimationUpdate = 0;
unsigned long apogeeTimer = 0;

//================================================================================
// 3-STATE KALMAN FILTER FUNCTIONS (Constant Acceleration Model)
//================================================================================
void kalman_init(float initial_altitude) {
    kalman_state.x[0] = initial_altitude; // Position
    kalman_state.x[1] = 0;                // Velocity
    kalman_state.x[2] = 0;                // Acceleration
    
    kalman_state.P[0][0] = 1.0; kalman_state.P[0][1] = 0.0; kalman_state.P[0][2] = 0.0;
    kalman_state.P[1][0] = 0.0; kalman_state.P[1][1] = 100.0; kalman_state.P[1][2] = 0.0;
    kalman_state.P[2][0] = 0.0; kalman_state.P[2][1] = 0.0; kalman_state.P[2][2] = 100.0;

    kalman_state.r = R_ALTITUDE;
    kalman_state.sigma_a = SIGMA_ACCEL_NOISE;
}

void kalman_predict(float dt) {
    // State transition matrix F
    float F[3][3] = {{1, dt, 0.5f*dt*dt}, {0, 1, dt}, {0, 0, 1}};
    
    // Predict state: x_pred = F * x
    float x_pred[3];
    x_pred[0] = F[0][0]*kalman_state.x[0] + F[0][1]*kalman_state.x[1] + F[0][2]*kalman_state.x[2];
    x_pred[1] = F[1][0]*kalman_state.x[0] + F[1][1]*kalman_state.x[1] + F[1][2]*kalman_state.x[2];
    x_pred[2] = F[2][0]*kalman_state.x[0] + F[2][1]*kalman_state.x[1] + F[2][2]*kalman_state.x[2];
    
    kalman_state.x[0] = x_pred[0];
    kalman_state.x[1] = x_pred[1];
    kalman_state.x[2] = x_pred[2];

    // Process noise covariance matrix Q
    float dt2 = dt*dt, dt3 = dt2*dt, dt4 = dt3*dt;
    float sa2 = kalman_state.sigma_a * kalman_state.sigma_a;
    float Q[3][3] = {
      {0.25f*dt4*sa2, 0.5f*dt3*sa2, 0.5f*dt2*sa2},
      {0.5f*dt3*sa2, dt2*sa2, dt*sa2},
      {0.5f*dt2*sa2, dt*sa2, 1.0f*sa2}
    };

    // Predict covariance: P_pred = F * P * F' + Q
    float FP[3][3], P_pred[3][3];
    // P_pred = F * P
    for(int i=0; i<3; ++i) for(int j=0; j<3; ++j) FP[i][j] = F[i][0]*kalman_state.P[0][j] + F[i][1]*kalman_state.P[1][j] + F[i][2]*kalman_state.P[2][j];
    // P_pred = (F*P) * F' + Q
    for(int i=0; i<3; ++i) for(int j=0; j<3; ++j) P_pred[i][j] = FP[i][0]*F[j][0] + FP[i][1]*F[j][1] + FP[i][2]*F[j][2] + Q[i][j];
    
    for(int i=0; i<3; ++i) for(int j=0; j<3; ++j) kalman_state.P[i][j] = P_pred[i][j];
}

void kalman_update(float baro_alt, float acc_z) {
    
    // 1. Update with Barometer
    float y_baro = baro_alt - kalman_state.x[0]; // H_baro = [1, 0, 0]
    float S_baro = kalman_state.P[0][0] + kalman_state.r;
    float K_baro[3] = {kalman_state.P[0][0]/S_baro, kalman_state.P[1][0]/S_baro, kalman_state.P[2][0]/S_baro};
    
    kalman_state.x[0] += K_baro[0] * y_baro;
    kalman_state.x[1] += K_baro[1] * y_baro;
    kalman_state.x[2] += K_baro[2] * y_baro;

    float I_KH[3][3] = {{1-K_baro[0], 0, 0}, {-K_baro[1], 1, 0}, {-K_baro[2], 0, 1}};
    float P_new[3][3] = {0};
    for(int i=0; i<3; ++i) for(int j=0; j<3; ++j) for(int k=0; k<3; ++k) P_new[i][j] += I_KH[i][k] * kalman_state.P[k][j];
    for(int i=0; i<3; ++i) for(int j=0; j<3; ++j) kalman_state.P[i][j] = P_new[i][j];

    // 2. Update with Accelerometer (as a pseudo-measurement of state 2)
    // This part is more heuristic than a pure Kalman filter but helps anchor the acceleration state.
    float R_accel = 1.0; // Uncertainty of accelerometer reading
    float y_accel = acc_z - kalman_state.x[2]; // H_accel = [0, 0, 1]
    float S_accel = kalman_state.P[2][2] + R_accel;
    float K_accel[3] = {kalman_state.P[0][2]/S_accel, kalman_state.P[1][2]/S_accel, kalman_state.P[2][2]/S_accel};

    kalman_state.x[0] += K_accel[0] * y_accel;
    kalman_state.x[1] += K_accel[1] * y_accel;
    kalman_state.x[2] += K_accel[2] * y_accel;

    float I_KH2[3][3] = {{1, 0, -K_accel[0]}, {0, 1, -K_accel[1]}, {0, 0, 1-K_accel[2]}};
    float P_new2[3][3] = {0};
    for(int i=0; i<3; ++i) for(int j=0; j<3; ++j) for(int k=0; k<3; ++k) P_new2[i][j] += I_KH2[i][k] * kalman_state.P[k][j];
    for(int i=0; i<3; ++i) for(int j=0; j<3; ++j) kalman_state.P[i][j] = P_new2[i][j];
}


//================================================================================
// SETUP FUNCTION
//================================================================================
void setup() {
  pinMode(PIN_LED_STATUS, OUTPUT); pinMode(PIN_MOSFET_IGNITER, OUTPUT); pinMode(PIN_MOSFET_PARACHUTE, OUTPUT);
  pinMode(PIN_ARMING_SWITCH, INPUT_PULLUP); digitalWrite(PIN_MOSFET_IGNITER, LOW); digitalWrite(PIN_MOSFET_PARACHUTE, LOW);

  Serial.begin(115200); Serial.println("Flight Computer Initializing...");

  WDT_timings_t config;
  config.trigger = 5; // 5 seconds timeout
  config.timeout = 8; // 8 seconds before reset
  wdt.begin(config);

  if (!SD.begin(PIN_SD_CS)) { enterFailureState("SD_INIT_FAIL"); return; }
  dataFile = SD.open("flightlog.csv", FILE_WRITE);
  if (dataFile) {
    dataFile.println("Timestamp,State,RawAlt,KF_Alt,KF_Vel,KF_Accel,Temp,Humidity,Pressure,Batt V,Batt C,Roll,Pitch,Yaw,LinAccZ,PitchCmd,YawCmd");
    dataFile.flush(); logSystemEvent("System Initialized");
  } else { enterFailureState("SD_OPEN_FAIL"); return; }

  if (!bno.begin()) { enterFailureState("IMU_INIT_FAIL"); return; }
  bno.setExtCrystalUse(true);

  if (!baro.begin()) { enterFailureState("BARO_INIT_FAIL"); return; }

  hdc.begin();
  ina219.begin();

  pinMode(RFM95_RST, OUTPUT); digitalWrite(RFM95_RST, HIGH); delay(100);
  digitalWrite(RFM95_RST, LOW); delay(10); digitalWrite(RFM95_RST, HIGH); delay(10);
  if (!rf95.init()) { Serial.println("LoRa radio init failed"); }
  rf95.setFrequency(RF95_FREQ); rf95.setTxPower(23, false);
  transmitTelemetry("Radio Initialized");

  servoPitch.attach(PIN_SERVO_PITCH); servoYaw.attach(PIN_SERVO_YAW);
  servoPitch.writeMicroseconds(SERVO_PITCH_CENTER_US); servoYaw.writeMicroseconds(SERVO_YAW_CENTER_US);
  
  GPS.begin(9600); GPS.sendCommand(PMTK_SET_NMEA_OUTPUT_RMCGGA); GPS.sendCommand(PMTK_SET_NMEA_UPDATE_1HZ);

  logSystemEvent("Initialization Complete");
  currentState = PRE_FLIGHT_CHECKS; stateTimer = millis();
}
//================================================================================
// MAIN LOOP
//================================================================================
void loop() {
  wdt.feed();
  
  readSensors();
  if (micros() - lastStateEstimationUpdate >= STATE_ESTIMATION_INTERVAL_US) {
    lastStateEstimationUpdate = micros();
    updateStateEstimation();
  }
  logData(); handleRadio();
  
  if (currentState == ASCENT_TVC) {
    if (micros() - lastTvcUpdate >= TVC_UPDATE_INTERVAL_US) { lastTvcUpdate = micros(); runTvcPidController(); }
  }

  switch (currentState) {
    case PRE_FLIGHT_CHECKS: handlePreFlightChecks(); break; case ARMED: handleArmed(); break;
    case LAUNCH_COUNTDOWN: handleLaunchCountdown(); break; case IGNITED: handleIgnited(); break;
    case ASCENT_TVC: handleAscentTvc(); break; case COAST: handleCoast(); break;
    case PARACHUTE_DEPLOY: handleParachuteDeploy(); break; case DESCENT: handleDescent(); break;
    case TOUCHDOWN: handleTouchdown(); break; case RECOVERY_MODE: handleRecoveryMode(); break;
    case ABORTED: handleAbortedFailure("ABORTED"); break; case FAILURE_STATE: handleAbortedFailure("FAILURE"); break;
    default: break;
  }
}
//================================================================================
// SENSOR & ESTIMATION FUNCTIONS
//================================================================================
void readSensors() {
  euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
  linearAccel = bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);
  rawBaroAltitude = baro.getAltitude(); GPS.read();
}
void updateStateEstimation() {
    static unsigned long lastTime = 0;
    unsigned long currentTime = micros();
    if (lastTime == 0) { lastTime = currentTime; return; }
    float dt = (currentTime - lastTime) / 1000000.0f;
    lastTime = currentTime;
    
    kalman_predict(dt);
    kalman_update(rawBaroAltitude, linearAccel.z());
    
    altitude = kalman_state.x[0];
    velocity = kalman_state.x[1];
    acceleration = kalman_state.x[2];
    
    if (altitude > maxAltitude) { maxAltitude = altitude; }
}
//================================================================================
// STATE HANDLER FUNCTIONS
//================================================================================
void handlePreFlightChecks() {
  static int check_sub_state = 0;
  if (check_sub_state == 0) {
    float totalAltitude = 0; for (int i = 0; i < 10; i++) { totalAltitude += baro.getAltitude(); delay(50); }
    initialAltitude = totalAltitude / 10.0; kalman_init(initialAltitude);
    logSystemEvent("Baro Calibrated. Initial Alt: " + String(initialAltitude));
    lastActionTimer = millis(); check_sub_state = 1;
  } else if (check_sub_state == 1 && millis() - lastActionTimer > 500) {
    servoPitch.write(90 + SERVO_MAX_ANGLE); servoYaw.write(90 + SERVO_MAX_ANGLE); lastActionTimer = millis(); check_sub_state = 2;
  } else if (check_sub_state == 2 && millis() - lastActionTimer > 1000) {
    servoPitch.write(90 - SERVO_MAX_ANGLE); servoYaw.write(90 - SERVO_MAX_ANGLE); lastActionTimer = millis(); check_sub_state = 3;
  } else if (check_sub_state == 3 && millis() - lastActionTimer > 1000) {
    servoPitch.write(90); servoYaw.write(90); logSystemEvent("Checks Passed. Waiting for arm."); transmitTelemetry("Checks OK. Ready to Arm.");
    lastActionTimer = millis(); check_sub_state = 4;
  } else if (check_sub_state == 4) {
    if (digitalRead(PIN_ARMING_SWITCH) == LOW) {
      currentState = ARMED; logSystemEvent("System ARMED."); transmitTelemetry("ARMED"); stateTimer = millis();
    } else { if (millis() - lastActionTimer > 500) { digitalWrite(PIN_LED_STATUS, !digitalRead(PIN_LED_STATUS)); lastActionTimer = millis(); } }
  }
}
void handleArmed() {
    if (ina219.getBusVoltage_V() < MIN_BATT_VOLTAGE) {
        enterFailureState("LOW_BATTERY");
        return;
    }
    if(millis() - lastActionTimer > 100) { digitalWrite(PIN_LED_STATUS, !digitalRead(PIN_LED_STATUS)); lastActionTimer = millis(); }
}
void handleLaunchCountdown() {
  long countdown = 10 - ((millis() - stateTimer) / 1000);
  if (countdown <= 0) {
    currentState = IGNITED; logSystemEvent("Countdown complete. IGNITING."); transmitTelemetry("IGNITION");
    stateTimer = millis(); lastActionTimer = millis();
  } else { if (millis() - lastActionTimer >= 1000) { String telemetry = "T-" + String(countdown); transmitTelemetry(telemetry.c_str()); lastActionTimer = millis(); } }
}
void handleIgnited() {
  static bool igniterFired = false;
  if (!igniterFired) { digitalWrite(PIN_MOSFET_IGNITER, HIGH); igniterFired = true; stateTimer = millis(); }
  if (millis() - stateTimer > 1000) { digitalWrite(PIN_MOSFET_IGNITER, LOW); }
  // Liftoff requires two conditions: acceleration and altitude gain
  if (linearAccel.z() > (LIFTOFF_ACCEL_G * 9.81) && (altitude - initialAltitude) > LIFTOFF_ALTITUDE_M) {
    logSystemEvent("Liftoff Detected!"); transmitTelemetry("LIFTOFF"); currentState = ASCENT_TVC; stateTimer = millis(); lastTvcUpdate = micros();
  }
  if (millis() - lastActionTimer > 5000) { enterFailureState("LIFTOFF_FAIL"); }
}
void handleAscentTvc() {
  if (acceleration < (BURNOUT_ACCEL_G * 9.81)) { // Using filtered acceleration
    if (millis() - stateTimer > BURNOUT_DETECT_MS) {
      currentState = COAST; logSystemEvent("Motor Burnout Detected."); transmitTelemetry("BURNOUT");
      servoPitch.writeMicroseconds(SERVO_PITCH_CENTER_US); servoYaw.writeMicroseconds(SERVO_YAW_CENTER_US); stateTimer = millis();
    }
  } else { stateTimer = millis(); }
}
void handleCoast() {
    // **Software Redundancy for Apogee Detection**
    // Condition 1: Kalman filter velocity estimate is negative.
    // Condition 2: Raw barometer altitude is also starting to decrease from its max.
    static float coastMaxBaroAlt = 0;
    if (rawBaroAltitude > coastMaxBaroAlt) { coastMaxBaroAlt = rawBaroAltitude; }

    if (velocity < 0 && rawBaroAltitude < (coastMaxBaroAlt - 1.0)) { // Velocity is negative AND we've dropped at least 1m from the baro peak
        if (apogeeTimer == 0) { apogeeTimer = millis(); }
        if (millis() - apogeeTimer > APOGEE_CONFIRMATION_MS) {
            currentState = PARACHUTE_DEPLOY;
            logSystemEvent("Apogee Confirmed. Max Alt: " + String(maxAltitude));
            transmitTelemetry("APOGEE");
            stateTimer = millis();
        }
    } else {
        apogeeTimer = 0; // Reset timer if conditions are not met
    }
}
void handleParachuteDeploy() {
  static bool chargeFired = false;
  if (millis() - stateTimer > 500 && !chargeFired) {
    digitalWrite(PIN_MOSFET_PARACHUTE, HIGH); logSystemEvent("Firing Parachute Charge."); transmitTelemetry("PARACHUTE DEPLOY");
    lastActionTimer = millis(); chargeFired = true;
  }
  if (chargeFired && (millis() - lastActionTimer > 1000)) { digitalWrite(PIN_MOSFET_PARACHUTE, LOW); currentState = DESCENT; stateTimer = millis(); }
}
void handleDescent() {
    // **Software Redundancy for Touchdown Detection**
    // Condition 1: Large G-spike from the accelerometer.
    // Condition 2: Altitude is near the ground.
    // Condition 3: Filtered velocity is near zero (i.e., not in freefall).
    float totalAccMag = sqrt(linearAccel.x()*linearAccel.x() + linearAccel.y()*linearAccel.y() + linearAccel.z()*linearAccel.z());
    bool isNearGround = (altitude - initialAltitude) < TOUCHDOWN_ALTITUDE_M;
    bool isStableOnGround = abs(velocity) < 2.0;

    if (isNearGround && isStableOnGround && totalAccMag > (TOUCHDOWN_ACCEL_G * 9.81)) {
        currentState = TOUCHDOWN;
        logSystemEvent("Touchdown Detected.");
        transmitTelemetry("TOUCHDOWN");
        stateTimer = millis();
    }
}
void handleTouchdown() {
    logSystemEvent("Disarming and entering recovery."); dataFile.close(); currentState = RECOVERY_MODE; stateTimer = millis(); lastActionTimer = millis();
}
void handleRecoveryMode() {
  if(millis() - lastActionTimer > 2000) {
      digitalWrite(PIN_LED_STATUS, HIGH);
      if (GPS.newNMEAreceived()) { GPS.parse(GPS.lastNMEA()); }
      if (GPS.fix) { String coords = "GPS: " + String(GPS.latitudeDegrees, 4) + "," + String(GPS.longitudeDegrees, 4); transmitTelemetry(coords.c_str()); }
      else { transmitTelemetry("No GPS fix"); }
      lastActionTimer = millis();
  }
  if(millis() - lastActionTimer > 100) { digitalWrite(PIN_LED_STATUS, LOW); }
}
void handleAbortedFailure(const char* reason) {
  digitalWrite(PIN_MOSFET_IGNITER, LOW); digitalWrite(PIN_MOSFET_PARACHUTE, LOW);
  servoPitch.writeMicroseconds(SERVO_PITCH_CENTER_US); servoYaw.writeMicroseconds(SERVO_YAW_CENTER_US);
  static bool isFirstCall = true;
  if(isFirstCall) { logSystemEvent(reason); transmitTelemetry(reason); isFirstCall = false; lastActionTimer = millis(); }
  if(millis() - lastActionTimer > 200) { digitalWrite(PIN_LED_STATUS, !digitalRead(PIN_LED_STATUS)); lastActionTimer = millis(); }
}
void enterFailureState(const char* reason) { currentState = FAILURE_STATE; handleAbortedFailure(reason); }
//================================================================================
// CONTROLLER & UTILITY FUNCTIONS
//================================================================================
void runTvcPidController() {
  double currentPitch = euler.y(), currentYaw = euler.z();
  pitchError = setpointPitch - currentPitch; yawError = setpointYaw - currentYaw;
  pitchIntegral += KI * pitchError; yawIntegral += KI * yawError;
  pitchIntegral = constrain(pitchIntegral, -10, 10); yawIntegral = constrain(yawIntegral, -10, 10);
  double d_pitch = KD * (pitchError - lastPitchError), d_yaw = KD * (yawError - lastYawError);
  lastPitchError = pitchError; lastYawError = yawError;
  pidPitchOutput = (KP * pitchError) + pitchIntegral + d_pitch; pidYawOutput = (KP * yawError) + yawIntegral + d_yaw;
  int servoPitchAngle = 90 + constrain(pidPitchOutput, -SERVO_MAX_ANGLE, SERVO_MAX_ANGLE);
  int servoYawAngle = 90 + constrain(pidYawOutput, -SERVO_MAX_ANGLE, SERVO_MAX_ANGLE);
  servoPitch.write(servoPitchAngle); servoYaw.write(servoYawAngle);
}
void logData() {
  static unsigned long lastLogTime = 0; static int logCounter = 0;
  if (millis() - lastLogTime < 100) return;
  lastLogTime = millis();
  if (dataFile) {
    double temp = 0.0, humidity = 0.0;
    hdc.readTemperatureHumidityOnDemand(temp, humidity, TRIGGERMODE_LP0);

    dataFile.print(millis()); dataFile.print(","); dataFile.print(currentState); dataFile.print(",");
    dataFile.print(rawBaroAltitude, 2); dataFile.print(","); dataFile.print(altitude, 2); dataFile.print(",");
    dataFile.print(velocity, 2); dataFile.print(","); dataFile.print(acceleration, 2); dataFile.print(",");
    dataFile.print(temp, 2); dataFile.print(","); dataFile.print(humidity, 2); dataFile.print(",");
    dataFile.print(baro.getPressure(), 2); dataFile.print(",");
    dataFile.print(ina219.getBusVoltage_V()); dataFile.print(","); dataFile.print(ina219.getCurrent_mA()); dataFile.print(",");
    dataFile.print(euler.x()); dataFile.print(","); dataFile.print(euler.y()); dataFile.print(",");
    dataFile.print(euler.z()); dataFile.print(","); dataFile.print(linearAccel.z()); dataFile.print(",");
    dataFile.print(pidPitchOutput); dataFile.print(","); dataFile.println(pidYawOutput);
    if (++logCounter >= 10) { dataFile.flush(); logCounter = 0; }
  }
}
void handleRadio() {
    if (rf95.available()) {
        uint8_t buf[RH_RF95_MAX_MESSAGE_LEN]; uint8_t len = sizeof(buf);
        if (rf95.recv(buf, &len)) {
            if (strcmp((char*)buf, "LAUNCH") == 0 && currentState == ARMED) {
                currentState = LAUNCH_COUNTDOWN; logSystemEvent("Launch command received."); transmitTelemetry("Countdown Started");
                stateTimer = millis(); lastActionTimer = millis();
            } else if (strcmp((char*)buf, "ABORT") == 0 && currentState < COAST) {
                currentState = ABORTED; logSystemEvent("ABORT command received.");
            }
        }
    }
}
void logSystemEvent(const String& event) {
  if (dataFile) {
    dataFile.print(millis()); dataFile.print(",EVENT,"); dataFile.println(event); dataFile.flush();
  }
  Serial.print("EVENT: "); Serial.println(event);
}
void transmitTelemetry(const char* message) {
  rf95.send((uint8_t *)message, strlen(message));
  Serial.print("Telemetry Sent: "); Serial.println(message);
}