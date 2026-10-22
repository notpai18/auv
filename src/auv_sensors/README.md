# auv_sensors

Sensor abstraction, simulation, and state estimation layer for the Autonomous Underwater Vehicle (AUV).
Provides standardized `/auv/...` sensor topics so high-level navigation, controls, telemetry, and computer vision operate identically in Gazebo Harmonic simulation, in offline unit testing, and on physical hardware.

---

## Published Topics

| Topic | Type | Source / Provider | Description |
|---|---|---|---|
| `/auv/depth` | `std_msgs/msg/Float32` | `depth_sensor` | Depth below water surface (m, positive downwards) |
| `/auv/depth/pressure` | `sensor_msgs/msg/FluidPressure` | `depth_sensor` | Absolute hydrostatic pressure (Pa) |
| `/auv/dvl/velocity` | `geometry_msgs/msg/TwistWithCovarianceStamped` | `dvl_sensor` | 3D body-frame velocity $[u, v, w]$ with acoustic covariance |
| `/auv/dvl/altitude` | `std_msgs/msg/Float32` | `dvl_sensor` | Acoustic altitude above seabed (m) |
| `/auv/dvl/altitude_range` | `sensor_msgs/msg/Range` | `dvl_sensor` | Ultrasound range message for altitude |
| `/auv/dvl/odom` | `nav_msgs/msg/Odometry` | `dvl_sensor` | DVL dead-reckoned odometry |
| `/auv/dvl/status` | `std_msgs/msg/String` | `dvl_sensor` | DVL bottom lock status and beam health |
| `/auv/state_estimate` | `nav_msgs/msg/Odometry` | `sensor_fusion_node` | 6-DOF fused state estimate (IMU + Depth + DVL + Mag) |
| `/auv/attitude_rpy` | `geometry_msgs/msg/Vector3Stamped` | `sensor_fusion_node` | Filtered Roll, Pitch, Yaw angles (deg) |
| `/auv/filter_debug` | `std_msgs/msg/Float32MultiArray` | `sensor_fusion_node` | 12-field sensor fusion diagnostic array |
| `/auv/emergency_stop` | `std_msgs/msg/Bool` | `safety_supervisor_node` | Vehicle emergency stop flag |
| `/auv/safety_status` | `std_msgs/msg/String` | `safety_supervisor_node` | Real-time safety supervisor diagnostics |
| `/auv/leak` | `std_msgs/msg/Bool` | `leak_mock` | Hull leak detector (`False` = dry, `True` = leak detected) |
| `/auv/battery` | `sensor_msgs/msg/BatteryState` | `battery_mock` | 4S LiPo battery voltage, current, percentage, and health |
| `/auv/imu` | `sensor_msgs/msg/Imu` | Gazebo bridge / `imu_mock` | 9-DOF IMU orientation, angular velocity, and linear acceleration |
| `/auv/magnetometer` | `sensor_msgs/msg/MagneticField` | Gazebo bridge / `imu_mock` | 3-axis magnetic field vector (Tesla) |
| `/auv/stereo_front/left/image_raw` | `sensor_msgs/msg/Image` | Gazebo camera bridge | Front stereo left camera raw image feed |
| `/auv/stereo_front/right/image_raw` | `sensor_msgs/msg/Image` | Gazebo camera bridge | Front stereo right camera raw image feed |
| `/auv/bottom/image_raw` | `sensor_msgs/msg/Image` | Gazebo camera bridge | Downward-facing camera raw image feed |
| `/auv/stereo_front/left/camera_info` | `sensor_msgs/msg/CameraInfo` | Gazebo bridge | Front left camera calibration parameters |
| `/auv/stereo_front/right/camera_info` | `sensor_msgs/msg/CameraInfo` | Gazebo bridge | Front right camera calibration parameters |
| `/auv/bottom/camera_info` | `sensor_msgs/msg/CameraInfo` | Gazebo bridge | Downward camera calibration parameters |

---

## Subscribed Topics

| Topic | Type | Consumer | Description |
|---|---|---|---|
| `/auv/odom` | `nav_msgs/msg/Odometry` | `depth_sensor`, `dvl_sensor`, `imu_mock` | Vehicle state & position used to derive depth and synthetic inertial data |
| `/auv/imu` | `sensor_msgs/msg/Imu` | `sensor_fusion_node`, `safety_supervisor_node`, `dvl_sensor` | Gyroscope angular velocity and accelerometer specific force |
| `/auv/magnetometer` | `sensor_msgs/msg/MagneticField` | `sensor_fusion_node` | 3-axis magnetic compass field |
| `/auv/depth` | `std_msgs/msg/Float32` | `sensor_fusion_node`, `safety_supervisor_node`, `depth_hold_node` | Hydrostatic depth in meters |
| `/auv/dvl/velocity` | `geometry_msgs/msg/TwistWithCovarianceStamped` | `sensor_fusion_node` | DVL 3D body velocities |
| `/auv/leak` | `std_msgs/msg/Bool` | `safety_supervisor_node` | Hull leak detector |
| `/auv/battery` | `sensor_msgs/msg/BatteryState` | `safety_supervisor_node` | Battery state of charge |

---

## Real AUV Sensor Physics & Roles

```
                               ┌────────────────────────────────────────────────────────┐
                               │                    REAL AUV SENSORS                    │
                               └───────────────────────────┬────────────────────────────┘
               ┌───────────────────────┬───────────────────┼────────────────────┬──────────────────────┐
               ▼                       ▼                   ▼                    ▼                      ▼
       3-AXIS GYROSCOPE          PRESSURE TRANSDUCER     DVL (ACOUSTIC)      MAGNETOMETER          LEAK & BATTERY
       High-Rate Damping          Hydrostatic Depth     Acoustic Velocity     Heading Compass      Safety & Failsafe
               │                       │                   │                    │                      │
               ▼                       ▼                   ▼                    ▼                      ▼
  [Direct Rate Feedback]        [Depth PID Hold]    [Dead-Reckoning]    [Tilt-Compensated]    [Auto Emergency Ascent]
  Attitude Stabilization        Vertical Position     v_x, v_y, v_z         Yaw Heading        Thruster Kill & Alert
               │                       │                   │                    │
               └───────────────────────┴─────────┬─────────┴────────────────────┘
                                                 ▼
                                     ┌───────────────────────┐
                                     │  sensor_fusion_node   │
                                     │    6-DOF EKF Filter   │
                                     └───────────┬───────────┘
                                                 ▼
                                         /auv/state_estimate
```

### 1. 3-Axis Gyroscope (IMU)
- **Role in Real AUV**: Measures angular rates ($\omega_x, \omega_y, \omega_z$). Real underwater vehicles experience destabilizing hydrodynamic cross-moments (e.g., Munk moment, vortex shedding) and thruster torque reactions.
- **Active Damping**: The Gyroscope provides instantaneous derivative feedback ($-K_d \cdot \omega$) to the thrusters, dampening roll, pitch, and yaw oscillations before attitude error builds up.

### 2. Hydrostatic Pressure Transducer (Depth Sensor)
- **Role in Real AUV**: Because GPS RF signals attenuate within millimeters of water, hydrostatic pressure $P = P_{\text{atm}} + \rho g d$ provides the only absolute vertical position reference in underwater navigation.
- **Closed-Loop Depth Hold**: Directly feeds `depth_hold_node` and `sensor_fusion_node` to maintain precise submergence depth ($z = -d$).

### 3. Doppler Velocity Log (DVL)
- **Role in Real AUV**: Sends 4 ultrasonic beams down to the seabed in a Janus configuration. Measures Doppler shift to compute drift-free 3D body velocities ($u, v, w$) and seabed altitude.
- **Dead-Reckoning Navigation**: The cornerstone of underwater dead-reckoning when submerged.

### 4. 3-Axis Magnetometer (Electronic Compass)
- **Role in Real AUV**: Measures Earth's magnetic field vector. Tilt-compensated using roll and pitch from accelerometer to establish absolute magnetic heading ($\psi$). Eliminates gyro yaw drift.

### 5. Leak & Battery Safety Supervisors
- **Role in Real AUV**: Hull ingress probes detect seawater leaks at the lowest point of the hull. The supervisor commands immediate emergency surface ascent and thruster shutdown before electronics are destroyed.

---

## Nodes in `auv_sensors`

### 1. `dvl_sensor`
Simulates a 4-beam acoustic Doppler Velocity Log (e.g., Water Linked DVL A50).
- Publishes `/auv/dvl/velocity` (`TwistWithCovarianceStamped`)
- Publishes `/auv/dvl/altitude` (`Float32` & `Range`)
- Publishes `/auv/dvl/odom` (integrated DVL dead-reckoning)
- Publishes `/auv/dvl/status` (beam tracking status)

### 2. `sensor_fusion_node`
6-DOF Extended Sensor Fusion & State Estimator.
- Fuses Gyroscope, Accelerometer, Magnetometer, Depth Sensor, and DVL.
- Publishes `/auv/state_estimate` (`nav_msgs/Odometry`), `/auv/attitude_rpy`, and `/auv/filter_debug`.

### 3. `safety_supervisor_node`
Real-time safety and autonomous failsafe monitor.
- Continuously guards against hull leaks, low battery (<13.6V), tumbling tilt (>50°), and depth exceedance.
- Publishes `/auv/emergency_stop` and `/auv/safety_status`.

### 4. `depth_sensor`
Simulates a high-precision pressure transducer (e.g., BlueRobotics Bar02/Bar30).
- Computes depth: $d = \max(0, z_{\text{surface}} - z_{\text{odom}} + \mathcal{N}(0, \sigma^2))$.
- Computes absolute pressure: $P = P_{\text{atm}} + \rho g d$.

### 5. `leak_mock`
Simulates an optical/conductivity hull water ingress detector (`/auv/leak`).

### 6. `battery_mock`
Simulates a Smart Power Sense Module (4S LiPo battery, `/auv/battery`).

### 7. `imu_mock`
Offline IMU and Magnetometer generator for testing without Gazebo.

---

## Build and Launch

### Build:
```bash
cd ~/auv_ws
colcon build --packages-select auv_sensors auv_planner auv_controls
source install/setup.bash
```

### Launch full sensor layer:
```bash
ros2 launch auv_sensors sensors.launch.py
```
