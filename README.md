# HydRON Constellation Digital Twin & GUI Monitor

Welcome to the **HydRON Digital Twin (DT) Builder and GUI Monitor**, an interactive simulation environment designed for real-time visualization, configuration, and analysis of multi-layer satellite constellations (LEO, MEO, GEO) and their ground communications network.

Developed in Rust using the `egui` immediate-mode GUI framework, this project implements high-fidelity orbital mechanics, attitude control systems (ADCS), atmospheric attenuation models, and dynamic network routing simulation.

---

## 📶 Key Features

### 1. Interactive 3D Globe & Map Visualizer
* **Textured 3D Globe**: Renders a sphere representing Earth using `earth.jpg` coordinates, projected dynamically based on Greenwich Sidereal Time (GST) to align with inertial coordinates (ECI to ECEF).
* **Multi-Layer Constellation Rendering**: Visualizes circular orbits and positions for LEO, MEO, and GEO segments with configurable visual filters.
* **Camera Controls**: Zoom with the mouse wheel; rotate the globe by clicking and dragging on empty space.
* **Direct Satellite Dragging**: Click and drag any visible satellite directly on the screen to slide the entire constellation segment along its orbit, preserving the nominal circular spacing and physical velocity.

### 2. Network Link Capacity & Routing Simulation
* **Ground-to-Satellite Links (SGL)**: Simulates atmospheric attenuation on laser links between satellites and ground stations using an exponential atmospheric model and slant-path angles.
* **Inter-Satellite Links (ISL)**: Simulates laser links between adjacent satellites.
* **LEO Capacity Overrides**: Inter-satellite links involving at least one LEO satellite operate at a dynamically configured, stable capacity (bypassing free-space path loss attenuation) to simulate advanced laser terminals.
* **Real-Time Telemetry Panels**:
  * **LEO Throughput Panel**: A collapsible panel displaying live SGL and ISL throughput for LEO satellites, with selectable list entries.
  * **Ground Station Aggregate Throughput**: Live graphs showing station-by-station and total network aggregate data rates.

### 3. Simulation & Time Control
* **Play / Pause**: Toggle real-time propagation.
* **Time Warp Slider**: Accelerate or decelerate simulation time dynamically (from -50x to +50x).
* **System Reset**: Restore the simulation and constellations to initial values specified in `config.toml`.

### 4. Noise & Disturbance Injection (ADCS)
* **Active Attitude Kinematics**: Simulates reaction wheels and magnetorquers stabilizing the satellites.
* **Disturbance Injector**: Inject a 3-axis torque disturbance vector ($T_x, T_y, T_z$) to observe how the ADCS algorithm stabilizes the satellite bus.
* **Sensor Noise Configurations**: Configure noise levels for Gyro, Magnetometer, Sun Sensor, and Star Tracker dynamically.

### 5. 24h CSV Exporter
* Run a full 24-hour simulation sequence using the current configuration and export the results to a CSV file detailing ground station throughputs, link counts, and overall network data rate.

---

## 🛠 Architectural & Mathematical Modeling

### 1. Orbital Mechanics
Satellite orbits are propagated using a **Runge-Kutta 4th-order (RK4)** numerical integrator. The acceleration model incorporates:
* **Two-Body Gravity**: Standard Newtonian gravity around Earth ($\mu$).
* **J2 Oblateness Perturbation**: Accurately models the Earth's non-spherical mass distribution.
* **Atmospheric Drag**: Applied to LEO and lower MEO satellites using an exponential atmospheric density model ($\rho(h)$) and drag coefficient $C_d$.
* **Solar Radiation Pressure (SRP)**: Solar pressure model based on the sun vector and reflectivity coefficient $C_r$.

### 2. Spacecraft Attitude Dynamics & ADCS
Attitude is represented using quaternions $q = [\eta, \epsilon_1, \epsilon_2, \epsilon_3]$ to avoid gimbal lock:
* **Kinematics**: Rotational kinematics integrated via quaternion updates.
* **Stabilization**: Employs reaction wheel torques ($T_{rw}$) and magnetorquer control dipole commands ($m_{mtq}$) interacting with Earth's magnetic field ($B$).

### 3. Laser Link Capacity
Networking bandwidth uses a custom range-based capacity model:
$$C = C_{max} \cdot \left(\frac{d_{ref}}{d}\right)^2 \cdot \alpha_{atmos}$$
Where:
* $C_{max}$ is the dynamic satellite maximum capacity configured in the GUI.
* $d_{ref}$ is the reference link distance.
* $d$ is the actual distance between nodes.
* $\alpha_{atmos}$ is the atmospheric attenuation coefficient (only for SGL, based on local station weather states and slant path length).

---

## 🚀 Getting Started

### Prerequisites
* Rust compiler (MSRV 1.70+ recommended)
* Cargo package manager

### Building and Running

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd hydron
   ```
2. **Build the project**:
   ```bash
   cargo build --release
   ```
3. **Run the application**:
   ```bash
   cargo run
   ```

*Make sure `earth.jpg` and `config.toml` are in the working directory from which you run the application.*

---

## ⚙ Configuration (`config.toml`)

The application loads its default parameters from a `config.toml` file in the root directory. You can edit this file to configure:
* **Constellations**: Number of satellites, nominal altitudes, orbital inclinations, RAANs, and satellite mass/areas.
* **Ground Stations**: Geographical coordinates (latitude, longitude, altitude) and capacity limits.
* **Atmosphere**: Transition matrices for Markov weather state models and laser extinction values.
* **Environment Constants**: Earth gravity parameters, J2 coefficient, SRP constants, and atmospheric scale heights.

---

## 🎮 Interactive Controls Guide

### Left Panel (Configuration & Limits)
* **⚙ Visual Filters**: Checkboxes to toggle LEO ISL, MEO ISL, GEO ISL, or Ground Links (SGL) on/off. Includes a logarithmic map zoom slider.
* **📶 Bitrate Massimo Satelliti**: Dynamically adjust the peak bitrate capacity (Gbps) for LEO, MEO, and GEO satellites. Changes take effect instantly across all simulation calculations and the CSV exporter.
* **📡 Modifica Costellazione**: Change constellation sizes, altitudes, and inclinations on the fly.
* **🏠 Stazioni di Terra**: Add new ground stations or manually override local weather states (e.g., Clear Sky, Light Rain, Heavy Rain, Storm) to observe SGL link degradation.

### Central Panel (3D Map & Plot)
* **3D Visualizer**:
  * Drag empty space to rotate the Earth.
  * Use mouse scroll to zoom in/out.
  * **Drag Satellites**: Click directly on a satellite and drag it to rotate the whole constellation segment along its orbit plane.
* **📊 Station Throughput Plot**: Live graph of ground station and total network aggregate data rates.

### Right Panel (Telemetry & Console)
* **📶 Bitrate Satelliti LEO**: Monitor live LEO satellite total throughput and specific active link breakdowns (SGL/ISL speeds). Click a satellite's name in the list to select it.
* **Satellite Telemetry**: Read exact ECI position/velocity coordinates, attitude quaternions, and ADCS actuator states.
* **Iniettore Disturbi ADCS**: Inject 3D torques to test stabilization.
* **Rumore Sensori**: Slide values to increase sensor noise, introducing jitter to the stabilization algorithm.
* **System Logs**: Live event feed tracking connections, disconnections, and export triggers.

---

## 🤖 Development & Credits
This project was developed in pair programming with the **Gemini AI Coding Agent** (Google DeepMind's Advanced Agentic Coding system, *Antigravity*).
