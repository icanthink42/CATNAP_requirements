# CATNAP
Coupled Adiabatic Tank and Nozzle Analysis Program - A multiphase flow and regenerative cooling solver for dual-acting vapor pressurized propulsion systems.

CATNAP couples an iterative control volume solver with multiphase flow functions, regenerative cooling functions, and NASA CEA combustion analysis to form a high fidelity estimate of propulsion system characteristics over the course of a burn. Generally, it is meant to be used
for Nitrous Oxide (N2O) and a generic fuel (Ethanol, IPA, Jet-A). 

Install dependencies with:
```sh
pip install -r requirements.txt
```

## Configuration

CATNAP uses a `config.toml` file for all simulation parameters. To get started, copy the example config:

```sh
# Linux/macOS
cp config.example.toml config.toml

# Windows (Command Prompt)
copy config.example.toml config.toml

# Windows (PowerShell)
Copy-Item config.example.toml config.toml
```

Edit `config.toml` to customize your simulation. The config file contains sections for:

- **[simulation]** - Time and timestep settings
- **[propellants]** - Oxidizer and fuel names (CoolProp/CEA compatible)
- **[initial_conditions]** - Tank mass, temperatures, pressures
- **[mass_flow]** - Flow rates for oxidizer, fuel, coolant, film cooling
- **[injector]** - Discharge coefficients, angles, orifice areas
- **[nozzle]** - Throat/exit diameters, lengths, radii of curvature
- **[thermodynamics]** - Gas properties (gamma, R, combustion temp)
- **[regen_cooling]** - Channel geometry and solver parameters
- **[materials]** - Chamber thermal conductivity
- **[visualization]** - Plot settings
- **[output]** - Engine name, dashboard file paths

For standalone exe releases, place `config.toml` and `catnap_dashboard.html` in the same folder as `CATNAP.exe`.

Required inputs:
-Geometry: engine, cooling channels, and injector
-Propellants (CEA_obj class input used in combustion and transport analysis)
-Initial Conditions: saturation temperature, vapor mass fraction, total mass, target mass flow rates, target chamber pressure
-Material Properties: hotwall thermal conductivity, channel surface roughness, [maximum temperature, maximum stress] - set by user

CATNAP is unfinished! There may be bugs I have not caught.
