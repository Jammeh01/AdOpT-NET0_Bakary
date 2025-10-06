import adopt_net0 as adopt
import json
from pathlib import Path
import os
import pandas as pd
import numpy as np
from data import *
from adopt_net0.data_preprocessing import load_climate_data_from_api

print("🇮🇹 Starting Italian Energy Model Optimization...")

# Get the directory where the main.py script is located
script_dir = Path(__file__).parent

# Create folders
results_data_path = script_dir / "userData"
results_data_path.mkdir(parents=True, exist_ok=True)

# create input data path and optimization templates
input_data_path = script_dir / "macro_decarbonisation_working"
input_data_path.mkdir(parents=True, exist_ok=True)

print("📋 Creating optimization templates...")
# Create template input JSONs
adopt.create_optimization_templates(input_data_path)

print("🗺️ Setting up topology...")
# Load json template
with open(input_data_path / "Topology.json", "r") as json_file:
    topology = json.load(json_file)
# Nodes
topology["nodes"] = ["northwest", "northeast", "center", "south", "islands"]
# Carriers: The Carries/ Vectors we have in the CGE model are gas, oil, electricity
topology["carriers"] = ["electricity", "heat", "gas", "hydrogen"]
# Investment periods:
topology["investment_periods"] = ["period1"]
# Save json template
with open(input_data_path / "Topology.json", "w") as json_file:
    json.dump(topology, json_file, indent=4)

print("⚙️ Configuring model settings...")
# Load json template
with open(input_data_path / "ConfigModel.json", "r") as json_file:
    configuration = json.load(json_file)
# Set time aggregation settings:
configuration["optimization"]["typicaldays"]["N"]["value"] = 30
configuration["optimization"]["typicaldays"]["method"]["value"] = 1
# Set MILP gap
configuration["solveroptions"]["mipgap"]["value"] = 0.02
# Save json template
with open(input_data_path / "ConfigModel.json", "w") as json_file:
    json.dump(configuration, json_file, indent=4)

print("📁 Creating input data folder structure...")
adopt.create_input_data_folder_template(input_data_path)

print("📍 Setting up node locations...")
# Define node locations based on actual Italian regional energy centers
node_location = pd.read_csv(
    input_data_path / "NodeLocations.csv", sep=';', index_col=0, header=0)
node_lon = {'northwest': 9.2, 'northeast': 11.9, 'center': 12.5,
            'south': 14.8, 'islands': 12.5}  # longitude in degrees
node_lat = {'northwest': 45.4, 'northeast': 45.5, 'center': 42.8,
            'south': 40.8, 'islands': 38.0}  # latitude in degrees
# Elevation in meters
node_alt = {'northwest': 120, 'northeast': 50,
            'center': 250, 'south': 200, 'islands': 0}
for node in ['northwest', 'northeast', 'center', 'south', 'islands']:
    node_location.at[node, 'lon'] = node_lon[node]
    node_location.at[node, 'lat'] = node_lat[node]
    node_location.at[node, 'alt'] = node_alt[node]

node_location = node_location.reset_index()
node_location.to_csv(input_data_path / "NodeLocations.csv", sep=';', index=False)

print("🏭 Configuring regional technologies...")

# Northwest technological configuration
with open(input_data_path / "period1" / "node_data" / "northwest" / "Technologies.json", "r") as json_file:
    technologies = json.load(json_file)
technologies["new"] = ["HeatPump_AirSourced", "Storage_Battery",
                       "Photovoltaic", "WindTurbine_Onshore_4000", "Storage_H2"]
technologies["existing"] = {"Hydro_Reservoir": 18500, "GasTurbine_simple": 12000,
                            "Boiler_Small_NG": 4500, "Photovoltaic": 3200, "WindTurbine_Onshore_4000": 800}

with open(input_data_path / "period1" / "node_data" / "northwest" / "Technologies.json", "w") as json_file:
    json.dump(technologies, json_file, indent=4)

# NorthEast technological configuration
with open(input_data_path / "period1" / "node_data" / "northeast" / "Technologies.json", "r") as json_file:
    technologies = json.load(json_file)
technologies["new"] = ["HeatPump_AirSourced", "Storage_Battery",
                       "Photovoltaic", "WindTurbine_Onshore_4000", "Storage_H2"]
technologies["existing"] = {"Hydro_Reservoir": 12800, "GasTurbine_simple": 8500,
                            "Boiler_Small_NG": 3800, "Photovoltaic": 2800, "WindTurbine_Onshore_4000": 600}

with open(input_data_path / "period1" / "node_data" / "northeast" / "Technologies.json", "w") as json_file:
    json.dump(technologies, json_file, indent=4)

# Center technological configuration
with open(input_data_path / "period1" / "node_data" / "center" / "Technologies.json", "r") as json_file:
    technologies = json.load(json_file)
technologies["new"] = ["HeatPump_AirSourced", "Storage_Battery",
                       "Photovoltaic", "WindTurbine_Onshore_4000", "Storage_H2"]
technologies["existing"] = {"Hydro_Reservoir": 8400, "GasTurbine_simple": 9200,
                            "Boiler_Small_NG": 3200, "Photovoltaic": 3800, "WindTurbine_Onshore_4000": 1400}

with open(input_data_path / "period1" / "node_data" / "center" / "Technologies.json", "w") as json_file:
    json.dump(technologies, json_file, indent=4)

# Add required technologies for node 'south'
with open(input_data_path / "period1" / "node_data" / "south" / "Technologies.json", "r") as json_file:
    technologies = json.load(json_file)
technologies["new"] = ["HeatPump_AirSourced", "Storage_Battery", "Photovoltaic", "WindTurbine_Onshore_4000",
                       "WindTurbine_Offshore_9500", "Storage_H2"]
technologies["existing"] = {"Hydro_Reservoir": 5200, "GasTurbine_simple": 7800,
                            "Boiler_Small_NG": 2400, "Photovoltaic": 6200, "WindTurbine_Onshore_4000": 3800}

with open(input_data_path / "period1" / "node_data" / "south" / "Technologies.json", "w") as json_file:
    json.dump(technologies, json_file, indent=4)

# Islands technological configuration
with open(input_data_path / "period1" / "node_data" / "islands" / "Technologies.json", "r") as json_file:
    technologies = json.load(json_file)
technologies["new"] = ["HeatPump_AirSourced", "Storage_Battery", "Photovoltaic", "WindTurbine_Onshore_4000",
                       "WindTurbine_Offshore_9500", "Storage_H2"]
technologies["existing"] = {"Hydro_Reservoir": 2100, "GasTurbine_simple": 3200,
                            "Boiler_Small_NG": 800, "Photovoltaic": 2600, "WindTurbine_Onshore_4000": 2100}

with open(input_data_path / "period1" / "node_data" / "islands" / "Technologies.json", "w") as json_file:
    json.dump(technologies, json_file, indent=4)

print("⚡ Copying technology data (using built-in database)...")
# Copy built-in technology data
adopt.copy_technology_data(input_data_path)

print("🌍 Loading climate data...")
# Load climate data for renewable resource assessment
adopt.load_climate_data_from_api(input_data_path)

print("💧 Adding hydro inflow data...")
# Import hydro inflow data for all nodes
import_hydro_inflows(input_data_path)

print("🔌 Setting up electricity networks...")
# Add networks
with open(input_data_path / "period1" / "Networks.json", "r") as json_file:
    networks = json.load(json_file)
networks["new"] = ["electricityOnshore"]
networks["existing"] = ["electricityOnshore"]

with open(input_data_path / "period1" / "Networks.json", "w") as json_file:
    json.dump(networks, json_file, indent=4)

# Create network topology (simplified version)
os.makedirs(input_data_path / "period1" / "network_topology" /
            "existing" / "electricityOnshore", exist_ok=True)

# Connection matrix
connection = pd.read_csv(input_data_path / "period1" / "network_topology" /
                         "existing" / "connection.csv", sep=";", index_col=0)
connection.loc["northwest", "northeast"] = 1
connection.loc["northeast", "northwest"] = 1
connection.loc["northwest", "center"] = 1
connection.loc["center", "northwest"] = 1
connection.loc["northeast", "center"] = 1
connection.loc["center", "northeast"] = 1
connection.loc["center", "south"] = 1
connection.loc["south", "center"] = 1
connection.loc["south", "islands"] = 1
connection.loc["islands", "south"] = 1
connection.to_csv(input_data_path / "period1" / "network_topology" /
                  "existing" / "electricityOnshore" / "connection.csv", sep=";")

# Distance matrix
distance = pd.read_csv(input_data_path / "period1" / "network_topology" /
                       "existing" / "distance.csv", sep=";", index_col=0)
distance.loc["northwest", "northeast"] = 350
distance.loc["northeast", "northwest"] = 350
distance.loc["northwest", "center"] = 450
distance.loc["center", "northwest"] = 450
distance.loc["northeast", "center"] = 420
distance.loc["center", "northeast"] = 420
distance.loc["center", "south"] = 380
distance.loc["south", "center"] = 380
distance.loc["south", "islands"] = 180
distance.loc["islands", "south"] = 180
distance.to_csv(input_data_path / "period1" / "network_topology" /
                "existing" / "electricityOnshore" / "distance.csv", sep=";")

# Size matrix (transmission capacity)
size = pd.read_csv(input_data_path / "period1" / "network_topology" /
                   "existing" / "size.csv", sep=";", index_col=0)
size.loc["northwest", "northeast"] = 6500
size.loc["northeast", "northwest"] = 6500
size.loc["northwest", "center"] = 8200
size.loc["center", "northwest"] = 8200
size.loc["northeast", "center"] = 5800
size.loc["center", "northeast"] = 5800
size.loc["center", "south"] = 6800
size.loc["south", "center"] = 6800
size.loc["south", "islands"] = 1000
size.loc["islands", "south"] = 1000
size.to_csv(input_data_path / "period1" / "network_topology" /
            "existing" / "electricityOnshore" / "size.csv", sep=";")

# Remove template files
os.remove(input_data_path / "period1" / "network_topology" / "existing" / "connection.csv")
os.remove(input_data_path / "period1" / "network_topology" / "existing" / "distance.csv")
os.remove(input_data_path / "period1" / "network_topology" / "existing" / "size.csv")

# Copy network data
print("📡 Copying network data...")
adopt.copy_network_data(input_data_path)

# Configure network economics
with open(input_data_path / "period1" / "network_data" / "electricityOnshore.json", "r") as json_file:
    network_data = json.load(json_file)

network_data["Economics"]["gamma2"] = 50000
network_data["Economics"]["gamma4"] = 400

with open(input_data_path / "period1" / "network_data" / "electricityOnshore.json", "w") as json_file:
    json.dump(network_data, json_file, indent=4)

print("📊 Setting up demand profiles...")
# Regional distribution based on actual Italian energy data
regional_annual_demand = {
    'northwest': 85143.204,   # MWh (scaled down for faster computation)
    'northeast': 68082.028,   # MWh
    'center': 52094.814,      # MWh
    'south': 39087.215,       # MWh
    'islands': 18050.764      # MWh
}

# Create basic demand profiles
for node in ['northwest', 'northeast', 'center', 'south', 'islands']:
    # Electricity demand
    el_demand = regional_annual_demand[node] / 8760  # Convert to average hourly
    
    # Heat demand (higher in north)
    heat_multiplier = {'northwest': 0.4, 'northeast': 0.35,
                       'center': 0.25, 'south': 0.15, 'islands': 0.1}
    heat_demand = el_demand * heat_multiplier[node]
    
    # Set demand
    adopt.fill_carrier_data(input_data_path, value_or_data=el_demand, columns=[
                            'Demand'], carriers=['electricity'], nodes=[node])
    adopt.fill_carrier_data(input_data_path, value_or_data=heat_demand, columns=[
                            'Demand'], carriers=['heat'], nodes=[node])

print("⛽ Setting up import constraints...")
# Import constraints based on Italian energy import data
import_constraints = {
    'northwest': {'gas_limit': 1500, 'electricity_limit': 400, 'electricity_price': 120, 'gas_price': 28},
    'northeast': {'gas_limit': 800, 'electricity_limit': 200, 'electricity_price': 115, 'gas_price': 30},
    'center': {'gas_limit': 500, 'electricity_limit': 100, 'electricity_price': 125, 'gas_price': 32},
    'south': {'gas_limit': 1200, 'electricity_limit': 50, 'electricity_price': 130, 'gas_price': 29},
    'islands': {'gas_limit': 300, 'electricity_limit': 0, 'electricity_price': 150, 'gas_price': 35}
}

# Apply import constraints
for node in ['northwest', 'northeast', 'center', 'south', 'islands']:
    constraints = import_constraints[node]
    
    # Set import limits
    adopt.fill_carrier_data(input_data_path, value_or_data=constraints['gas_limit'],
                            columns=['Import limit'], carriers=['gas'], nodes=[node])
    adopt.fill_carrier_data(input_data_path, value_or_data=constraints['electricity_limit'],
                            columns=['Import limit'], carriers=['electricity'], nodes=[node])
    
    # Set import prices
    adopt.fill_carrier_data(input_data_path, value_or_data=constraints['gas_price'],
                            columns=['Import price'], carriers=['gas'], nodes=[node])
    adopt.fill_carrier_data(input_data_path, value_or_data=constraints['electricity_price'],
                            columns=['Import price'], carriers=['electricity'], nodes=[node])

print("🚀 Running the Italian Energy System Optimization Model...")

try:
    # Run the optimization model
    m = adopt.ModelHub()
    print("✓ ModelHub created")
    
    m.read_data(input_data_path)
    print("✓ Data loaded successfully")
    
    print("⚡ Starting optimization (this may take a few minutes)...")
    m.quick_solve()
    print("🎉 Model solved successfully!")
    
    print("\n📈 RESULTS SUMMARY:")
    print("=" * 50)
    print(f"✓ Italian Energy Model Optimization Complete!")
    print(f"✓ Results saved to: {results_data_path}")
    print(f"✓ Input data used: {input_data_path}")
    print("=" * 50)
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()