import adopt_net0 as adopt
import json
from pathlib import Path
import os
import pandas as pd
import numpy as np
from Hydro.data import *
from adopt_net0.data_preprocessing import load_climate_data_from_api

# Create folders
results_data_path = Path("./userData")
results_data_path.mkdir(parents=True, exist_ok=True)

# create input data path and optimization templates
input_data_path = Path("./macro_decarbonisation")
input_data_path.mkdir(parents=True, exist_ok=True)

# Create template input JSONs
adopt.create_optimization_templates(input_data_path)
path_files_technologies = Path("./files_technologies")

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

adopt.create_input_data_folder_template(input_data_path)

# Define node locations based on actual Italian regional energy centers (2021 data)
# Actual coordinates of major Italian energy hubs (based on Terna grid structure)
node_location = pd.read_csv(
    input_data_path / "NodeLocations.csv", sep=';', index_col=0, header=0)
node_lon = {'northwest': 9.2, 'northeast': 11.9, 'center': 12.5,
            'south': 14.8, 'islands': 12.5}  # longitude in degrees
node_lat = {'northwest': 45.4, 'northeast': 45.5, 'center': 42.8,
            'south': 40.8, 'islands': 38.0}  # latitude in degrees
# Elevation in meters #this neeed also needed to be changhe to 0 the alti. of Islands.
node_alt = {'northwest': 120, 'northeast': 50,
            'center': 250, 'south': 200, 'islands': 0}
for node in ['northwest', 'northeast', 'center', 'south', 'islands']:
    node_location.at[node, 'lon'] = node_lon[node]
    node_location.at[node, 'lat'] = node_lat[node]
    node_location.at[node, 'alt'] = node_alt[node]

node_location = node_location.reset_index()
node_location.to_csv(input_data_path / "NodeLocations.csv",
                     sep=';', index=False)

adopt.show_available_technologies()

# Add required technologies for node 'North' (We will be changing this and also adding new technologies like CCS)      (Waste-to-energy)

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
                       # WindTurbine_offshore  and also municipality boilers
                       "WindTurbine_Offshore_9500", "Storage_H2"]
technologies["existing"] = {"Hydro_Reservoir": 5200, "GasTurbine_simple": 7800,
                            "Boiler_Small_NG": 2400, "Photovoltaic": 6200, "WindTurbine_Onshore_4000": 3800}  # MWth and MWe

with open(input_data_path / "period1" / "node_data" / "south" / "Technologies.json", "w") as json_file:
    json.dump(technologies, json_file, indent=4)


# ISlands technological configuration
with open(input_data_path / "period1" / "node_data" / "islands" / "Technologies.json", "r") as json_file:
    technologies = json.load(json_file)
technologies["new"] = ["HeatPump_AirSourced", "Storage_Battery", "Photovoltaic", "WindTurbine_Onshore_4000",
                       # WindTurbine_offshore  and also municipality boilers
                       "WindTurbine_Offshore_9500", "Storage_H2"]
technologies["existing"] = {"Hydro_Reservoir": 2100, "GasTurbine_simple": 3200,
                            "Boiler_Small_NG": 800, "Photovoltaic": 2600, "WindTurbine_Onshore_4000": 2100}  # MWth and MWe

with open(input_data_path / "period1" / "node_data" / "islands" / "Technologies.json", "w") as json_file:
    json.dump(technologies, json_file, indent=4)


# Copy over technology files
adopt.copy_technology_data(input_data_path, path_files_technologies)

adopt.show_available_networks()

# Add networks (here we will add an onshore electricity network, liket the one in the case study as an example)
with open(input_data_path / "period1" / "Networks.json", "r") as json_file:
    networks = json.load(json_file)
networks["new"] = ["electricityOnshore"]
networks["existing"] = ["electricityOnshore"]

with open(input_data_path / "period1" / "Networks.json", "w") as json_file:
    json.dump(networks, json_file, indent=4)

# === Make a new folder for the existing network (We can also create the dataframe for this and import it.)
os.makedirs(input_data_path / "period1" / "network_topology" /
            "existing" / "electricityOnshore", exist_ok=True)

print("Existing network")

# === Connection (Existing)
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
connection.loc["south", "islands"] = 1  # Sicily connection
connection.loc["islands", "south"] = 1
connection.to_csv(input_data_path / "period1" / "network_topology" /
                  "existing" / "electricityOnshore" / "connection.csv", sep=";")
print("Connection:", connection)

# Delete the original template
os.remove(input_data_path / "period1" /
          "network_topology" / "existing" / "connection.csv")

# === Distance (Existing)
distance = pd.read_csv(input_data_path / "period1" / "network_topology" /
                       "existing" / "distance.csv", sep=";", index_col=0)
distance.loc["northwest", "northeast"] = 350  # milan-venice corridor
distance.loc["northeast", "northwest"] = 350
distance.loc["northwest", "center"] = 450    # Milan-Rome corridor
distance.loc["center", "northwest"] = 450
distance.loc["northeast", "center"] = 420     # Venice-Rome corridor
distance.loc["center", "northeast"] = 420
distance.loc["center", "south"] = 380         # Rome-Naples corridor
distance.loc["south", "center"] = 380
distance.loc["south", "islands"] = 180  # sicily cables (SA.PE.I + SA.CO.I)
distance.loc["islands", "south"] = 180
distance.to_csv(input_data_path / "period1" / "network_topology" /
                "existing" / "electricityOnshore" / "distance.csv", sep=";")
print("Distance:", distance)

# Delete the original template
os.remove(input_data_path / "period1" /
          "network_topology" / "existing" / "distance.csv")

# === Size (Existing)
size = pd.read_csv(input_data_path / "period1" / "network_topology" /
                   "existing" / "size.csv", sep=";", index_col=0)
size.loc["northwest", "northeast"] = 6500   # Po Valley 380kV lines
size.loc["northeast", "northwest"] = 6500
size.loc["northwest", "center"] = 8200     # North-Central 380kV corridor
size.loc["center", "northwest"] = 8200
size.loc["northeast", "center"] = 5800     # Adriatic corridor
size.loc["center", "northeast"] = 5800
size.loc["center", "south"] = 6800        # Central-South lines
size.loc["south", "center"] = 6800
# Sicily cables (SA.PE.I 500MW + SA.CO.I 500MW)
size.loc["south", "islands"] = 1000
size.loc["islands", "south"] = 1000
size.to_csv(input_data_path / "period1" / "network_topology" /
            "existing" / "electricityOnshore" / "size.csv", sep=";")
print("Size:", size)

# Delete the original template
os.remove(input_data_path / "period1" /
          "network_topology" / "existing" / "size.csv")


print("New network")
# === Make a new folder for the new network
os.makedirs(input_data_path / "period1" / "network_topology" /
            "new" / "electricityOnshore", exist_ok=True)

# === Max Size Arc (New)
arc_size = pd.read_csv(input_data_path / "period1" / "network_topology" /
                       "new" / "size_max_arcs.csv", sep=";", index_col=0)
arc_size.loc["northwest", "northeast"] = 10000  # Po Valley reinforcement
arc_size.loc["northeast", "northwest"] = 10000
arc_size.loc["northwest", "center"] = 12000   # North-Central reinforcement
arc_size.loc["center", "northwest"] = 12000
arc_size.loc["northeast", "center"] = 8000  # Adriatic reinforcement
arc_size.loc["center", "northeast"] = 8000
arc_size.loc["center", "south"] = 10000    # Central-South reinforcement
arc_size.loc["south", "center"] = 10000
arc_size.loc["south", "islands"] = 2000    # Additional island connections
arc_size.loc["islands", "south"] = 2000
arc_size.loc["northwest", "south"] = 8000  # Direct North-South bypass
arc_size.loc["south", "northwest"] = 8000
arc_size.loc["northeast", "south"] = 7000  # Northeast-South direct
arc_size.loc["south", "northeast"] = 7000
arc_size.loc["center", "islands"] = 5000   # Central-Islands direct
arc_size.loc["islands", "center"] = 5000
arc_size.to_csv(input_data_path / "period1" / "network_topology" /
                "new" / "electricityOnshore" / "size_max_arcs.csv", sep=";")
print("Max size per arc:", arc_size)

# === Connection (New)
connection = pd.read_csv(input_data_path / "period1" /
                         "network_topology" / "new" / "connection.csv", sep=";", index_col=0)
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
connection.loc["northwest", "south"] = 1
connection.loc["south", "northwest"] = 1
connection.loc["northwest", "islands"] = 1
connection.loc["islands", "northwest"] = 1
connection.loc["northeast", "south"] = 1
connection.loc["south", "northeast"] = 1
connection.loc["northeast", "islands"] = 1
connection.loc["islands", "northeast"] = 1
connection.loc["center", "islands"] = 1
connection.loc["islands", "center"] = 1
connection.to_csv(input_data_path / "period1" / "network_topology" /
                  "new" / "electricityOnshore" / "connection.csv", sep=";")
print("Connection:", connection)

# Delete connection template
os.remove(input_data_path / "period1" /
          "network_topology" / "new" / "connection.csv")

# === Distance (New)
distance = pd.read_csv(input_data_path / "period1" /
                       "network_topology" / "new" / "distance.csv", sep=";", index_col=0)
distance.loc["northwest", "northeast"] = 350   # Turin/Milan to Venice/Trieste
distance.loc["northeast", "northwest"] = 350
distance.loc["northwest", "center"] = 450      # Turin/Milan to Rome
distance.loc["center", "northwest"] = 450
distance.loc["northwest", "south"] = 750       # Turin/Milan to Naples/Bari
distance.loc["south", "northwest"] = 750
distance.loc["northwest", "islands"] = 950    # Turin/Milan to Palermo/Cagliari
distance.loc["islands", "northwest"] = 950
distance.loc["northeast", "center"] = 420      # Venice to Rome
distance.loc["center", "northeast"] = 420
distance.loc["northeast", "south"] = 680       # Venice to Naples/Bari
distance.loc["south", "northeast"] = 680
distance.loc["northeast", "islands"] = 880    # Venice to Palermo/Cagliari
distance.loc["islands", "northeast"] = 880
distance.loc["center", "south"] = 380          # Rome to Naples
distance.loc["south", "center"] = 380
distance.loc["center", "islands"] = 520        # Rome to Palermo/Cagliari
distance.loc["islands", "center"] = 520
# Naples/Bari to Palermo/Cagliari
distance.loc["south", "islands"] = 180
distance.loc["islands", "south"] = 180
distance.to_csv(input_data_path / "period1" / "network_topology" /
                "new" / "electricityOnshore" / "distance.csv", sep=";")
print("Distance:", distance)

# Delete distance template
os.remove(input_data_path / "period1" /
          "network_topology" / "new" / "distance.csv")

# Delete size_max_arcs template
os.remove(input_data_path / "period1" /
          "network_topology" / "new" / "size_max_arcs.csv")

adopt.copy_network_data(input_data_path)

with open(input_data_path / "period1" / "network_data" / "electricityOnshore.json", "r") as json_file:
    network_data = json.load(json_file)

network_data["Economics"]["gamma2"] = 50000
network_data["Economics"]["gamma4"] = 400

with open(input_data_path / "period1" / "network_data" / "electricityOnshore.json", "w") as json_file:
    json.dump(network_data, json_file, indent=4)

# Demand profiles and import constraints based on 2021 Italian energy data

# Regional distribution based on CGE model
regional_annual_demand = {
    'northwest': 8514320.4,   # MWh (Lombardy + Piedmont industrial)
    'northeast': 6808202.8,   # MWh (Veneto + Emilia industrial/agricultural)
    'center': 5209481.4,      # MWh (Lazio + Tuscany + others)
    'south': 3908721.5,       # MWh (Campania + Puglia + others)
    'islands': 1805076.4      # MWh (Sicily + Sardinia)
}

# Create hourly profiles (simplified seasonal/daily patterns)
hours = np.arange(8760)
base_profile = np.ones(8760)

# We add seasonal variation (winter heating, summer cooling)
seasonal = 0.15 * np.sin(2 * np.pi * hours / 8760 - np.pi/2)

# We add daily variation (peak during day, low at night)
daily = 0.3 * np.sin(2 * np.pi * (hours % 24) / 24 - np.pi/2)

hourly_data = {}
for node in ['northwest', 'northeast', 'center', 'south', 'islands']:
    # Electricity demand profile
    el_profile = base_profile + seasonal + daily
    el_profile = el_profile / \
        np.mean(el_profile) * regional_annual_demand[node] / 8760

    # Heat demand (higher in north, seasonal pattern)
    heat_multiplier = {'northwest': 0.4, 'northeast': 0.35,
                       'center': 0.25, 'south': 0.15, 'islands': 0.1}
    heat_seasonal = 0.6 * \
        np.maximum(0, -np.sin(2 * np.pi * hours / 8760 - np.pi/2))
    heat_profile = (base_profile * 0.2 + heat_seasonal) * \
        heat_multiplier[node] * regional_annual_demand[node] / 8760

    hourly_data[node] = pd.DataFrame({
        'electricity': el_profile,
        'heat': heat_profile
    })

# Fill carrier demand data for each region
for node in ['northwest', 'northeast', 'center', 'south', 'islands']:
    if 'hourly_data' in locals():
        el_demand = hourly_data[node]['electricity']
        heat_demand = hourly_data[node]['heat']
    else:
        el_demand = hourly_data[node].iloc[:, 1]
        heat_demand = hourly_data[node].iloc[:, ]

    adopt.fill_carrier_data(input_data_path, value_or_data=el_demand, columns=[
                            'Demand'], carriers=['electricity'], nodes=[node])
    adopt.fill_carrier_data(input_data_path, value_or_data=heat_demand, columns=[
                            'Demand'], carriers=['heat'], nodes=[node])


# 2021 Italian energy import data and cross-border capacities
import_constraints = {
    'northwest': {
        # High gas imports via Alpine pipelines (TAP, etc.)
        'gas_limit': 15000,
        'electricity_limit': 4000,  # Imports from France/Switzerland
        'electricity_price': 120,   # EUR/MWh average 2021
        'gas_price': 28             # EUR/MWh average 2021
    },
    'northeast': {
        'gas_limit': 8000,       # Gas from Eastern Europe
        'electricity_limit': 2000,  # Imports from Austria/Slovenia
        'electricity_price': 115,
        'gas_price': 30
    },
    'center': {
        'gas_limit': 5000,       # Limited direct gas imports
        'electricity_limit': 1000,  # Limited cross-border
        'electricity_price': 125,
        'gas_price': 32
    },
    'south': {
        'gas_limit': 12000,      # TAP pipeline, LNG terminals
        'electricity_limit': 500,   # Limited cross-border
        'electricity_price': 130,
        'gas_price': 29
    },
    'islands': {
        'gas_limit': 3000,       # LNG terminals (limited)
        'electricity_limit': 0,     # No direct imports (island systems)
        'electricity_price': 150,   # Higher island prices
        'gas_price': 35
    }
}

# Apply import constraints and pricing
for node in ['northwest', 'northeast', 'center', 'south', 'islands']:
    constraints = import_constraints[node]

    # Set import limits (MW for electricity, MW equivalent for gas)
    adopt.fill_carrier_data(input_data_path, value_or_data=constraints['gas_limit'],
                            columns=['Import limit'], carriers=['gas'], nodes=[node])
    adopt.fill_carrier_data(input_data_path, value_or_data=constraints['electricity_limit'],
                            columns=['Import limit'], carriers=['electricity'], nodes=[node])

    # Set import prices (EUR/MWh)
    adopt.fill_carrier_data(input_data_path, value_or_data=constraints['gas_price'],
                            columns=['Import price'], carriers=['gas'], nodes=[node])
    adopt.fill_carrier_data(input_data_path, value_or_data=constraints['electricity_price'],
                            columns=['Import price'], carriers=['electricity'], nodes=[node])

    # Set emission factors
    adopt.fill_carrier_data(input_data_path, value_or_data=0.35,
                            columns=['Import emission factor'], carriers=['electricity'], nodes=[node])
    adopt.fill_carrier_data(input_data_path, value_or_data=0.2,
                            columns=['Import emission factor'], carriers=['gas'], nodes=[node])

# Load climate data for renewable resource assessment
print("Loading fresh climate data...")
adopt.load_climate_data_from_api(input_data_path)

# Import hydro inflow data for all nodes
print("Adding hydro inflow data...")
import_hydro_inflows(input_data_path)
print("Hydro inflow data added successfully")

# Run the optimization model with the baseline scenario (no ETS)
m = adopt.ModelHub()
m.read_data(input_data_path)
m.quick_solve()
