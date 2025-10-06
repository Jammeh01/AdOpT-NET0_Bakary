import adopt_net0 as adopt
from pathlib import Path
import json
import pandas as pd

# Get the directory where the script is located
script_dir = Path(__file__).parent

# Create a very minimal test case
test_data_path = script_dir / "test_minimal"
test_data_path.mkdir(parents=True, exist_ok=True)

print("Creating minimal test case...")

# Create templates
adopt.create_optimization_templates(test_data_path)

# Load and modify topology
with open(test_data_path / "Topology.json", "r") as f:
    topology = json.load(f)

# Use only one node and basic carriers
topology["nodes"] = ["test_node"]
topology["carriers"] = ["electricity"]
topology["investment_periods"] = ["period1"]

with open(test_data_path / "Topology.json", "w") as f:
    json.dump(topology, f, indent=4)

# Create input data structure
adopt.create_input_data_folder_template(test_data_path)

# Add node location data
node_location = pd.read_csv(test_data_path / "NodeLocations.csv", sep=';', index_col=0, header=0)
node_location.at['test_node', 'lon'] = 12.5  # Rome coordinates
node_location.at['test_node', 'lat'] = 41.9
node_location.at['test_node', 'alt'] = 20
node_location = node_location.reset_index()
node_location.to_csv(test_data_path / "NodeLocations.csv", sep=';', index=False)

# Use only one very basic technology
with open(test_data_path / "period1" / "node_data" / "test_node" / "Technologies.json", "r") as f:
    technologies = json.load(f)

technologies["new"] = ["Photovoltaic"]  # Start with just one simple technology
technologies["existing"] = {}

with open(test_data_path / "period1" / "node_data" / "test_node" / "Technologies.json", "w") as f:
    json.dump(technologies, f, indent=4)

# Copy just the Photovoltaic technology
path_files_technologies = script_dir / "files_technologies"
try:
    adopt.copy_technology_data(test_data_path, path_files_technologies)
    print("Technology data copied successfully")
except Exception as e:
    print(f"Error copying technology data: {e}")

# Set minimal carrier data
try:
    adopt.fill_carrier_data(test_data_path, value_or_data=100, columns=['Demand'], 
                           carriers=['electricity'], nodes=['test_node'])
    print("Carrier data filled successfully")
except Exception as e:
    print(f"Error filling carrier data: {e}")

# Test reading data
print("Testing data read...")
try:
    m = adopt.ModelHub()
    m.read_data(test_data_path)
    print("SUCCESS: Data read successfully!")
    
    # Try quick solve
    print("Testing quick solve...")
    m.quick_solve()
    print("SUCCESS: Model solved successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()