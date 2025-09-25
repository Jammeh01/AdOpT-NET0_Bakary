import pandas as pd
from pathlib import Path
import json
import adopt_net0 as adopt
import pandas as pd


def import_hydro_inflows(input_data_path):
    data_path = Path("./Hydro/hydro inflow.xlsx")
    hydro_inflows = pd.read_excel(
        data_path, sheet_name="MacroRegion_Weekly_GWh", index_col=0)
    nodes = hydro_inflows.columns.tolist()
    hydro_inflows_hourly = pd.DataFrame(index=range(0, 8760), columns=nodes)

    # Fill hourly data from weekly data
    for node in nodes:
        for week in range(0, 52):
            start_hour = week * 168
            end_hour = (week + 1) * 168
            if end_hour > 8760:
                end_hour = 8760
            hydro_inflows_hourly.loc[start_hour:end_hour - 1,
                                     node] = hydro_inflows.loc[week+1, node] * 1000 / 168

        # Fill remaining hours (8736:8759) with last week's data
        if 8736 < 8760:
            hydro_inflows_hourly.loc[8736:8759,
                                     node] = hydro_inflows.loc[52, node] * 1000 / 168

    # Ensure all NaN values are filled
    hydro_inflows_hourly = hydro_inflows_hourly.ffill().bfill()

    for model_node in ['northwest', 'northeast', 'center', 'south', 'islands']:
        climate_data_file = input_data_path / "period1" / \
            "node_data" / model_node / "ClimateData.csv"

        try:
            # Read the climate data
            climate_data = pd.read_csv(climate_data_file, sep=";")

            # Get hydro inflow data for this node
            if model_node in hydro_inflows_hourly.columns:
                hydro_series = hydro_inflows_hourly[model_node].copy()
            else:
                print(
                    f"Warning: No specific hydro data for {model_node}, using average")
                hydro_series = hydro_inflows_hourly.mean(axis=1)

            # Ensure the series has the correct length and no NaN values
            if len(hydro_series) != 8760:
                print(
                    f"Warning: Hydro data length mismatch for {model_node}: {len(hydro_series)} vs 8760")
                hydro_series = hydro_series.reindex(
                    range(8760), method='ffill')

            # Ensure hydro data maintains pandas Series format with proper index
            # This is required for AdOpT-NET0's .iloc indexing in hydro_open.py
            hydro_series.index = range(len(hydro_series))

            # Add to climate data while preserving Series format
            climate_data["Hydro_Reservoir_existing_inflow"] = hydro_series

            # Save with proper format
            climate_data.to_csv(climate_data_file, index=False, sep=";")
            print(f"Successfully updated hydro inflow data for {model_node}")

        except Exception as e:
            print(f"Error processing climate data for {model_node}: {e}")
            continue
