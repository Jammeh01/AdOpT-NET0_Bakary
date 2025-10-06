import pandas as pd
from pathlib import Path
import json
import adopt_net0 as adopt
import pandas as pd

# Get the directory where the main.py script is located
script_dir = Path(__file__).parent

def import_hydro_inflows(input_data_path):
    """Importare i dati di flusso dei fiumi nei bacini idroelettrici. Per il pompaggio chiuso assumiamo 0
    visto che abbiamo assunto che tutti i flussi siano diretti ai bacini aperti"""
    # Assuming input_data_path is the AdOpT-NET0_Bakary directory
    data_path = script_dir / "macro_decarbonisation" / "Hydro inflows 2017.xlsx"
    
    # Check available sheet names first
    xl_file = pd.ExcelFile(data_path)
    print(f"Available sheets: {xl_file.sheet_names}")
    
    # Find the correct sheet name (handling potential spaces)
    target_sheet = None
    for sheet in xl_file.sheet_names:
        if "MacroRegion_Weekly_GWh" in sheet:
            target_sheet = sheet.strip()  # Remove any leading/trailing spaces
            break
    
    if target_sheet is None:
        raise ValueError(f"Could not find MacroRegion_Weekly_GWh sheet in {xl_file.sheet_names}")
    
    print(f"Using sheet: '{target_sheet}'")
    hydro_inflows = pd.read_excel(data_path, sheet_name=target_sheet, index_col=0)
    nodes = hydro_inflows.columns.tolist()
    hydro_inflows_hourly = pd.DataFrame(index=range(0, 8760), columns=nodes)

    for node in nodes:
        for week in range(0,52):
            start_hour = week * 168
            end_hour = (week + 1) * 168
            #Convert from GWh/week to MWh/h
            hydro_inflows_hourly.loc[start_hour:end_hour - 1, node] = hydro_inflows.loc[week+1, node]*1000/168
            #add last day of the year manually
            hydro_inflows_hourly.loc[8736:8760, node] = hydro_inflows.loc[52, node]


    for node in nodes:
        climate_data_file = (
                input_data_path / "period1" / "node_data" / node / "ClimateData.csv"
        )
        # Read the CSV with semicolon separator
        climate_data = pd.read_csv(climate_data_file, sep=";")
        climate_data["Hydro_Reservoir_existing_inflow"] = hydro_inflows_hourly[node].values
        climate_data.to_csv(climate_data_file, index=False, sep=";")
