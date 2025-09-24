import pandas as pd
from pathlib import Path
import json
import adopt_net0 as adopt
import pandas as pd

def import_hydro_inflows(input_data_path):
    data_path = Path("./Hydro/hydro inflow.xlsx")
    hydro_inflows =  pd.read_excel(data_path, sheet_name="MacroRegion_Weekly_GWh", index_col=0)
    nodes = hydro_inflows.columns.tolist()
    hydro_inflows_hourly = pd.DataFrame(index=range(0, 8760), columns=nodes)

    for node in nodes:
        for week in range(0,52):
            start_hour = week * 168
            end_hour = (week + 1) * 168
            hydro_inflows_hourly.loc[start_hour:end_hour - 1, node] = hydro_inflows.loc[week+1, node]*1000/168
            hydro_inflows_hourly.loc[8736:8759, node] = hydro_inflows.loc[52, node]


    for node in nodes:
        climate_data_file = (
                input_data_path / "period1" / "node_data" / node / "ClimateData.csv"
        )
        climate_data = pd.read_csv(climate_data_file)
        climate_data["Hydro_Reservoir_existing_inflow"] = hydro_inflows_hourly[node].values
        climate_data.to_csv(climate_data_file, index=False, sep=";")