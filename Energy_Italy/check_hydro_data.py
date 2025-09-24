import pandas as pd
import numpy as np

nodes = ['northwest', 'northeast', 'center', 'south', 'islands']

for node in nodes:
    df = pd.read_csv(
        f'./macro_decarbonisation/period1/node_data/{node}/ClimateData.csv', sep=';')
    hydro_col = df['Hydro_Reservoir_existing_inflow']
    print(f'{node}:')
    print(f'  Type: {type(hydro_col)}')
    print(f'  Has iloc: {hasattr(hydro_col, "iloc")}')
    print(f'  Mean: {hydro_col.mean():.2f}')
    print(f'  Length: {len(hydro_col)}')
    print(f'  Has NaN: {hydro_col.isna().any()}')
    print()
