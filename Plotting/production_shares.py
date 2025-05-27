import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline

from adopt_net0 import extract_datasets_from_h5group

# Define the data paths
RESULT_FOLDER = "Z:/AdOpt_NET0/AdOpt_results/MY/"
DATA_TO_EXCEL_PATH1 = 'C:/EHubversions/AdOpT-NET0_Julia//Plotting/production_shares_olefins.xlsx'
DATA_TO_EXCEL_PATH2 = 'C:/EHubversions/AdOpT-NET0_Julia//Plotting/production_shares_ammonia.xlsx'
DATAPATH = "C:/EHubversions/AdOpT-NET0_Julia/Plotting"


def fetch_and_process_data_production(resultfolder, data_to_excel_path_olefins, data_to_excel_path_ammonia,
                                      result_types, tec_mapping, categories):
    all_results = []
    olefin_results = []
    ammonia_results = []

    for result_type in result_types:
        resultfolder_type = f"{resultfolder}{result_type}"

        columns = pd.MultiIndex.from_product(
            [[str(result_type)], ["Chemelot"], ['2030', '2040', '2050']],
            names=["Resulttype", "Location", "Interval"]
        )

        result_data = pd.DataFrame(0.0, index=tec_mapping.keys(), columns=columns)
        production_sum_olefins = pd.DataFrame(0.0, index=categories.keys(), columns=columns)
        production_sum_ammonia = pd.DataFrame(0.0, index=categories.keys(), columns=columns)

        for location in result_data.columns.levels[1]:
            folder_name = f"{location}"
            summarypath = os.path.join(resultfolder_type, folder_name, "Summary.xlsx")

            try:
                summary_results = pd.read_excel(summarypath)
            except FileNotFoundError:
                print(f"Warning: Summary file not found for {result_type} - {location}")
                continue

            for interval in result_data.columns.levels[2]:
                for case in summary_results['case']:
                    if pd.notna(case) and interval in case:
                        h5_path = Path(summary_results[summary_results['case'] == case].iloc[0][
                                           'time_stamp']) / "optimization_results.h5"
                        if h5_path.exists():
                            with h5py.File(h5_path, "r") as hdf_file:
                                tec_operation = extract_datasets_from_h5group(
                                    hdf_file["operation/technology_operation"])
                                tec_operation = {k: v for k, v in tec_operation.items() if len(v) >= 8670}
                                df_tec_operation = pd.DataFrame(tec_operation)

                                for tec in tec_mapping.keys():
                                    para = tec_mapping[tec][2] + "_output"
                                    if (interval, location, tec, para) in df_tec_operation:
                                        output_car = df_tec_operation[interval, location, tec, para]

                                        if tec in ['CrackerFurnace', 'MPW2methanol', 'SteamReformer'] and (
                                                interval, location, tec, 'CO2captured_output') in df_tec_operation:
                                            numerator = df_tec_operation[
                                                interval, location, tec, 'CO2captured_output'].sum()
                                            denominator = (
                                                    df_tec_operation[
                                                        interval, location, tec, 'CO2captured_output'].sum()
                                                    + df_tec_operation[interval, location, tec, 'emissions_pos'].sum()
                                            )

                                            frac_CC = numerator / denominator if (
                                                    denominator > 1 and numerator > 1) else 0

                                            tec_CC = tec + "_CC"
                                            result_data.loc[tec, (result_type, location, interval)] = sum(
                                                output_car) * (1 - frac_CC)
                                            result_data.loc[tec_CC, (result_type, location, interval)] = sum(
                                                output_car) * frac_CC
                                        else:
                                            result_data.loc[tec, (result_type, location, interval)] = sum(output_car)

                                    tec_existing = tec + "_existing"
                                    if (interval, location, tec_existing, para) in df_tec_operation:
                                        output_car = df_tec_operation[interval, location, tec_existing, para]

                                        if tec in ['CrackerFurnace', 'MPW2methanol', 'SteamReformer'] and (
                                                interval, location, tec_existing,
                                                'CO2captured_output') in df_tec_operation:
                                            numerator = df_tec_operation[
                                                interval, location, tec_existing, 'CO2captured_output'].sum()
                                            denominator = (
                                                    df_tec_operation[
                                                        interval, location, tec_existing, 'CO2captured_output'].sum()
                                                    + df_tec_operation[
                                                        interval, location, tec_existing, 'emissions_pos'].sum()
                                            )

                                            frac_CC = numerator / denominator if (
                                                    denominator > 1 and numerator > 1) else 0

                                            tec_CC = tec + "_CC"
                                            result_data.loc[tec, (result_type, location, interval)] += sum(
                                                output_car) * (1 - frac_CC)
                                            result_data.loc[tec_CC, (result_type, location, interval)] += sum(
                                                output_car) * frac_CC
                                        else:
                                            result_data.loc[tec, (result_type, location, interval)] += sum(output_car)

                            for tec in tec_mapping.keys():
                                if tec_mapping[tec][0] == 'Olefin':
                                    olefin_production = result_data.loc[tec, (result_type, location, interval)] * \
                                                        tec_mapping[tec][3]
                                    production_sum_olefins.loc[
                                        tec_mapping[tec][1], (result_type, location, interval)] += olefin_production
                                elif tec_mapping[tec][0] == 'Ammonia':
                                    ammonia_production = result_data.loc[tec, (result_type, location, interval)] * \
                                                         tec_mapping[tec][3]
                                    production_sum_ammonia.loc[
                                        tec_mapping[tec][1], (result_type, location, interval)] += ammonia_production

        all_results.append(result_data)
        olefin_results.append(production_sum_olefins)
        ammonia_results.append(production_sum_ammonia)

    production_sum_olefins = pd.concat(olefin_results, axis=1)
    production_sum_olefins.to_excel(data_to_excel_path_olefins)
    production_sum_ammonia = pd.concat(ammonia_results, axis=1)
    production_sum_ammonia.to_excel(data_to_excel_path_ammonia)


def plot_production_shares(df, categories):
    df.columns.name = None
    df = df.T.reset_index()
    df = df.rename(columns={'index': 'Year'})
    df['Year'] = df['Year'].astype(int)

    years = df['Year'].values
    available_categories = [cat for cat in categories if cat in df.columns]
    df = df[available_categories]

    shares = df.div(df.sum(axis=1), axis=0)
    x_smooth = np.linspace(years.min(), years.max(), 300)

    interpolated = {}
    for col in available_categories:
        spline = make_interp_spline(years, shares[col], k=2)
        interpolated[col] = spline(x_smooth)

    y_stack = np.row_stack([interpolated[col] for col in available_categories])

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.stackplot(x_smooth, y_stack, labels=available_categories, colors=[categories[c] for c in available_categories])
    # ax.set_title("Share of Total Production by Technology")
    ax.set_ylabel("Share of Total Production")
    ax.set_xlabel("Year")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xlim(years.min(), years.max())
    ax.set_xticks([2030, 2040, 2050])
    ax.set_xticklabels([r"$2030$", r"$2040$", r"$2050$"])
    ax.set_ylim(0, 1)
    plt.rcParams['font.family'] = 'serif'
    plt.tight_layout()


def plot_production_shares_stacked(df1, df2, categories, interpolation="spline"):
    def preprocess(df):
        df.columns.name = None
        df = df.T.reset_index()
        df = df.rename(columns={'index': 'Year'})
        df['Year'] = df['Year'].astype(int)
        return df

    df1 = preprocess(df1)
    df2 = preprocess(df2)

    # Merge on 'Year' and sum technology columns
    merged = df1.copy()
    for cat in categories:
        if cat in df2.columns and cat in df1.columns:
            merged[cat] = df1[cat] + df2[cat]
        elif cat in df2.columns:
            merged[cat] = df2[cat]
        elif cat in df1.columns:
            merged[cat] = df1[cat]

    # Manually add row for Year 2025
    if 2025 not in merged['Year'].values:
        extra_row = {cat: 0 for cat in categories}
        extra_row['Conventional'] = 1
        extra_row['Year'] = 2025
        merged = pd.concat([merged, pd.DataFrame([extra_row])], ignore_index=True)
        merged = merged.sort_values('Year')

    years = merged['Year'].values
    available_categories = [cat for cat in categories if cat in merged.columns]
    df = merged[available_categories]

    shares = df.div(df.sum(axis=1), axis=0)
    x_smooth = np.linspace(years.min(), years.max(), 300)

    if interpolation == "spline":
        x = np.linspace(years.min(), years.max(), 300)
        interpolated = {}
        for col in available_categories:
            spline = make_interp_spline(years, shares[col], k=2)
            interpolated[col] = spline(x)
        y_stack = np.row_stack([interpolated[col] for col in available_categories])

    elif interpolation == "linear":
        x = years
        y_stack = np.row_stack([shares[col].values for col in available_categories])

    elif interpolation == "step":
        x = np.repeat(years, 2)[1:]
        y_stack = {}
        for col in available_categories:
            y = shares[col].values
            y_step = np.repeat(y, 2)[:-1]
            y_stack[col] = y_step

    else:
        raise ValueError(f"Unsupported interpolation method: {interpolation}")

    fig, ax = plt.subplots(figsize=(6, 3))

    if interpolation in ("spline", "linear"):
        ax.stackplot(x, y_stack, labels=available_categories,
                     colors=[categories[c] for c in available_categories])

    elif interpolation == "step":
        x = np.repeat(years, 2)[1:]

        # Build step-wise shares
        step_shares = []
        for col in available_categories:
            y = shares[col].values
            y_step = np.repeat(y, 2)[:-1]
            step_shares.append(y_step)

        # Stack cumulatively for area plot
        bottoms = np.zeros_like(step_shares[0])
        for i, col in enumerate(available_categories):
            top = bottoms + step_shares[i]
            ax.fill_between(x, bottoms, top, step='post',
                            color=categories[col], label=col)
            bottoms = top

    #Layout
    ax.set_ylabel("Share of Total Production")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xlim(years.min(), years.max())
    ax.set_xticks([2025, 2030, 2040, 2050])
    ax.set_xticklabels([r"Current", r"$2030$", r"$2040$", r"$2050$"])
    ax.set_ylim(0, 1)
    plt.rcParams['font.family'] = 'serif'
    plt.tight_layout()


def main():
    set_result_types = ['EmissionLimit Greenfield', 'EmissionLimit Brownfield']

    tec_mapping = {
        "CrackerFurnace": ("Olefin", "Conventional", "olefins", 0.439),
        "CrackerFurnace_CC": ("Olefin", "Carbon Capture", "olefins", 0.439),
        "CrackerFurnace_Electric": ("Olefin", "Electrification", "olefins", 0.439),
        "SteamReformer": ("Ammonia", "Conventional", "HBfeed", 0.168),
        "SteamReformer_CC": ("Ammonia", "Carbon Capture", "HBfeed", 0.168),
        "WGS_m": ("Ammonia", "Electrification", "hydrogen", 0.168),
        "AEC": ("Ammonia", "Electrification", "hydrogen", 0.168),
        "RWGS": ("Olefin", r"CO$_2$ utilization", "syngas", 0.270),
        "DirectMeOHsynthesis": ("Olefins", r"CO$_2$ utilization", "methanol", 0.328),
        "EDH": ("Olefin", "Bio-based feedstock", "ethylene", 1),
        "PDH": ("Olefin", "Bio-based feedstock", "propylene", 1),
        "MPW2methanol": ("Olefin", "Plastic waste recycling", "methanol", 0.328),
        "MPW2methanol_CC": ("Olefin", "Plastic waste recycling with CC", "methanol", 0.328),
        "CO2electrolysis": ("Olefin", r"CO$_2$ utilization", "ethylene", 1),
    }

    categories = {
        "Conventional": '#8C8B8B',
        "Carbon Capture": '#3E7EB0',
        "Electrification": '#EDD253',
        r"CO$_2$ utilization": '#E18826',
        "Bio-based feedstock": '#84AA6F',
        "Plastic waste recycling": '#B475B2',
        "Plastic waste recycling with CC": '#533A8C',
    }

    get_data = 0

    if get_data == 1:
        fetch_and_process_data_production(RESULT_FOLDER, DATA_TO_EXCEL_PATH1, DATA_TO_EXCEL_PATH2, set_result_types,
                                          tec_mapping, categories)

    production_sum_olefins = pd.read_excel(DATA_TO_EXCEL_PATH1, index_col=0, header=[0, 1, 2])
    production_sum_ammonia = pd.read_excel(DATA_TO_EXCEL_PATH2, index_col=0, header=[0, 1, 2])

    result_type = 'EmissionLimit Brownfield'
    location = 'Chemelot'
    product = "stacked"
    interpolation = "linear"

    if product == "Olefin":
        df_plot = production_sum_olefins.loc[:, (result_type, location)].copy()
        plot_production_shares(df_plot, categories)
    elif product == 'Ammonia':
        df_plot = production_sum_ammonia.loc[:, (result_type, location)].copy()
        plot_production_shares(df_plot, categories)
    elif product == 'stacked':
        df_plot_olefin = production_sum_olefins.loc[:, (result_type, location)].copy()
        df_plot_ammonia = production_sum_ammonia.loc[:, (result_type, location)].copy()
        plot_production_shares_stacked(df_plot_ammonia, df_plot_olefin, categories, interpolation=interpolation)

    #Make the plots
    ext_map = {'Brownfield': '_bf', 'Greenfield': '_gf'}
    ext = next((v for k, v in ext_map.items() if k in result_type), '')
    filename = f'production_share_{interpolation}{ext}'

    saveas = 'pdf'
    if saveas == 'svg':
        savepath = f'C:/Users/5637635/Documents/OneDrive - Universiteit Utrecht/Research/Multiyear Modeling/MY_Plots/{filename}.svg'
        plt.savefig(savepath, format='svg')
    elif saveas == 'pdf':
        savepath = f'C:/Users/5637635/Documents/OneDrive - Universiteit Utrecht/Research/Multiyear Modeling/MY_Plots/{filename}.pdf'
        plt.savefig(savepath, format='pdf')
    elif saveas == 'both':
        savepath = f'C:/Users/5637635/Documents/OneDrive - Universiteit Utrecht/Research/Multiyear Modeling/MY_Plots/{filename}.pdf'
        plt.savefig(savepath, format='pdf')
        savepath = f'C:/Users/5637635/Documents/OneDrive - Universiteit Utrecht/Research/Multiyear Modeling/MY_Plots/{filename}.svg'
        plt.savefig(savepath, format='svg')

    plt.show()


if __name__ == "__main__":
    main()
