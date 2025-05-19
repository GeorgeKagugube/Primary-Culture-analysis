# Starter Python notebook for analysing Fura-2 AM calcium imaging data from a dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.integrate import simpson
from scipy.optimize import curve_fit, OptimizeWarning
import seaborn as sns
import warnings

##Path to save files and figures to 
path = '''/Users/gwk/Desktop/PhD/Data/PhD_data/March_03_25_Final_Analysis/Figure/Primary_cell_culture/Representative figures/'''

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df
    
def compute_mean(data):
    data['Mean'] = data.iloc[:, [i for i in range(1,len(data.columns))]].mean(axis=1)
    return data

def compute_std(data):
    data['std'] = data.iloc[:, [i for i in range(3,len(data.columns)-1)]].std(axis=1)
    return data

def insertTime(df):
    return df.insert(1, 'Time', (df['Time (ms)']/1000))

def cleanVarNames(df):
    df.drop([var for var in df.columns if var[0] == 'U'],axis = 1, inplace = True)
    df.drop(['Frame','Time (ms)'], axis=1, inplace=True)
    return df

def plot_fura2_rhodamine(time, fura2_trace,fura2_trace2, rhodamine_trace,rhodamine_trace2, position='upper left'):
    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Plot Fura-2 AM on the left y-axis
    ax1.plot(time, fura2_trace, color='blue', label='Fura-2 AM (Ca²⁺)', linestyle='-.')
    ax1.plot(time, fura2_trace2, color='green', label='Fura-2 AM (Ca²⁺)', linestyle='-.')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Fura-2 AM Ratio (Ca²⁺)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')


    # Create a second y-axis for Rhodamine 123
    ax2 = ax1.twinx()
    ax2.plot(time, rhodamine_trace, color='blue', label='Rhodamine 123 (ΔΨm)')
    ax2.plot(time, rhodamine_trace2, color='green', label='Rhodamine 123 (ΔΨm)')
    ax2.set_ylabel('Rhodamine 123 (ΔΨm)', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Title and combined legend
    #plt.title(title)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc=position)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path+'/calcium_rhodamine123_mut.png')
    plt.show()

# Starter Python notebook for analyzing Fura-2 AM calcium imaging data from a dataframe
# --- Step 1: Load the data ---
# Assume a CSV file where:
# - Column 0: time (seconds)
# - Column 1 onwards: individual cells' Fura-2 AM fluorescence ratios
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df
    
# --- Step 2: Analyze a single trace with multiple peaks ---
def analyze_single_trace(time, trace, stim_regions=None, plot=True):
    results = []

    baseline_end_idx = int(0.1 * len(trace))
    baseline = np.mean(trace[:baseline_end_idx])
    noise_std = np.std(trace[:baseline_end_idx])

    peaks, _ = find_peaks(trace, height=baseline + 3 * noise_std, distance=10)

    def exp_decay(t, A, tau, C):
        return A * np.exp(-t / tau) + C

    for i, peak_idx in enumerate(peaks):
        peak_time = time[peak_idx]
        peak = trace[peak_idx]
        delta_ca = peak - baseline/baseline
        time_to_peak = peak_time - time[0]

        if stim_regions is not None:
            region = next(((start, end) for start, end in stim_regions if start <= peak_time <= end), (time[0], time[-1]))
            stim_start, stim_end = region
        else:
            stim_start = time[max(0, peak_idx - 10)]
            #stim_end = time[min(len(time)-1, peak_idx + 30)]
            stim_end = time[min(len(time)-1, peak_idx)]

        stim_mask = (time >= stim_start) & (time <= stim_end)
        auc = simpson(trace[stim_mask] - baseline, dx=(time[1] - time[0]))

        decay_tau = np.nan
        if peak_idx < len(trace) - 5:
            decay_time = time[peak_idx:] - time[peak_idx]
            decay_trace = trace[peak_idx:]
            if np.any(np.diff(decay_trace) < 0):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", OptimizeWarning)
                        popt, _ = curve_fit(
                            exp_decay,
                            decay_time,
                            decay_trace,
                            p0=(delta_ca, 1, baseline),
                            bounds=([0, 0.001, 0], [np.inf, 100, np.max(decay_trace)])
                        )
                        decay_tau = popt[1]
                except (RuntimeError, ValueError):
                    pass

        snr = delta_ca / noise_std if noise_std else np.nan

        duration = np.nan
        max_upstroke = np.nan
        max_downstroke = np.nan
        repolarisation_slope = np.nan

        if i == 0:
            window_size = 20
            start_idx = max(0, peak_idx - window_size)
            end_idx = min(len(trace) - 1, peak_idx + window_size)
            sub_time = time[start_idx:end_idx+1]
            sub_trace = trace[start_idx:end_idx+1]

            half_max = baseline + 0.5 * delta_ca
            above_half = np.where(sub_trace >= half_max)[0]
            if len(above_half) >= 2:
                duration = sub_time[above_half[-1]] - sub_time[above_half[0]]

            dt = np.diff(sub_time)
            dy = np.diff(sub_trace)
            slopes = dy / dt
            peak_pos = peak_idx - start_idx
            max_upstroke = np.max(slopes[:peak_pos]) if peak_pos > 0 else np.nan
            max_downstroke = np.min(slopes[peak_pos:]) if peak_pos < len(slopes) else np.nan

            # Estimate repolarisation slope (mean slope after the peak over 10 points)
            if peak_pos + 10 < len(slopes):
                repolarisation_slope = np.mean(slopes[peak_pos:peak_pos + 10])

        result = {
            'Peak_Index': peak_idx,
            'Peak_Time': peak_time,
            'Peak': peak,
            'Delta_Ca': delta_ca,
            'Time_to_Peak': time_to_peak,
            'AUC': auc,
            'Decay_Tau': decay_tau,
            'SNR': snr,
            'Duration': duration,
            'Max_Upstroke': max_upstroke,
            'Max_Downstroke': max_downstroke,
            'Repolarisation_Slope': repolarisation_slope
        }
        results.append(result)

        if plot:
            plt.figure(figsize=(10, 4))
            plt.plot(time, trace, label='Calcium Trace')
            plt.axhline(baseline, color='gray', linestyle='--', label='Baseline')
            plt.scatter(time[peak_idx], peak, color='red', label='Peak')
            plt.axvspan(stim_start, stim_end, color='yellow', alpha=0.2, label='Stimulus Region')
            plt.xlabel('Time (s)')
            plt.ylabel('Fura-2 Ratio')
            plt.title(f'Peak {i+1} Analysis')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    return pd.DataFrame(results)

# --- Step 3: Analyze all cells ---
def analyze_all_cells(df, stim_regions=None, plot_each=False):
    time = df.iloc[:, 0].values
    all_results = []

    for cell in df.columns[1:]:
        trace = df[cell].values
        cell_results = analyze_single_trace(time, trace, stim_regions=stim_regions, plot=plot_each)
        cell_results['Cell'] = cell
        all_results.append(cell_results)

    return pd.concat(all_results, ignore_index=True)
## Calcium 
#het_cal = 
wt_cal = pd.read_csv("../../../Analysis/Round1_6mice/csvfiles/M4_Fura-2_Calcium.csv")
hom_cal1 = pd.read_csv("../../../Analysis/Round1_6mice/csvfiles/M3_Fura-2_Calcium.csv")
hom_cal2 = pd.read_csv("../../../Analysis/Round1_6mice/csvfiles/M2_Fura-2_Calcium.csv")

## These have a poor signal and therefore will not be considered in the analysis with WT, but maybe with hets data
hom_cal3 = pd.read_csv("../../../Analysis/Round2-5mice/csvfiles/M2_Fura-2_Ratio.csv")
hom_cal4 = pd.read_csv("../../../Analysis/Round2-5mice/csvfiles/M5_Fura-2_Ratio.csv")

## Mitochondrial membrane potentials
#het_mm = 
wt_mm = pd.read_csv("../../../Analysis/Round1_6mice/csvfiles/M4_Rhod123_MMP.csv")
hom_mm1 = pd.read_csv("../../../Analysis/Round1_6mice/csvfiles/M3_Rhod123_MMP.csv")
hom_mm2 = pd.read_csv("../../../Analysis/Round1_6mice/csvfiles/M2_Rhod123_MMP.csv")

## Heterozygous 
het_cal1 = pd.read_csv("../../27_11_2024/csvfiles/mp K F2 Rh123 Glut ATP fccp 1 ctl astrocytes v1 calcium.csv")
het_cal2 = pd.read_csv("../../27_11_2024/csvfiles/mp K F2 Rh123 Glut ATP fccp 1 ctl astrocytes v1 rhodamin123.csv")

time = wt_cal['Time']
fura2 = wt_cal['Mean Intensity .5']
rhod = wt_mm['490/10(466.1) Mean Intensity .5']
plot_fura2_rhodamine(time,fura2_trace=fura2, rhodamine_trace=rhod)

## Calcium signal
insertTime(wt_cal)
insertTime(hom_cal1)
insertTime(hom_cal2)

## Mitochondrial membrane potential signal
insertTime(wt_mm)
insertTime(hom_mm1)
insertTime(hom_mm2)

## Clean the variable names here
cleanVarNames(wt_cal)
cleanVarNames(hom_cal1)
cleanVarNames(hom_cal2)

## Mitochondrial membrane potential datasets
cleanVarNames(wt_mm)
cleanVarNames(hom_mm1)
cleanVarNames(hom_mm2)

print('')

stim_regions = [(650,1000)]
summary_results = analyze_all_cells(newdf, stim_regions=stim_regions, plot_each=True)

fdata = pd.DataFrame(summary_results)
fdata
fdata.drop(['SNR','Decay_Tau'], axis=1, inplace=True)
fdata.dropna(inplace=True)
fdata.to_csv('../../28_11_2024/Morning_imaging_1_5_No_Mn/csv_files/calciumDynamics.csv')



