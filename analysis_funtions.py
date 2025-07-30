import pandas as pd
import math
import matplotlib.pyplot as plt
import datetime
from scipy.fft import fft
import numpy as np
import os
import glob


def get_csv_file_paths(folder_path):
    """
    Get a list of CSV file paths from the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing CSV files
        
    Returns:
        list: List of CSV file paths
    """
    # Use glob to find all CSV files in the folder
    csv_pattern = os.path.join(folder_path, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    # Sort the files for consistent ordering
    csv_files.sort()
    
    return csv_files


def poop():
    print("poop")

def raw_csv_to_df_add_rms(file):

    df = pd.read_csv(file, skiprows=6)
    
    # Remove all quotes from the column names
    df.columns = df.columns.str.replace('"', '', regex=False)
    
    # Remove all rows after 2085
    df = df.iloc[:2086]
    
    # Remove all columns greater than 19
    df = df.iloc[:, :19]
    
    # Debug: print column names to see what we have
    print("Available columns:", df.columns.tolist())
    
    # Convert all data to string first to handle quotes, then to numeric
    for col in df.columns:
        if any(target in col for target in ['IA(A)', 'IB(A)', 'IC(A)', 'VA(V)', 'VB(V)', 'VC(V)']):
            # Convert to string, remove quotes, then convert to numeric
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('"', '').str.replace("'", ''), 
                                  errors='coerce')
            print(f"Converted {col}: min={df[col].min()}, max={df[col].max()}")
    
    # Add timestamp column
    df['timestamp'] = df.index / 1920
    df.insert(0, 'timestamp', df.pop('timestamp'))

    # Scale all column VA, VB and VC by sqrt(2)
    df['VA(V)'] = df['VA(V)'] * (2 ** 0.5)
    df['VB(V)'] = df['VB(V)'] * (2 ** 0.5)
    df['VC(V)'] = df['VC(V)'] * (2 ** 0.5)
    #current too
    df['IA(A)'] = df['IA(A)'] * (2 ** 0.5)
    df['IB(A)'] = df['IB(A)'] * (2 ** 0.5)
    df['IC(A)'] = df['IC(A)'] * (2 ** 0.5)

    rms_targets = {
        'VA(V)': 'RMS_VA_T',
        'VB(V)': 'RMS_VB_T',
        'VC(V)': 'RMS_VC_T',
        'IA(A)': 'RMS_IA_T',
        'IB(A)': 'RMS_IB_T',
        'IC(A)': 'RMS_IC_T',
    }

    # Initialize new columns
    for new_col in rms_targets.values():
        df[new_col] = None

    # Calculate RMS for each target column
    for col, new_col in rms_targets.items():
        if col in df.columns:  # Check if column exists
            print(f"Calculating RMS for {col}")
            for i in range(len(df)):
                if i < 32:
                    df.at[i, new_col] = None
                else:
                    total = 0
                    for j in range(i - 31, i + 1):
                        val = df.at[j, col]
                        if pd.notna(val):  # Check for NaN values
                            total += val**2
                    df.at[i, new_col] = (total / 32)**0.5

    return df

def plot_and_save(df, fault_time, num_cycles_pre, num_cycles_post, voltages, rms_voltages, currents, rms_currents, inv_power, rlc_power, trial, event_num):
    closest_index = (df['timestamp'] - fault_time).abs().idxmin()
    df2 = df.iloc[closest_index-(num_cycles_pre*32):closest_index+(num_cycles_post*32), :]
    
    # Convert timestamp to milliseconds and shift so fault time is at 0 ms
    df2 = df2.copy()
    df2['timestamp_ms'] = (df2['timestamp'] - fault_time) * 1000  # Convert to ms and shift
    
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create descriptive title and filename components
    title_suffix = f"Initial Inverter Power:{inv_power}kW, Load Power:{rlc_power}kW, Trial:{trial}, Event:{event_num}"
    filename_suffix = f"INV{inv_power}_RLC{rlc_power}_T{trial}_E{event_num}_{current_time}"
    
    # Create voltage plot if any voltage waveforms are selected
    if voltages == 1 or rms_voltages == 1:
        plt.figure(figsize=(12, 8))
        
        if voltages == 1:
            plt.plot(df2['timestamp_ms'], df2['VA(V)'], label='VA', linewidth=1.5, color='red')
            plt.plot(df2['timestamp_ms'], df2['VB(V)'], label='VB', linewidth=1.5, color='green')
            plt.plot(df2['timestamp_ms'], df2['VC(V)'], label='VC', linewidth=1.5, color='blue')
        if rms_voltages == 1:
            plt.plot(df2['timestamp_ms'], df2['RMS_VA_T'], label='VA RMS', linewidth=2, color='lightcoral')
            plt.plot(df2['timestamp_ms'], df2['RMS_VB_T'], label='VB RMS', linewidth=2, color='lightgreen')
            plt.plot(df2['timestamp_ms'], df2['RMS_VC_T'], label='VC RMS', linewidth=2, color='lightblue')

        plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Time of Fault')
        #add a line 2 cycles after the fault time (2 cycles = 2 * 32 / 1920 seconds = 33.33 ms)
        plt.axvline(x=2 * 32 / 1920 * 1000, color='purple', linestyle='--', linewidth=2, label='2 Cycles After Fault')
        #add a horizontal line at 480 * sqrt(2/3) * .9
        plt.axhline(y=480* .9 * ((2/3) ** 0.5), color='orange', linestyle='--', linewidth=2, label='90% Peak voltage')   
        #add another horizontal line at 480 / sqrt(3) * .95
        plt.axhline(y=480* .95 / ((3) ** 0.5), color='brown', linestyle='--', linewidth=2, label='95% RMS voltage')
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (V)')
        plt.title(f"Voltage Waveforms - {title_suffix}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'output_plots/voltage_{filename_suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create current plot if any current waveforms are selected
    if currents == 1 or rms_currents == 1:
        plt.figure(figsize=(12, 8))
        
        if currents == 1:
            plt.plot(df2['timestamp_ms'], df2['IA(A)'], label='IA', linewidth=1.5, color='red')
            plt.plot(df2['timestamp_ms'], df2['IB(A)'], label='IB', linewidth=1.5, color='green')
            plt.plot(df2['timestamp_ms'], df2['IC(A)'], label='IC', linewidth=1.5, color='blue')
        if rms_currents == 1:
            plt.plot(df2['timestamp_ms'], df2['RMS_IA_T'], label='IA RMS', linewidth=2, color='lightcoral')
            plt.plot(df2['timestamp_ms'], df2['RMS_IB_T'], label='IB RMS', linewidth=2, color='lightgreen')
            plt.plot(df2['timestamp_ms'], df2['RMS_IC_T'], label='IC RMS', linewidth=2, color='lightblue')

        plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Time of Fault')
        plt.xlabel('Time (ms)')
        plt.ylabel('Current (A)')
        plt.title(f"Current Waveforms - {title_suffix}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'output_plots/current_{filename_suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()

#does fft and plots, also does thd
def fft_and_save(df, fault_time, cycles, waveform):
    plt.figure(figsize=(4, 12))    
    closest_index = (df['timestamp'] - fault_time).abs().idxmin()
    thd_list =[]
    for i in range(cycles):
        start_index = closest_index + (i * 32)
        if start_index + 32 > len(df):
            break

        # Get data slice
        cycle_df = df.iloc[start_index:start_index+32, :]
        signal = cycle_df[waveform].to_numpy()
        N = len(signal)
        fft_values = fft(signal)
        frequencies = np.fft.fftfreq(N, d=1/1920)

        # Normalize
        amplitude = 2 * np.abs(fft_values[:N // 2]) / N
        freqs = frequencies[:N // 2]


        #make dataframe with frequency and amplitude
        fft_df = pd.DataFrame({'Frequency (Hz)': freqs, 'Magnitude (V)': amplitude})
        # Isolate fundamental (60 Hz bin, within tolerance)
        fundamental = fft_df[fft_df['Frequency (Hz)'].between(59.5, 60.5)]['Magnitude (V)'].values
        if fundamental[0] > 0:
            V1 = fundamental[0]
            print(V1)
            #remove the first row where frequency is 0 anf the row where frequency is 60
            fft_df = fft_df[(fft_df['Frequency (Hz)'] != 0) & (fft_df['Frequency (Hz)'] != 60)]
            thd = np.sqrt(np.sum(fft_df['Magnitude (V)']**2)) / V1
            thd_list.append(thd)
        else:
            thd = 9999
            thd_list.append(thd)  # No fundamental found, avoid crashing
        
        # Plot in subplot i+1 (subplot index starts at 1)
        plt.subplot(cycles, 1, i + 1)
        plt.scatter(freqs, amplitude, s=5)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (V)')
        if thd != 9999:
            plt.title(f'FFT of Cycle {i + 1} - {waveform}' + f' (THD: {thd*100:.2f}%)')
        else:
            plt.title(f'FFT of Cycle {i + 1} - {waveform}' + ' (THD: Undefined)')

        


    # Save plot
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(f'output_plots/fft_plot_{current_time}.png')

    #print(thd_list)
    
    """
    #clear plot
    plt.clf()
    #create a new plot for THD
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(thd_list)), thd_list)
    plt.xlabel('Cycle Number')
    plt.ylabel('Total Harmonic Distortion (THD)')
    plt.title('THD Across Cycles')
    plt.grid()
    plt.savefig(f'output_plots/thd_plot_{current_time}.png')
    plt.show()
    print(f'THD values: {thd_list}')
    """

def calculate_thd_table(df, fault_time, num_cycles_pre, num_cycles_post, inv_power=0, rlc_power=0, trial=1, event_num=0, create_plot=True):
    """
    Calculate THD values for all voltage and current waveforms across multiple cycles.
    
    Args:
        df: DataFrame with waveform data
        fault_time: Time of fault in seconds
        num_cycles_pre: Number of cycles before fault
        num_cycles_post: Number of cycles after fault
        inv_power: Inverter power for plot title (optional)
        rlc_power: RLC power for plot title (optional)
        trial: Trial number for plot title (optional)
        event_num: Event number for plot title (optional)
        create_plot: Whether to create and save a plot (default True)
        
    Returns:
        pd.DataFrame: THD table with cycles as rows and waveforms as columns
    """
    # Define the waveforms to analyze
    waveforms = ['VA(V)', 'VB(V)', 'VC(V)', 'IA(A)', 'IB(A)', 'IC(A)']
    
    # Calculate total number of cycles and find fault index
    total_cycles = num_cycles_pre + num_cycles_post
    closest_index = (df['timestamp'] - fault_time).abs().idxmin()
    start_index = closest_index - (num_cycles_pre * 32)
    
    # Initialize results dictionary
    thd_results = {}
    
    # Calculate THD for each waveform
    for waveform in waveforms:
        if waveform not in df.columns:
            continue
            
        thd_list = []
        
        for cycle in range(total_cycles):
            cycle_start = start_index + (cycle * 32)
            cycle_end = cycle_start + 32
            
            # Check if we have enough data
            if cycle_end > len(df):
                thd_list.append(np.nan)
                continue
                
            # Get data slice for this cycle
            cycle_df = df.iloc[cycle_start:cycle_end, :]
            signal = cycle_df[waveform].to_numpy()
            
            # Perform FFT
            N = len(signal)
            fft_values = fft(signal)
            frequencies = np.fft.fftfreq(N, d=1/1920)
            
            # Normalize
            amplitude = 2 * np.abs(fft_values[:N // 2]) / N
            freqs = frequencies[:N // 2]
            
            # Create dataframe with frequency and amplitude
            fft_df = pd.DataFrame({'Frequency (Hz)': freqs, 'Magnitude': amplitude})
            
            # Isolate fundamental (60 Hz bin, within tolerance)
            fundamental = fft_df[fft_df['Frequency (Hz)'].between(59.5, 60.5)]['Magnitude'].values
            
            if len(fundamental) > 0 and fundamental[0] > 0:
                V1 = fundamental[0]
                # Remove DC component and fundamental frequency
                harmonics_df = fft_df[(fft_df['Frequency (Hz)'] > 0) & 
                                    (~fft_df['Frequency (Hz)'].between(59.5, 60.5))]
                
                # Calculate THD
                if len(harmonics_df) > 0:
                    thd = np.sqrt(np.sum(harmonics_df['Magnitude']**2)) / V1
                    thd_list.append(thd * 100)  # Convert to percentage
                else:
                    thd_list.append(0.0)
            else:
                thd_list.append(np.nan)  # No fundamental found
        
        thd_results[waveform] = thd_list
    
    # Create cycle labels (negative for pre-fault, positive for post-fault)
    cycle_labels = []
    for i in range(total_cycles):
        cycle_num = i - num_cycles_pre + 1
        if cycle_num <= 0:
            cycle_labels.append(f"Pre-{abs(cycle_num - 1)}")
        else:
            cycle_labels.append(f"Post-{cycle_num - 1}")
    
    # Create DataFrame
    thd_table = pd.DataFrame(thd_results, index=cycle_labels)
    
    # Create plot if requested
    if create_plot:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        title_suffix = f"Initial Inverter Power:{inv_power}kW, Load Power:{rlc_power}W, Trial:{trial}, Event:{event_num}"
        filename_suffix = f"INV{inv_power}_RLC{rlc_power}_T{trial}_E{event_num}_{current_time}"
        
        # Create separate plots for voltage and current
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Voltage THD plot
        voltage_cols = ['VA(V)', 'VB(V)', 'VC(V)']
        voltage_data = thd_table[voltage_cols]
        
        for col in voltage_cols:
            if col in voltage_data.columns:
                ax1.plot(range(len(voltage_data)), voltage_data[col], 
                        marker='o', linewidth=2, markersize=6, label=col)
        
        ax1.axvline(x=num_cycles_pre, color='red', linestyle='--', linewidth=2, 
                   alpha=0.7, label='Fault Time')
        ax1.set_xlabel('Cycle')
        ax1.set_ylabel('THD (%)')
        #set y range to be -5 to 100
        ax1.set_ylim(-5, 100)
        ax1.set_title(f'Voltage THD vs Cycles - {title_suffix}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(len(cycle_labels)))
        ax1.set_xticklabels(cycle_labels, rotation=45)
        
        # Current THD plot
        current_cols = ['IA(A)', 'IB(A)', 'IC(A)']
        current_data = thd_table[current_cols]
        
        for col in current_cols:
            if col in current_data.columns:
                ax2.plot(range(len(current_data)), current_data[col], 
                        marker='s', linewidth=2, markersize=6, label=col)
        
        ax2.axvline(x=num_cycles_pre, color='red', linestyle='--', linewidth=2, 
                   alpha=0.7, label='Fault Time')
        ax2.set_xlabel('Cycle')
        ax2.set_ylabel('THD (%)')
        ax2.set_title(f'Current THD vs Cycles - {title_suffix}')
        ax2.set_ylim(-5, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(len(cycle_labels)))
        ax2.set_xticklabels(cycle_labels, rotation=45)
        
        # Adjust layout and save
        plt.tight_layout()
        plot_filename = f'output_plots/THD_plot_{filename_suffix}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"THD plot saved to: {plot_filename}")
    
    return thd_table


def calculate_event_statistics(df, inv_power, rlc_power, event_num, trial, fault_time=None):
    """
    Calculate key statistics for a single event to be added to results_statistics.csv
    
    Args:
        df: DataFrame with waveform data
        inv_power: Inverter power setting
        rlc_power: RLC load power setting  
        event_num: Event number
        trial: Trial number
        fault_time: Time of fault occurrence in seconds (optional)
        
    Returns:
        dict: Dictionary with calculated statistics
    """
    
    # Calculate initial RMS voltage (from first cycle - first 32 samples)
    first_cycle_data = df.iloc[:32]
    
    # Calculate RMS for each phase in first cycle
    va_rms_initial = np.sqrt(np.mean(first_cycle_data['VA(V)']**2))
    vb_rms_initial = np.sqrt(np.mean(first_cycle_data['VB(V)']**2))
    vc_rms_initial = np.sqrt(np.mean(first_cycle_data['VC(V)']**2))
    
    # Average initial RMS across all phases
    vrms_initial = (va_rms_initial + vb_rms_initial + vc_rms_initial) / 3
    
    # Find minimum RMS voltage across all RMS columns
    rms_columns = ['RMS_VA_T', 'RMS_VB_T', 'RMS_VC_T']
    vrms_min = float('inf')
    
    for col in rms_columns:
        if col in df.columns:
            col_min = df[col].dropna().min()
            if col_min < vrms_min:
                vrms_min = col_min
    
    # Find maximum instantaneous voltage
    voltage_columns = ['VA(V)', 'VB(V)', 'VC(V)']
    vmax = 0
    
    for col in voltage_columns:
        if col in df.columns:
            col_max = max(abs(df[col].max()), abs(df[col].min()))
            if col_max > vmax:
                vmax = col_max
    
    # Find maximum instantaneous current
    current_columns = ['IA(A)', 'IB(A)', 'IC(A)']
    imax = 0
    
    for col in current_columns:
        if col in df.columns:
            col_max = max(abs(df[col].max()), abs(df[col].min()))
            if col_max > imax:
                imax = col_max
    
    # Calculate 95% RMS voltage threshold
    threshold_95_rms = 480 / np.sqrt(3) * 0.95  # ≈ 263.2V
    
    # Calculate 90% peak voltage threshold
    threshold_90_peak = 480 * np.sqrt(2/3) * 0.9  # ≈ 352.7V
    
    # Check for peak voltage violations (any cycle peak below 90% threshold)
    peak_violation = 0
    voltage_columns = ['VA(V)', 'VB(V)', 'VC(V)']
    
    # Check each cycle (every 32 samples)
    for i in range(0, len(df) - 31, 32):  # Step by 32 samples for each cycle
        cycle_data = df.iloc[i:i+32]
        
        for col in voltage_columns:
            if col in df.columns:
                # Get the maximum peak (positive or negative) for this phase in this cycle
                cycle_peak = max(abs(cycle_data[col].max()), abs(cycle_data[col].min()))
                if cycle_peak < threshold_90_peak:
                    peak_violation = 1
                    break  # If any phase in any cycle violates, set to 1 and stop checking
        
        if peak_violation == 1:
            break  # Stop checking cycles once violation is found
    
    # Find duration each phase RMS voltage stays below 95% threshold
    rms_phase_mapping = {
        'RMS_VA_T': 'VA_below_95_duration',
        'RMS_VB_T': 'VB_below_95_duration', 
        'RMS_VC_T': 'VC_below_95_duration'
    }
    
    below_95_durations = {}
    
    for rms_col, duration_col in rms_phase_mapping.items():
        if rms_col in df.columns:
            if fault_time is not None:
                # Find the fault time index
                fault_index = (df['timestamp'] - fault_time).abs().idxmin()
                
                # Define the analysis window: from fault time to 7 cycles later
                # 7 cycles = 7 * 32 samples = 224 samples
                end_index = min(fault_index + 7 * 32, len(df) - 1)
                
                # Get data from fault time to 7 cycles later
                analysis_data = df.iloc[fault_index:end_index + 1].copy()
                
                # Get valid (non-NaN) RMS data within analysis window
                valid_data = analysis_data[analysis_data[rms_col].notna()]
                
                if not valid_data.empty:
                    # Find indices where RMS voltage is below threshold
                    below_threshold = valid_data[valid_data[rms_col] < threshold_95_rms]
                    
                    if not below_threshold.empty:
                        # Find first time below threshold
                        first_below_index = below_threshold.index[0]
                        first_below_time = df.loc[first_below_index, 'timestamp']
                        
                        # Find last time below threshold
                        last_below_index = below_threshold.index[-1]
                        last_below_time = df.loc[last_below_index, 'timestamp']
                        
                        # Check if voltage recovers after the last below-threshold point within analysis window
                        remaining_data = valid_data[valid_data.index > last_below_index]
                        recovery_found = False
                        recovery_time = last_below_time
                        
                        if not remaining_data.empty:
                            # Find first point after last below-threshold that is above threshold
                            above_threshold = remaining_data[remaining_data[rms_col] >= threshold_95_rms]
                            if not above_threshold.empty:
                                recovery_index = above_threshold.index[0]
                                recovery_time = df.loc[recovery_index, 'timestamp']
                                recovery_found = True
                        
                        # Calculate duration
                        if recovery_found:
                            duration = recovery_time - first_below_time
                        else:
                            # If no recovery found within 7 cycles, use time until end of analysis window
                            end_time = df.loc[end_index, 'timestamp']
                            duration = end_time - first_below_time
                        
                        below_95_durations[duration_col] = round(duration, 4)
                    else:
                        below_95_durations[duration_col] = 0.0  # Never dropped below threshold
                else:
                    below_95_durations[duration_col] = None  # No valid RMS data
            else:
                # Fallback to original method if no fault time provided
                # Get valid (non-NaN) RMS data
                valid_data = df[df[rms_col].notna()]
                
                if not valid_data.empty:
                    # Find indices where RMS voltage is below threshold
                    below_threshold = valid_data[valid_data[rms_col] < threshold_95_rms]
                    
                    if not below_threshold.empty:
                        # Find first time below threshold
                        first_below_index = below_threshold.index[0]
                        first_below_time = df.loc[first_below_index, 'timestamp']
                        
                        # Find last time below threshold
                        last_below_index = below_threshold.index[-1]
                        last_below_time = df.loc[last_below_index, 'timestamp']
                        
                        # Check if voltage recovers after the last below-threshold point
                        remaining_data = valid_data[valid_data.index > last_below_index]
                        recovery_found = False
                        recovery_time = last_below_time
                        
                        if not remaining_data.empty:
                            # Find first point after last below-threshold that is above threshold
                            above_threshold = remaining_data[remaining_data[rms_col] >= threshold_95_rms]
                            if not above_threshold.empty:
                                recovery_index = above_threshold.index[0]
                                recovery_time = df.loc[recovery_index, 'timestamp']
                                recovery_found = True
                        
                        # Calculate duration
                        if recovery_found:
                            duration = recovery_time - first_below_time
                        else:
                            # If no recovery found, use time until end of valid data
                            duration = last_below_time - first_below_time
                        
                        below_95_durations[duration_col] = round(duration, 4)
                    else:
                        below_95_durations[duration_col] = 0.0  # Never dropped below threshold
                else:
                    below_95_durations[duration_col] = None  # No valid RMS data
        else:
            below_95_durations[duration_col] = None
    
    # Calculate maximum voltage THD using THD table functionality
    # Use the same parameters as the main analysis (2 cycles pre, 7 cycles post)
    if fault_time is not None:
        try:
            # Calculate THD table without creating plots
            thd_table = calculate_thd_table(df, fault_time, num_cycles_pre=2, num_cycles_post=7, 
                                          inv_power=inv_power, rlc_power=rlc_power, 
                                          trial=trial, event_num=event_num, create_plot=False)
            
            # Extract maximum THD for voltage phases (VA(V), VB(V), VC(V))
            voltage_columns = ['VA(V)', 'VB(V)', 'VC(V)']
            max_voltage_thd = 0
            
            for col in voltage_columns:
                if col in thd_table.columns:
                    # Get the maximum THD value for this voltage phase (excluding NaN values)
                    col_max_thd = thd_table[col].dropna().max()
                    if not pd.isna(col_max_thd) and col_max_thd > max_voltage_thd:
                        max_voltage_thd = col_max_thd
                        
        except Exception as e:
            # If THD calculation fails, set to 0
            max_voltage_thd = 0
            print(f"Warning: THD calculation failed for Event {event_num}: {str(e)}")
    else:
        # If no fault time provided, set THD to 0
        max_voltage_thd = 0
    
    # Combine all statistics
    stats = {
        'INV_kW': inv_power / 1000,  # Convert W to kW
        'RLC_kW': rlc_power / 1000,  # Convert W to kW
        'Event': event_num,
        'Trial': trial,
        'Vrms_initial': round(vrms_initial, 2),
        'Vrms_min': round(vrms_min, 2),
        'Vmax': round(vmax, 2),
        'Imax': round(imax, 2),
        'max_voltage_thd': round(max_voltage_thd, 2),
        'peak_violation': peak_violation
    }
    
    # Add the below-95% duration data
    stats.update(below_95_durations)
    
    return stats


def create_or_update_results_csv(event_stats, filename="results_statistics.csv"):
    """
    Create or update the results_statistics.csv file with new event data
    
    Args:
        event_stats: Dictionary with event statistics
        filename: Name of the CSV file to create/update
    """
    
    # Check if file exists
    if os.path.exists(filename):
        # Read existing data
        results_df = pd.read_csv(filename)
        
        # Check if this event already exists
        existing = results_df[
            (results_df['Event'] == event_stats['Event']) & 
            (results_df['Trial'] == event_stats['Trial'])
        ]
        
        if not existing.empty:
            # Update existing row
            idx = existing.index[0]
            for key, value in event_stats.items():
                results_df.loc[idx, key] = value
        else:
            # Add new row
            results_df = pd.concat([results_df, pd.DataFrame([event_stats])], ignore_index=True)
    else:
        # Create new file
        results_df = pd.DataFrame([event_stats])
    
    # Sort by Event number and Trial
    results_df = results_df.sort_values(['Event', 'Trial']).reset_index(drop=True)
    
    # Save to CSV
    results_df.to_csv(filename, index=False)
    
    print(f"Results statistics saved to: {filename}")
    print(f"Current data:")
    print(results_df)
    
    return results_df

    