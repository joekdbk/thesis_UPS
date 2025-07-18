import pandas as pd
import math
import matplotlib.pyplot as plt
import datetime
from scipy.fft import fft
import numpy as np


def poop():
    print("poop")

def raw_csv_to_df_add_rms(file):
    file = "SELevents07012025_csv\CEV_10015_R.csv"
    fault_time = 0.128646 #sceonds
    num_cycles_pre = 2
    num_cycles_post = 10

    df = pd.read_csv(file, skiprows=6)
    #remove all quotes from the column names
    df.columns = df.columns.str.replace('"', '', regex=False)
    #remove all rows after 511
    df = df.iloc[:512]
    #remove all collumns greater than 19
    df = df.iloc[:, :19]
    #add a column named timestamp that starts at 0 and increments by 1/1920 for each row
    df['timestamp'] = df.index / 1920
    df.insert(0, 'timestamp', df.pop('timestamp'))

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
        for i in range(len(df)):
            if i < 32:
                df.at[i, new_col] = None
            else:
                total = 0
                for j in range(i - 31, i + 1):
                    val = float(df.at[j, col])
                    total += (val * math.sqrt(2))**2
                df.at[i, new_col] = (total / 32)**0.5


    return df

def plot_and_save(df,fault_time,num_cycles_pre,num_cycles_post,voltages,rms_voltages, currents, rms_currents):
    closest_index = (df['timestamp'] - fault_time).abs().idxmin()
    df2 = df.iloc[closest_index-(num_cycles_pre*32):closest_index+(num_cycles_post*32), :]

    if voltages == 1:
        plt.plot(df2['timestamp'], df2['VA(V)'], label='VA(V)')
        plt.plot(df2['timestamp'], df2['VB(V)'], label='VB(V)')
        plt.plot(df2['timestamp'], df2['VC(V)'], label='VC(V)')
    if currents == 1:
        plt.plot(df2['timestamp'], df2['IA(A)'], label='IA(A)')
        plt.plot(df2['timestamp'], df2['IB(A)'], label='IB(A)')
        plt.plot(df2['timestamp'], df2['IC(A)'], label='IC(A)')
    if rms_voltages == 1:
        plt.plot(df2['timestamp'], df2['RMS_VA_T'], label='RMS_VA_T')
        plt.plot(df2['timestamp'], df2['RMS_VB_T'], label='RMS_VB_T')
        plt.plot(df2['timestamp'], df2['RMS_VC_T'], label='RMS_VC_T')
    if rms_currents == 1:
        plt.plot(df2['timestamp'], df2['RMS_IA_T'], label='RMS_IA_T')
        plt.plot(df2['timestamp'], df2['RMS_IB_T'], label='RMS_IB_T')
        plt.plot(df2['timestamp'], df2['RMS_IC_T'], label='RMS_IC_T')
        
    plt.axvline(x=df2['timestamp'].iloc[num_cycles_pre*32], color='r', linestyle='--', label='Time of Fault')
    plt.xlabel('Time (s)')
    plt.title("Waveforms")
    plt.legend()
    plt.grid()
    # Save the plot
    #get the current date and time
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig('output_plots/time_plot' + current_time + '.png')

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
    plt.show()
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

    