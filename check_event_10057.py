import pandas as pd
import analysis_funtions as af
import matplotlib.pyplot as plt
import numpy as np

# Check event 10057 timing
data_dict = pd.read_csv('data_dictionary.csv')
event_10057 = data_dict[data_dict['Event #'] == 10057].iloc[0]

print("Event 10057 from data dictionary:")
print(f"Fault_Timestamp: {event_10057['Fault_Timestamp']}ms")
print(f"INV_power: {event_10057['INV_power']}W")
print(f"RLC_power: {event_10057['RLC_power']}W")
print(f"Trial: {event_10057['Trial']}")

# Load and process the CSV
csv_file = 'SEL_events07232025_csv/CEV_10057_R.csv'
df = af.raw_csv_to_df_add_rms(csv_file)

fault_time_s = event_10057['Fault_Timestamp'] / 1000  # Convert to seconds
print(f"\nFault time converted to seconds: {fault_time_s}s")
print(f"Data timestamp range: {df['timestamp'].min():.3f}s to {df['timestamp'].max():.3f}s")

# Find closest index to fault time
closest_index = (df['timestamp'] - fault_time_s).abs().idxmin()
closest_time = df.loc[closest_index, 'timestamp']
print(f"Closest timestamp to fault: {closest_time:.3f}s (index {closest_index})")

# Plot the first 1000ms of data
first_1000ms = df[df['timestamp'] <= 1.0]  # First 1000ms (1 second)
print(f"\nPlotting first 1000ms ({len(first_1000ms)} samples)")

plt.figure(figsize=(16, 10))

# Plot voltages
plt.subplot(2, 1, 1)
plt.plot(first_1000ms['timestamp'] * 1000, first_1000ms['VA(V)'], label='VA(V)', color='red', linewidth=1)
plt.plot(first_1000ms['timestamp'] * 1000, first_1000ms['VB(V)'], label='VB(V)', color='green', linewidth=1)
plt.plot(first_1000ms['timestamp'] * 1000, first_1000ms['VC(V)'], label='VC(V)', color='blue', linewidth=1)

# Add fault time marker
plt.axvline(x=event_10057['Fault_Timestamp'], color='black', linestyle='--', linewidth=2, label=f'Fault Time ({event_10057["Fault_Timestamp"]}ms)')

plt.xlabel('Time (ms)')
plt.ylabel('Voltage (V)')
plt.title('Event 10057 - First 1000ms Voltage Waveforms')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot currents
plt.subplot(2, 1, 2)
plt.plot(first_1000ms['timestamp'] * 1000, first_1000ms['IA(A)'], label='IA(A)', color='red', linewidth=1)
plt.plot(first_1000ms['timestamp'] * 1000, first_1000ms['IB(A)'], label='IB(A)', color='green', linewidth=1)
plt.plot(first_1000ms['timestamp'] * 1000, first_1000ms['IC(A)'], label='IC(A)', color='blue', linewidth=1)

# Add fault time marker
plt.axvline(x=event_10057['Fault_Timestamp'], color='black', linestyle='--', linewidth=2, label=f'Fault Time ({event_10057["Fault_Timestamp"]}ms)')

plt.xlabel('Time (ms)')
plt.ylabel('Current (A)')
plt.title('Event 10057 - First 1000ms Current Waveforms')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('event_10057_first_1000ms.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nPlot saved as: event_10057_first_1000ms.png")

# Check voltage around fault time
fault_window = df.iloc[closest_index-50:closest_index+50]
print(f"\nVoltage values around fault time:")
print(f"VA(V) range: {fault_window['VA(V)'].min():.1f} to {fault_window['VA(V)'].max():.1f}")
print(f"VB(V) range: {fault_window['VB(V)'].min():.1f} to {fault_window['VB(V)'].max():.1f}")
print(f"VC(V) range: {fault_window['VC(V)'].min():.1f} to {fault_window['VC(V)'].max():.1f}")

# Check if there's a voltage sag/disturbance at this time
pre_fault = df.iloc[closest_index-100:closest_index-50]
during_fault = df.iloc[closest_index:closest_index+100]

print(f"\nPre-fault voltage averages:")
print(f"VA: {pre_fault['VA(V)'].mean():.1f}V")
print(f"VB: {pre_fault['VB(V)'].mean():.1f}V") 
print(f"VC: {pre_fault['VC(V)'].mean():.1f}V")

print(f"\nDuring fault voltage averages:")
print(f"VA: {during_fault['VA(V)'].mean():.1f}V")
print(f"VB: {during_fault['VB(V)'].mean():.1f}V")
print(f"VC: {during_fault['VC(V)'].mean():.1f}V")

# Check for RMS values around fault time too
if 'RMS_VA_T' in df.columns:
    print(f"\nRMS voltages around fault time:")
    rms_window = df.iloc[closest_index-10:closest_index+50]
    print(f"RMS_VA_T range: {rms_window['RMS_VA_T'].dropna().min():.1f} to {rms_window['RMS_VA_T'].dropna().max():.1f}")
    print(f"RMS_VB_T range: {rms_window['RMS_VB_T'].dropna().min():.1f} to {rms_window['RMS_VB_T'].dropna().max():.1f}")
    print(f"RMS_VC_T range: {rms_window['RMS_VC_T'].dropna().min():.1f} to {rms_window['RMS_VC_T'].dropna().max():.1f}")

# Additional comparison with event 10031
print(f"\n" + "="*60)
print(f"COMPARISON: Event 10057 vs Event 10031")
print(f"="*60)
print(f"Event 10057: Fault at {event_10057['Fault_Timestamp']}ms, INV={event_10057['INV_power']}W, RLC={event_10057['RLC_power']}W, Trial={event_10057['Trial']}")
print(f"Event 10031: Fault at 104.0ms, INV=0W, RLC=125W, Trial=1")
print(f"Both events have same RLC power (125W) but different INV power and trials")
