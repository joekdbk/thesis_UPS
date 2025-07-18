import analysis_funtions as af

file = "SELevents07012025_csv\CEV_10015_R.csv"
df = af.raw_csv_to_df_add_rms(file)
#def plot_and_save(df,fault_time,num_cycles_pre,num_cycles_post,voltages,rms_voltages, currents, rms_currents):

fault_time = 0.128646 -.0166  # seconds
num_cycles_pre = 2
num_cycles_post = 2
voltages = 1
rms_voltages = 1
currents = 0
rms_currents = 0

#need to implement saving
#af.plot_and_save(df,fault_time,num_cycles_pre,num_cycles_post,voltages,rms_voltages, currents, rms_currents)

waveform = "VA(V)"
cycles = 5
af.fft_and_save(df, fault_time, cycles,waveform)

#make a dictionary with file paths and fault times
#Later