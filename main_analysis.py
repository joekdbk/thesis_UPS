import analysis_funtions as af
import pandas as pd

# Load the data dictionary to get fault times and test parameters
data_dict = pd.read_csv("data_dictionary.csv")
print("Data Dictionary loaded:")
print(data_dict.head())

#create a list of file paths found in "SEL_events07232025_csv"
csv_folder = "SEL_events07232025_csv"
csv_files = af.get_csv_file_paths(csv_folder)

# Get list of event numbers from data dictionary
valid_events = set(data_dict['Event #'].values)
print(f"Valid events in data dictionary: {len(valid_events)} events")
print(f"Event numbers: {sorted(valid_events)}")

# Process all CSV files that have matching events in data dictionary
processed_count = 0
for file in csv_files:
    # Extract event number from filename (e.g., "SEL_events07232025_csv/CEV_10031_R.csv" -> 10031)
    filename = file.split('\\')[-1] if '\\' in file else file.split('/')[-1]  # Handle both Windows and Unix paths
    try:
        event_num = int(filename.split('_')[1])  # Extract: 10031
    except (IndexError, ValueError):
        print(f"Skipping file with invalid format: {filename}")
        continue
    
    # Check if this event is in the data dictionary
    if event_num not in valid_events:
        print(f"Skipping Event {event_num}: Not found in data dictionary")
        continue
    
    print(f"\n{'='*60}")
    print(f"Processing Event {event_num} ({processed_count + 1})")
    print(f"{'='*60}")
    
    # Look up the fault data for this event
    event_data = data_dict[data_dict['Event #'] == event_num]
    if not event_data.empty:
        fault_time = event_data['Fault_Timestamp'].iloc[0] / 1000  # Convert ms to seconds
        inv_power = event_data['INV_power'].iloc[0]
        rlc_power = event_data['RLC_power'].iloc[0]
        trial = event_data['Trial'].iloc[0]
        print(f"Event {event_num}: Fault time = {fault_time}s, INV = {inv_power}W, RLC = {rlc_power}W, Trial = {trial}")
    else:
        # This shouldn't happen since we checked above, but just in case
        print(f"Event {event_num} not found in dictionary, skipping")
        continue

    try:
        df = af.raw_csv_to_df_add_rms(file)

        # Use fault time from data dictionary (already converted to seconds above)
        num_cycles_pre = 2
        num_cycles_post = 7
        voltages = 1
        rms_voltages = 1
        currents = 1
        rms_currents = 1

        af.plot_and_save(df, fault_time, num_cycles_pre, num_cycles_post, voltages, rms_voltages, currents, rms_currents, inv_power, rlc_power, trial, event_num)

        thd_table = af.calculate_thd_table(df, fault_time, num_cycles_pre, num_cycles_post, inv_power, rlc_power, trial, event_num)
        print("THD Summary:")
        print(thd_table)

        # Save the THD table to CSV
        current_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        thd_filename = f"output_plots/THD_table_INV{inv_power}_RLC{rlc_power}_T{trial}_E{event_num}_{current_time}.csv"
        thd_table.to_csv(thd_filename)
        print(f"THD table saved to: {thd_filename}")

        # Calculate event statistics and update results CSV
        event_stats = af.calculate_event_statistics(df, inv_power, rlc_power, event_num, trial, fault_time)
        results_df = af.create_or_update_results_csv(event_stats)
        
        processed_count += 1
        print(f"Successfully processed Event {event_num}")
        
    except Exception as e:
        print(f"Error processing Event {event_num}: {str(e)}")
        continue

print(f"\n{'='*60}")
print(f"PROCESSING COMPLETE")
print(f"{'='*60}")
print(f"Total events processed: {processed_count}")
print(f"Final results saved to: results_statistics.csv")

