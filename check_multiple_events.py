import pandas as pd
import analysis_funtions as af
import os

# Check multiple events to see if there's a pattern
events_to_check = [10031, 10033, 10037]

for event_num in events_to_check:
    print(f"\n{'='*50}")
    print(f"CHECKING EVENT {event_num}")
    print(f"{'='*50}")
    
    # Get data from dictionary
    data_dict = pd.read_csv('data_dictionary.csv')
    event_data = data_dict[data_dict['Event #'] == event_num].iloc[0]
    
    print(f"Dictionary data:")
    print(f"  INV_power: {event_data['INV_power']}W")
    print(f"  RLC_power: {event_data['RLC_power']}W")
    print(f"  Fault_Timestamp: {event_data['Fault_Timestamp']}ms")
    
    # Check if CSV file exists
    csv_file = f'SEL_events07232025_csv/CEV_{event_num}_R.csv'
    if os.path.exists(csv_file):
        print(f"CSV file exists: {csv_file}")
        
        # Check file modification time
        mtime = os.path.getmtime(csv_file)
        import datetime
        mod_time = datetime.datetime.fromtimestamp(mtime)
        print(f"File modified: {mod_time}")
        
        # Load and check basic data
        try:
            df = af.raw_csv_to_df_add_rms(csv_file)
            
            fault_time_s = event_data['Fault_Timestamp'] / 1000
            closest_index = (df['timestamp'] - fault_time_s).abs().idxmin()
            
            # Check for voltage disturbance
            pre_fault = df.iloc[closest_index-50:closest_index]
            during_fault = df.iloc[closest_index:closest_index+100]
            
            pre_avg = (pre_fault['VA(V)'].mean() + pre_fault['VB(V)'].mean() + pre_fault['VC(V)'].mean()) / 3
            during_avg = (during_fault['VA(V)'].mean() + during_fault['VB(V)'].mean() + during_fault['VC(V)'].mean()) / 3
            
            print(f"Average voltage pre-fault: {pre_avg:.1f}V")
            print(f"Average voltage during fault: {during_avg:.1f}V")
            print(f"Voltage drop: {pre_avg - during_avg:.1f}V ({((pre_avg - during_avg)/pre_avg)*100:.1f}%)")
            
        except Exception as e:
            print(f"Error processing: {e}")
    else:
        print(f"CSV file NOT FOUND: {csv_file}")
