import pandas as pd

def create_and_reorder_joined_results():
    """
    Create joined results by merging results_statistics.csv with data_dictionary.csv,
    then reorder the results to match the data dictionary order.
    """
    print("Creating joined results...")
    
    # Read the CSV files
    results_stats = pd.read_csv('results_statistics.csv')
    data_dict = pd.read_csv('data_dictionary.csv')
    
    # Rename the column in data_dictionary to match
    data_dict = data_dict.rename(columns={'Event #': 'Event'})
    
    # Convert Fault_Timestamp to seconds for consistency
    data_dict['Fault_Timestamp'] = data_dict['Fault_Timestamp'] / 1000.0
    
    # Perform the join on the 'Event' field
    joined_results = pd.merge(results_stats, data_dict, on='Event', how='inner', suffixes=('', '_dict'))
    
    # Drop duplicate columns that came from both datasets
    # Keep the values from results_statistics for INV_kW, RLC_kW, Trial since they're already processed
    columns_to_drop = ['INV_power', 'RLC_power', 'Trial_dict']
    joined_results = joined_results.drop(columns=[col for col in columns_to_drop if col in joined_results.columns])
    
    # Reorder columns for better readability
    column_order = ['Event', 'Trial', 'Fault_Timestamp', 'INV_kW', 'RLC_kW', 
                    'Vrms_initial', 'Vrms_min', 'Vmax', 'Imax', 'max_voltage_thd',
                    'VA_below_95_time', 'VB_below_95_time', 'VC_below_95_time',
                    'VA_below_95_duration', 'VB_below_95_duration', 'VC_below_95_duration',
                    'peak_violation']
    
    # Reorder columns (only include columns that exist)
    joined_results = joined_results[column_order]
    
    print(f"Successfully merged {len(joined_results)} records")
    print(f"Columns: {list(joined_results.columns)}")
    
    # Now reorder according to data dictionary order
    print("\nReordering results to match data dictionary order...")
    
    # Create a mapping of event to order from data dictionary
    event_order = {event: idx for idx, event in enumerate(data_dict['Event'])}
    
    # Add an order column to the joined results based on data dictionary order
    joined_results['order'] = joined_results['Event'].map(event_order)
    
    # Sort by the order column
    joined_results_reordered = joined_results.sort_values('order')
    
    # Drop the temporary order column
    joined_results_reordered = joined_results_reordered.drop('order', axis=1)
    
    # Save the final results
    joined_results_reordered.to_csv('joined_results.csv', index=False)
    
    # Also update final_analysis.csv with the same content
    joined_results_reordered.to_csv('final_analysis.csv', index=False)
    
    print("Results saved to both 'joined_results.csv' and 'final_analysis.csv'")
    print(f"Total records: {len(joined_results_reordered)}")
    
    print("\nEvent order (matches data dictionary):")
    for i, event in enumerate(joined_results_reordered['Event'], 1):
        print(f"{i:2d}. Event {event}")
    
    print("\nFirst few rows of final results:")
    print(joined_results_reordered.head())
    
    return joined_results_reordered

if __name__ == "__main__":
    # Run the combined function
    final_results = create_and_reorder_joined_results()
    print("\n" + "="*60)
    print("PROCESS COMPLETE")
    print("="*60)
    print("✅ Results successfully joined and reordered")
    print("✅ Files saved: joined_results.csv and final_analysis.csv")
