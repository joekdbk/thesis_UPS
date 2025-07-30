import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def create_post_analysis_plots():
    """
    Create post-analysis scatter plots for UPS performance analysis.
    Saves plots to the 'post_plots' folder.
    """
    print("Loading classified analysis data...")
    
    # Read the classified results
    df = pd.read_csv('final_analysis_classified.csv')
    
    print(f"Loaded {len(df)} events for post-analysis plotting")
    
    # Create output directory if it doesn't exist
    output_dir = 'post_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Calculate power change (RLC - INV) * 1000 for better visualization
    df['power_change_w'] = (df['RLC_kW'] - df['INV_kW']) * 1000
    
    # Check for potential data issues - validate Vrms_min values
    print(f"\nData validation for Vrms_min:")
    print(f"Event 10033: Vrms_initial = {df[df['Event'] == 10033]['Vrms_initial'].iloc[0]:.2f}V, Vrms_min = {df[df['Event'] == 10033]['Vrms_min'].iloc[0]:.2f}V")
    
    # Check if Vrms_min values seem reasonable (not negative, not greater than initial)
    invalid_data = df[(df['Vrms_min'] < 0) | (df['Vrms_min'] > df['Vrms_initial'])]
    if len(invalid_data) > 0:
        print(f"Warning: Found {len(invalid_data)} events with potentially invalid Vrms_min values:")
        print(invalid_data[['Event', 'Vrms_initial', 'Vrms_min']])
    
    # Calculate max RMS voltage drop as percentage
    # Using the initial voltage as nominal since it's close to 283V for all events
    df['max_voltage_drop_percent'] = ((df['Vrms_initial'] - df['Vrms_min']) / df['Vrms_initial']) * 100
    
    # Flag events where voltage drop might be unrealistic (>95%)
    high_drop_events = df[df['max_voltage_drop_percent'] > 95]
    if len(high_drop_events) > 0:
        print(f"\nEvents with >95% voltage drop (potential complete collapse):")
        for _, event in high_drop_events.iterrows():
            print(f"Event {event['Event']}: {event['max_voltage_drop_percent']:.1f}% drop (Vrms_min = {event['Vrms_min']:.2f}V)")
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Plot 1: Power Change vs THD
    print("\nCreating Plot 1: Power Change vs THD")
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with different colors for different classification results
    # Color by class_1 performance (most stringent classification)
    colors = ['red' if x == 0 else 'green' for x in df['class_1']]
    
    scatter1 = plt.scatter(df['power_change_w'], df['max_voltage_thd'], 
                          c=colors, alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
    
    plt.xlabel('Power Change (RLC - INV) [kW]', fontsize=12)
    plt.ylabel('Maximum Voltage THD [%]', fontsize=12)
    plt.title('UPS Performance: Power Change vs Maximum Voltage THD', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label='Class 1 Pass'),
                      Patch(facecolor='red', label='Class 1 Fail')]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add trend line
    z = np.polyfit(df['power_change_w'], df['max_voltage_thd'], 1)
    p = np.poly1d(z)
    plt.plot(df['power_change_w'], p(df['power_change_w']), "r--", alpha=0.8, linewidth=2, label=f'Trend: y = {z[0]:.3f}x + {z[1]:.1f}')
    
    # Add correlation coefficient
    correlation = np.corrcoef(df['power_change_w'], df['max_voltage_thd'])[0,1]
    plt.text(0.05, 0.95, f'Correlation: r = {correlation:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, 'power_change_vs_thd.png')
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot1_path}")
    
    # Plot 2: Power Change vs Max RMS Voltage Drop
    print("Creating Plot 2: Power Change vs Max RMS Voltage Drop")
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with different colors for different classification results
    # Color by class_1 performance (most stringent classification)
    scatter2 = plt.scatter(df['power_change_w'], df['max_voltage_drop_percent'], 
                          c=colors, alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
    
    plt.xlabel('Power Change (RLC - INV) [kW]', fontsize=12)
    plt.ylabel('Maximum RMS Voltage Drop [%]', fontsize=12)
    plt.title('UPS Performance: Power Change vs Maximum RMS Voltage Drop', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [Patch(facecolor='green', label='Class 1 Pass'),
                      Patch(facecolor='red', label='Class 1 Fail')]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add trend line
    z2 = np.polyfit(df['power_change_w'], df['max_voltage_drop_percent'], 1)
    p2 = np.poly1d(z2)
    plt.plot(df['power_change_w'], p2(df['power_change_w']), "r--", alpha=0.8, linewidth=2, label=f'Trend: y = {z2[0]:.3f}x + {z2[1]:.1f}')
    
    # Add correlation coefficient
    correlation2 = np.corrcoef(df['power_change_w'], df['max_voltage_drop_percent'])[0,1]
    plt.text(0.05, 0.95, f'Correlation: r = {correlation2:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, 'power_change_vs_voltage_drop.png')
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot2_path}")
    
    # Create a summary statistics table
    print("\nCreating summary statistics...")
    
    # Group by power change ranges for analysis
    df['power_range'] = pd.cut(df['power_change_w'], 
                              bins=[-400, -200, 0, 200, 400], 
                              labels=['Very Negative (-375 to -200W)', 'Negative (-200 to 0W)', 
                                     'Zero to Positive (0 to 200W)', 'High Positive (200W+)'])
    
    # Summary statistics
    summary_stats = df.groupby('power_range').agg({
        'max_voltage_thd': ['mean', 'std', 'min', 'max'],
        'max_voltage_drop_percent': ['mean', 'std', 'min', 'max'],
        'class_1': 'sum',
        'class_2': 'sum', 
        'class_3': 'sum'
    }).round(2)
    
    print("\nSummary Statistics by Power Change Range:")
    print(summary_stats)
    
    # Print detailed analysis
    print("\n" + "="*80)
    print("POST-ANALYSIS RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nCorrelation Analysis:")
    print(f"Power Change vs THD: r = {correlation:.3f}")
    print(f"Power Change vs Voltage Drop: r = {correlation2:.3f}")
    
    print(f"\nTrend Analysis:")
    print(f"THD Trend: For every 1W increase in power change, THD changes by {z[0]:.3f}%")
    print(f"Voltage Drop Trend: For every 1W increase in power change, voltage drop changes by {z2[0]:.3f}%")
    
    # Identify extreme cases
    max_thd_event = df.loc[df['max_voltage_thd'].idxmax()]
    max_drop_event = df.loc[df['max_voltage_drop_percent'].idxmax()]
    
    print(f"\nExtreme Cases:")
    print(f"Highest THD: Event {max_thd_event['Event']} with {max_thd_event['max_voltage_thd']:.1f}% THD")
    print(f"  - Power Change: {max_thd_event['power_change_w']:.0f}W")
    print(f"  - INV: {max_thd_event['INV_kW']}kW, RLC: {max_thd_event['RLC_kW']}kW")
    
    print(f"Largest Voltage Drop: Event {max_drop_event['Event']} with {max_drop_event['max_voltage_drop_percent']:.1f}% drop")
    print(f"  - Power Change: {max_drop_event['power_change_w']:.0f}W")
    print(f"  - INV: {max_drop_event['INV_kW']}kW, RLC: {max_drop_event['RLC_kW']}kW")
    
    # Plot 3: Pass/Fail Rate vs Power Change (using handwritten table)
    print("Creating Plot 3: Pass/Fail Rate vs Power Change (from handwritten table)")
    
    # Load handwritten table data
    try:
        handwritten_df = pd.read_csv('handwritten_table.csv')
        
        # Clean the data - remove empty rows
        handwritten_df = handwritten_df.dropna(subset=['Event #'])
        
        # Calculate power change in kW (RLC - INV)
        # Note: The handwritten table uses kVA values, convert to match our analysis scale
        # Convert kVA to kW (assuming power factor ~1)
        handwritten_df['power_change_kw'] = (handwritten_df['RLC kVA'] - handwritten_df['INV kVA'])
        
        # Convert Pass/Fail to binary (1 = Pass, 0 = Fail)
        handwritten_df['pass_binary'] = handwritten_df['Pass/Fail'].map({'P': 1, 'F': 0})
        
        # Group by power change and calculate pass rates
        power_groups = handwritten_df.groupby('power_change_kw').agg({
            'pass_binary': ['count', 'sum', 'mean']
        }).round(3)
        
        power_groups.columns = ['total_events', 'passes', 'pass_rate']
        power_groups = power_groups.reset_index()
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Create bar plot for pass rates with thicker bars
        bars = plt.bar(power_groups['power_change_kw'], power_groups['pass_rate'] * 100, 
                      width=0.08, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Color bars based on pass rate (green for high, red for low)
        for i, bar in enumerate(bars):
            pass_rate = power_groups.iloc[i]['pass_rate']
            if pass_rate >= 0.8:
                bar.set_color('green')
            elif pass_rate >= 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Add text labels on bars showing actual counts
        for i, row in power_groups.iterrows():
            plt.text(row['power_change_kw'], row['pass_rate'] * 100 + 2, 
                    f"{int(row['passes'])}/{int(row['total_events'])}", 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.xlabel('Power Change [kW]', fontsize=12)
        plt.ylabel('Pass Rate [%]', fontsize=12)
        plt.title('UPS Pass/Fail Rate vs Power Change', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, 110)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='Pass Rate ≥ 80%'),
                          Patch(facecolor='orange', label='Pass Rate 50-80%'),
                          Patch(facecolor='red', label='Pass Rate < 50%')]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Add overall statistics text
        overall_pass_rate = handwritten_df['pass_binary'].mean() * 100
        total_events = len(handwritten_df)
        total_passes = handwritten_df['pass_binary'].sum()
        
        plt.text(0.02, 0.98, f'Overall: {int(total_passes)}/{total_events} events passed ({overall_pass_rate:.1f}%)', 
                transform=plt.gca().transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plot3_path = os.path.join(output_dir, 'pass_fail_rate_vs_power_change.png')
        plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot3_path}")
        
        # Print detailed analysis
        print(f"\nPass/Fail Analysis by Power Change:")
        for _, row in power_groups.iterrows():
            print(f"Power Change {row['power_change_kw']:+5.1f}kW: {int(row['passes'])}/{int(row['total_events'])} passed ({row['pass_rate']*100:.1f}%)")
        
        print(f"\nOverall Pass Rate: {int(total_passes)}/{total_events} events ({overall_pass_rate:.1f}%)")
        
    except FileNotFoundError:
        print("Warning: handwritten_table.csv not found. Skipping pass/fail rate plot.")
    
    # Plot 4: Maximum Current vs Class 3 Pass/Fail
    print("Creating Plot 4: Maximum Current by Pass/Fail")
    plt.figure(figsize=(10, 8))
    
    # Convert class_3 numeric values to descriptive labels
    df['class_3_label'] = df['class_3'].map({1: 'Pass', 0: 'Fail'})
    
    # Create separate data for Pass and Fail
    pass_data = df[df['class_3'] == 1]['Imax']
    fail_data = df[df['class_3'] == 0]['Imax']
    
    # Create scatter plot only (no box plots)
    box_data = [pass_data, fail_data]
    box_labels = ['Pass', 'Fail']
    
    # Add individual data points as scatter plot
    for i, (data, label) in enumerate(zip(box_data, box_labels)):
        # Add some jitter to x-coordinates for better visualization
        x_coords = np.random.normal(i + 1, 0.1, size=len(data))
        plt.scatter(x_coords, data, alpha=0.7, s=80, 
                   color='green' if label == 'Pass' else 'red', 
                   edgecolors='black', linewidth=0.5, label=label if i == 0 else "")
    
    # Set x-axis labels and limits
    plt.xticks([1, 2], box_labels)
    plt.xlim(0.5, 2.5)
    
    plt.xlabel('Pass/Fail Result', fontsize=12)
    plt.ylabel('Maximum Current [A]', fontsize=12)
    plt.title('Maximum Current Distribution by Pass/Fail Status', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    pass_mean = pass_data.mean()
    fail_mean = fail_data.mean()
    pass_count = len(pass_data)
    fail_count = len(fail_data)
    
    stats_text = f'Pass: n={pass_count}\nFail: n={fail_count}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plot4_path = os.path.join(output_dir, 'imax_vs_class3_passfail.png')
    plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot4_path}")
    
    print(f"\nPlots saved to '{output_dir}' folder:")
    print(f"  - power_change_vs_thd.png")
    print(f"  - power_change_vs_voltage_drop.png")
    print(f"  - pass_fail_rate_vs_power_change.png")
    print(f"  - imax_vs_class3_passfail.png")
    
    return df

if __name__ == "__main__":
    results = create_post_analysis_plots()
    print("\n" + "="*80)
    print("POST-ANALYSIS PLOTTING COMPLETE")
    print("="*80)
    print("✅ Scatter plots created and saved")
    print("✅ Statistical analysis completed")
