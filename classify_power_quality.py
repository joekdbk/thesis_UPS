import pandas as pd

def classify_power_quality():
    """
    Classify power quality performance based on voltage sag duration criteria.
    Creates three classification columns based on different power quality standards.
    """
    print("Loading final analysis data...")
    
    # Read the final analysis CSV
    df = pd.read_csv('final_analysis.csv')
    
    print(f"Loaded {len(df)} events for classification")
    
    # Initialize classification columns (1 = pass, 0 = fail)
    df['class_1'] = 1
    df['class_2'] = 1  
    df['class_3'] = 1
    
    print("\nApplying Classification 1 criteria:")
    print("- Fails if any phase below 70% nominal for ≥5ms (0.005s)")
    print("- Fails if any phase below 86% nominal for ≥40ms (0.04s)")
    print("- Fails if any phase below 88% nominal for ≥60ms (0.06s)")
    print("- Fails if any phase below 89% nominal for ≥100ms (0.1s)")
    
    # Classification 1 criteria (fails if below thresholds)
    for index, row in df.iterrows():
        # Check 70% threshold for 5ms (0.005s)
        if (row['VA_below_70_duration'] >= 0.005 or 
            row['VB_below_70_duration'] >= 0.005 or 
            row['VC_below_70_duration'] >= 0.005):
            df.at[index, 'class_1'] = 0
            
        # Check 86% threshold for 40ms (0.04s)
        elif (row['VA_below_86_duration'] >= 0.04 or 
              row['VB_below_86_duration'] >= 0.04 or 
              row['VC_below_86_duration'] >= 0.04):
            df.at[index, 'class_1'] = 0
            
        # Check 88% threshold for 60ms (0.06s)
        elif (row['VA_below_88_duration'] >= 0.06 or 
              row['VB_below_88_duration'] >= 0.06 or 
              row['VC_below_88_duration'] >= 0.06):
            df.at[index, 'class_1'] = 0
            
        # Check 89% threshold for 100ms (0.1s)
        elif (row['VA_below_89_duration'] >= 0.1 or 
              row['VB_below_89_duration'] >= 0.1 or 
              row['VC_below_89_duration'] >= 0.1):
            df.at[index, 'class_1'] = 0
    
    print("\nApplying Classification 2 criteria:")
    print("- Fails if any phase below 0% nominal for ≥1ms (0.001s)")
    print("- Fails if any phase below 80% nominal for ≥20ms (0.02s)")
    print("- Fails if any phase below 86% nominal for ≥40ms (0.04s)")
    print("- Fails if any phase below 88% nominal for ≥60ms (0.06s)")
    print("- Fails if any phase below 89% nominal for ≥80ms (0.08s)")
    
    # Classification 2 criteria
    for index, row in df.iterrows():
        # Check 0% threshold for 1ms (0.001s) 
        if (row['VA_below_0_duration'] >= 0.001 or 
            row['VB_below_0_duration'] >= 0.001 or 
            row['VC_below_0_duration'] >= 0.001):
            df.at[index, 'class_2'] = 0
            
        # Check 80% threshold for 20ms (0.02s)
        elif (row['VA_below_80_duration'] >= 0.02 or 
              row['VB_below_80_duration'] >= 0.02 or 
              row['VC_below_80_duration'] >= 0.02):
            df.at[index, 'class_2'] = 0
            
        # Check 86% threshold for 40ms (0.04s)
        elif (row['VA_below_86_duration'] >= 0.04 or 
              row['VB_below_86_duration'] >= 0.04 or 
              row['VC_below_86_duration'] >= 0.04):
            df.at[index, 'class_2'] = 0
            
        # Check 88% threshold for 60ms (0.06s)
        elif (row['VA_below_88_duration'] >= 0.06 or 
              row['VB_below_88_duration'] >= 0.06 or 
              row['VC_below_88_duration'] >= 0.06):
            df.at[index, 'class_2'] = 0
            
        # Check 89% threshold for 80ms (0.08s)
        elif (row['VA_below_89_duration'] >= 0.08 or 
              row['VB_below_89_duration'] >= 0.08 or 
              row['VC_below_89_duration'] >= 0.08):
            df.at[index, 'class_2'] = 0
    
    print("\nApplying Classification 3 criteria:")
    print("- Fails if any phase below 0% nominal for ≥10ms (0.01s)")
    print("- Fails if any phase below 73% nominal for ≥100ms (0.1s)")
    print("- Fails if any phase below 80% nominal for ≥1000ms (1.0s)")
    
    # Classification 3 criteria
    for index, row in df.iterrows():
        # Check 0% threshold for 10ms (0.01s)
        if (row['VA_below_0_duration'] >= 0.01 or 
            row['VB_below_0_duration'] >= 0.01 or 
            row['VC_below_0_duration'] >= 0.01):
            df.at[index, 'class_3'] = 0
            
        # Check 73% threshold for 100ms (0.1s)
        elif (row['VA_below_73_duration'] >= 0.1 or 
              row['VB_below_73_duration'] >= 0.1 or 
              row['VC_below_73_duration'] >= 0.1):
            df.at[index, 'class_3'] = 0
            
        # Check 80% threshold for 1000ms (1.0s)
        elif (row['VA_below_80_duration'] >= 1.0 or 
              row['VB_below_80_duration'] >= 1.0 or 
              row['VC_below_80_duration'] >= 1.0):
            df.at[index, 'class_3'] = 0
    
    # Save the results
    df.to_csv('final_analysis_classified.csv', index=False)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS SUMMARY")
    print("="*60)
    
    class1_pass = df['class_1'].sum()
    class1_fail = len(df) - class1_pass
    class1_pass_rate = (class1_pass / len(df)) * 100
    
    class2_pass = df['class_2'].sum()
    class2_fail = len(df) - class2_pass
    class2_pass_rate = (class2_pass / len(df)) * 100
    
    class3_pass = df['class_3'].sum()
    class3_fail = len(df) - class3_pass
    class3_pass_rate = (class3_pass / len(df)) * 100
    
    print(f"Classification 1:")
    print(f"  Pass: {class1_pass} events ({class1_pass_rate:.1f}%)")
    print(f"  Fail: {class1_fail} events ({100-class1_pass_rate:.1f}%)")
    
    print(f"\nClassification 2:")
    print(f"  Pass: {class2_pass} events ({class2_pass_rate:.1f}%)")
    print(f"  Fail: {class2_fail} events ({100-class2_pass_rate:.1f}%)")
    
    print(f"\nClassification 3:")
    print(f"  Pass: {class3_pass} events ({class3_pass_rate:.1f}%)")
    print(f"  Fail: {class3_fail} events ({100-class3_pass_rate:.1f}%)")
    
    # Show detailed results for failed events
    print("\n" + "="*60)
    print("DETAILED FAILURE ANALYSIS")
    print("="*60)
    
    failed_events = df[(df['class_1'] == 0) | (df['class_2'] == 0) | (df['class_3'] == 0)]
    
    if len(failed_events) > 0:
        print(f"\nEvents that failed at least one classification:")
        for index, row in failed_events.iterrows():
            failures = []
            if row['class_1'] == 0:
                failures.append("Class 1")
            if row['class_2'] == 0:
                failures.append("Class 2")
            if row['class_3'] == 0:
                failures.append("Class 3")
            
            print(f"Event {row['Event']} (INV: {row['INV_kW']}kW, RLC: {row['RLC_kW']}kW): Failed {', '.join(failures)}")
    else:
        print("All events passed all classifications!")
    
    print(f"\nResults saved to: final_analysis_classified.csv")
    print(f"Total columns: {len(df.columns)} (added 3 classification columns)")
    
    return df

if __name__ == "__main__":
    classified_results = classify_power_quality()
    print("\n" + "="*60)
    print("PROCESS COMPLETE")
    print("="*60)
    print("✅ Power quality classification completed")
    print("✅ Results saved to final_analysis_classified.csv")
