import analysis_funtions as af

# create a list of file paths found in "SEL_events07232025_csv"
csv_folder = "SEL_events07232025_csv"
csv_files = af.get_csv_file_paths(csv_folder)

print("CSV files found:")
for file in csv_files:
    print(file)