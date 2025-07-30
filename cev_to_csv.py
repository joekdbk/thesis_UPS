import os
import csv

def convert_cev_to_csv(cev_folder, csv_folder):
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)

    for filename in os.listdir(cev_folder):
        if filename.lower().endswith('.cev'):
            cev_path = os.path.join(cev_folder, filename)
            csv_filename = os.path.splitext(filename)[0] + '.csv'
            csv_path = os.path.join(csv_folder, csv_filename)

            with open(cev_path, 'r') as cev_file:
                lines = cev_file.readlines()

            # Assuming .CEV files are comma-separated and have headers
            with open(csv_path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                for line in lines:
                    row = line.strip().split(',')
                    writer.writerow(row)

if __name__ == "__main__":
    cev_folder = "SEL_events07232025"
    csv_folder = "SEL_events07232025_csv"
    convert_cev_to_csv(cev_folder, csv_folder)