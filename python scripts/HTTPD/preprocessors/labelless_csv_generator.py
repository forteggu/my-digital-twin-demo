import csv
import os
import sys

def remove_column_to_csv(input_file, output_dir):
    # Extract the filename
    filename = os.path.basename(input_file)

    # Define the output file path
    output_file = os.path.join(output_dir, f"labelless_{filename}")

    # Read the CSV, remove the last column, and write the updated data to a new CSV
    try:
        with open(input_file, mode="r") as infile, open(output_file, mode="w", newline="") as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            for row in reader:
                # Remove the last column from each row
                updated_row = row[:-1]
                writer.writerow(updated_row)

        print(f"CSV with last column removed has been saved to {output_file}")
    except Exception as e:
        print(f"An error occurred processing {input_file}: {e}")

def process_directory(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        input_file = os.path.join(input_dir, filename)
        if os.path.isfile(input_file) and filename.endswith(".csv"):
            print(f"Processing file: {input_file}")
            remove_column_to_csv(input_file, output_dir)

if __name__ == "__main__":
    # Command-line arguments: script.py <input_directory> <output_directory>
    if len(sys.argv) != 3:
        print("Usage: script.py <input_directory> <output_directory>")
    else:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
        
        process_directory(input_dir, output_dir)
