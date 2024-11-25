import csv
import os
import sys

# Ensure the script receives at least one argument
if len(sys.argv) < 2:
    print("Usage: python script_name.py <log_file> [optional_column_name] [optional_column_value]")
    sys.exit(1)

# Get the input file name, output directory, optional column name, and value from the command-line arguments
input_file = sys.argv[1]
optional_column_name = sys.argv[2] if len(sys.argv) > 2 else None
optional_column_value = sys.argv[3] if len(sys.argv) > 3 else None
#output_dir = sys.argv[2] if len(sys.argv) > 2 else "../datasets/parsed/"
if optional_column_name:
    output_dir = "../datasets/parsed/"
else:
    output_dir = "../datasets/labelless/"

print("Input file",input_file)
print("optional_column_name",optional_column_name)
print("optional_column_value",optional_column_value)
print("Output file",output_dir)

# Check if the input file exists
if not os.path.isfile(input_file):
    print(f"Error: File '{input_file}' not found.")
    sys.exit(1)

# Read the log data from the file
try:
    with open(input_file, "r", encoding="utf-8") as file:
        log_data = file.readlines()
except Exception as e:
    print(f"Error reading the file: {e}")
    sys.exit(1)

# List to hold parsed logs
parsed_logs = []

# Process each log line
for line in log_data:
    line = line.strip()  # Remove leading/trailing whitespace
    if line.startswith('"') and line.endswith('"'):
        line = line[1:-1]  # Remove the outermost quotes
    log_entry = {"query": line}
    if line.startswith("Ejecutando consulta: SELECT * FROM users WHERE name = "):  # Check for the prefix
        # Remove the prefix and clean up the line
        cleaned_query = line.replace("Ejecutando consulta: SELECT * FROM users WHERE name = ", "").strip()
        log_entry = {"query": cleaned_query}
    if optional_column_name and optional_column_value:
        log_entry[optional_column_name] = optional_column_value  # Add the custom column and value
    parsed_logs.append(log_entry)

# Check if logs were parsed
if not parsed_logs:
    print("No valid log entries found. Please check your input file.")
    sys.exit(1)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Strip the file extension and replace it with .csv
base_name = os.path.splitext(os.path.basename(input_file))[0]
output_file = os.path.join(output_dir, f"{base_name}.csv")

# Write to CSV
try:
    with open(output_file, mode="w", newline="", encoding="utf-8") as csv_file:
        # Include custom column in the header if provided
        fieldnames = ["query"]
        if optional_column_name and optional_column_value:
            fieldnames.append(optional_column_name)
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(parsed_logs)
    print(f"Logs have been written to {output_file}")
except Exception as e:
    print(f"Error writing the CSV file: {e}")
    sys.exit(1)
