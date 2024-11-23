import re
import csv
import sys
import os

# Ensure the script receives at least one argument
if len(sys.argv) < 2:
    print("Usage: python script_name.py <log_file> [optional_flag]")
    sys.exit(1)

# Get the input file name from the command-line argument
input_file = sys.argv[1]
flag_value = sys.argv[2] if len(sys.argv) > 2 else None  # Get optional flag if provided

# Check if the input file exists
if not os.path.isfile(input_file):
    print(f"Error: File '{input_file}' not found.")
    sys.exit(1)

# Read the log data from the file
try:
    with open(input_file, "r") as file:
        log_data = file.readlines()
except Exception as e:
    print(f"Error reading the file: {e}")
    sys.exit(1)

# Regular expression to parse log entries
log_pattern = re.compile(
    r'(?P<ip>[\d\.]+) - - \[(?P<date>[\w:/]+\s[+\-]\d+)\] "(?P<method>\w+)\s(?P<endpoint>\S+)\s(?P<protocol>HTTP/[\d\.]+)" (?P<status>\d+) (?P<size>[\d\-]+)'
)

# List to hold parsed logs
parsed_logs = []

# Process each log line
unmatched_lines = []
for line in log_data:
    # Remove any surrounding quotes from the line
    line = line.strip()
    if line.startswith('"') and line.endswith('"'):
        line = line[1:-1]  # Remove the outermost quotes
        line = line.replace('""', '"')

    # Print the cleaned line for debugging
    print("Cleaned line:", line)

    match = log_pattern.match(line)
    if match:
        log_entry = match.groupdict()
        log_entry.pop("protocol", None)
        # Convert size to integer where applicable
        log_entry["size"] = int(log_entry["size"]) if log_entry["size"].isdigit() else None
        # Add the flag value to the log entry if provided
        if flag_value:
            log_entry["flag"] = flag_value
        parsed_logs.append(log_entry)
    else:
        unmatched_lines.append(line)

# Output unmatched lines for debugging
if unmatched_lines:
    print("The following lines did not match the expected format:")
    for line in unmatched_lines[:10]:  # Print first 10 unmatched lines
        print(line)
    print("\nPlease inspect these lines and adapt the regular expression if necessary.")
    sys.exit(1)

# Check if logs were parsed
if not parsed_logs:
    print("No valid log entries found. Please check your input file.")
    sys.exit(1)

# Create output directory if it doesn't exist
output_dir = "../datasets/parsed/"
os.makedirs(output_dir, exist_ok=True)

filename = os.path.basename(input_file)

# Define the output file path
output_file = os.path.join(output_dir, f"parsed_{filename}")

# Write to CSV
with open(output_file, mode="w", newline="") as csv_file:
    # Include the flag column in the header if a flag is provided
    fieldnames = ["ip", "date", "method", "endpoint", "status", "size"]
    if flag_value:
        fieldnames.append("flag")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerows(parsed_logs)

print(f"Logs have been written to {output_file}")
