import re
import csv
import sys
import os

# Get the input file name from the command-line argument
# Example usage
input_file = '../datasets/originals/httpd_logs_with_behavior.csv'  # Update this to the path of your input file

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

# Regular expression to parse log entries (including optional protocol)
log_pattern = re.compile(
    r'^"(?P<ip>[0-9\.]+) - - \[(?P<date>.*?)\] "(?P<method>[A-Z]*) ?(?P<endpoint>[^"]*?) ?(?P<protocol>HTTP/[\d\.]*)?" (?P<status>\d{3}) (?P<size>\d*|-)",(?P<flag>.+)$'
)


# List to hold parsed logs
parsed_logs = []

# Process each log line
unmatched_lines = []
for line in log_data:
    # Remove any surrounding quotes from the line
    line = line.strip()
    line = line.replace('""', '"')

    if line.startswith('"') and line.endswith('"'):
        line = line[1:-1]  # Remove the outermost quotes
    
    # Print the cleaned line for debugging
    print("Cleaned line:", line)

    match = log_pattern.match(line)
    if match:
        log_entry = match.groupdict()
        log_entry.pop("protocol", None)
        # Exclude the protocol from the parsed log entry
        #del log_entry["protocol"]
    
        # Convert size to integer where applicable
        log_entry["size"] = int(log_entry["size"]) if log_entry["size"].isdigit() else None
        
        # Add the log entry to the list
        parsed_logs.append(log_entry)
    else:
        print("Ignoring line: ", line)

# Check if logs were parsed
if not parsed_logs:
    print("No valid log entries found. Please check your input file.")
    sys.exit(1)

# Create output directory if it doesn't exist
output_dir = "../datasets/parsed/"
os.makedirs(output_dir, exist_ok=True)
file_name, file_extension = os.path.splitext(input_file)
# Output file name
output_file = os.path.join(output_dir, os.path.basename(file_name) + "_parsed.csv")

# Write to CSV
with open(output_file, mode="w", newline="") as csv_file:
    # Include the flag column in the header
    fieldnames = ["ip", "date", "method", "endpoint", "status", "size", "flag"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerows(parsed_logs)

print(f"Logs have been written to {output_file}")
