import csv
import sys

def csv_to_text(input_csv_file, output_text_file):
    # Open the CSV file for reading
    with open(input_csv_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)  # Read all rows into a list

    # Count the number of rows
    num_rows = len(rows)

    # Open the text file for writing
    with open(output_text_file, 'w') as text_file:
        # Write the number of rows as the first line of the text file
        text_file.write(str(num_rows) + '\n')

        # Iterate over the rows and write them to the text file
        for row in rows:
            # Replace commas with tabs and write the row to the text file
            text_file.write('\t'.join(row) + '\n')

if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python script.py input_csv_file output_text_file")
        sys.exit(1)

    # Get the input CSV file name from the command line argument
    input_csv_file = sys.argv[1]

    # Get the output text file name from the command line argument
    output_text_file = sys.argv[2]

    # Call the function to convert CSV to text
    csv_to_text(input_csv_file, output_text_file)