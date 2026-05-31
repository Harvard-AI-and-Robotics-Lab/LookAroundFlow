import json
import argparse
import os

def extract_and_format_metrics(json_file_path):
    """
    Loads a JSON file, extracts specific metrics, and formats them.

    Args:
        json_file_path (str): The path to the input JSON file.
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        metrics = data.get("metrics", {})

        bleu_4 = metrics.get("BLEU-4", 0.0)
        meteor = metrics.get("METEOR", 0.0)
        rouge_l = metrics.get("ROUGE-L", 0.0)

        # print(f"BLEU-4: {bleu_4:.2f}")
        # print(f"METEOR: {meteor:.2f}")
        # print(f"ROUGE-L: {rouge_l:.2f}")

    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{json_file_path}' is not a valid JSON file.")
    except KeyError:
        print("Error: The JSON file is missing the 'metrics' key.")

    return f' & {bleu_4:.2f} & {meteor:.2f} & {rouge_l:.2f}'

# This block runs when the script is executed directly
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extract and format BLEU-4, METEOR, and ROUGE-L metrics from a JSON file."
    )

    parser.add_argument("--csv_file", type=str, default='', help="The path to the input CSV file.")
    parser.add_argument("--json_file", type=str, default='', help="The path to the input JSON file.")
    parser.add_argument("--output", type=str, default='', help="output file path")

    args = parser.parse_args()

    output_str = ''
    # load the first line in csv_file without returning symbols
    with open(args.csv_file, 'r') as f:
      first_line = f.readline().strip()
    
    output_str += first_line
    output_str += extract_and_format_metrics(args.json_file)

    print(args.output)
    print(output_str)
    with open(args.output, 'w') as f:
        f.write(output_str + '\n')