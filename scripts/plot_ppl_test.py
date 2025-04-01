import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def extract_test_perplexity(file_path, model_name):
    dropout = file_path.split("_dp_")[-1].replace(".tsv", "")
    label = f"{model_name}_dp_{dropout}"

    with open(file_path, 'r') as f:
        lines = f.readlines()
        last_line = lines[-1].strip()
        if last_line.startswith("test"):
            _, test_ppl = last_line.split("\t")
            return label, float(test_ppl)
        else:
            print(f"Error: last line is not a test perplexity {file_path}")
            return None

def main():
    parser = argparse.ArgumentParser(description="Plot test perplexities under different dropout settings.")
    parser.add_argument("--input_dir", type=str, default=".", help="Directory containing *_ppl_dp_*.tsv files")
    parser.add_argument("--model", type=str, required=True, help="Model name prefix (e.g., LSTM, GRU, Transformer)")
    parser.add_argument("--output_plot", type=str, default="test_ppl.png", help="Output plot")
    args = parser.parse_args()

    pattern = os.path.join(args.input_dir, f"{args.model}_ppl_dp_*.tsv")
    files = glob.glob(pattern)

    results = []
    for file in files:
        result = extract_test_perplexity(file, args.model)
        if result:
            results.append(result)

    if not results:
        print("No test perplexity data found.")
        return

    results.sort(key=lambda x: float(x[0].split("_dp_")[1]))

    labels, perplexities = zip(*results)
    colors = cm.viridis(np.linspace(0, 1, len(perplexities)))
    plt.figure(figsize=(10, 6))
    plt.bar(labels, perplexities, color=colors)
    plt.xlabel("Dropout Setting")
    plt.ylabel("Test Perplexity")
    plt.title(f"Test Perplexity for {args.model} with Different Dropouts")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(args.output_plot)
    plt.show()

    print(f"Plot saved to {args.output_plot}")

if __name__ == "__main__":
    main()
