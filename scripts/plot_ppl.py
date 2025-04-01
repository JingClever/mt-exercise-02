import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

def read_tsv_file(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df = df.iloc[:-1]
    return df

def merge_model_files(model, input_dir):

    files = os.path.join(input_dir, f"{model}_ppl_dp_*.tsv")
    file_list = glob.glob(files)
    if not file_list:
        print(f"There are no files found with thisn name: {files}")
        return None

    dfs = []
    for file in file_list:
        df = read_tsv_file(file)
        dfs.append(df)
    

    df_model = reduce(lambda left, right: pd.merge(left, right, on="epoch", how="outer"), dfs)
    df_model['epoch'] = pd.to_numeric(df_model['epoch'], errors='coerce')
    df_model.sort_values("epoch", inplace=True)
    return df_model

def save_df_model(df_model, output_file):
    df_model.to_csv(output_file, sep='\t', index=False)
    print("The perplexity for the same model is saved to", output_file)

def plot_train_valid(df_model, output_image):
    print(df_model)

    train_cols = [col for col in df_model.columns if col.startswith("train_ppl_")]
    valid_cols = [col for col in df_model.columns if col.startswith("valid_ppl_")]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))


    for col in train_cols:
        axes[0].plot(df_model['epoch'], df_model[col], label=col)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Perplexity")
    axes[0].set_title("Training Perplexity per Epoch")
    axes[0].legend()
    axes[0].grid(True)

    for col in valid_cols:
        axes[1].plot(df_model['epoch'], df_model[col], label=col)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Perplexity")
    axes[1].set_title("Validation Perplexity per Epoch")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_image)
    print(f"Plot saved to {output_image}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Merge TSV files for a single model with different dropout settings and plot perplexity curves."
    )
    parser.add_argument("--model", required=True, help="Model name (e.g., LSTM)")
    parser.add_argument("--input_dir", default=".", help="Directory containing the TSV files")
    parser.add_argument("--output", default="merged.tsv", help="Output file name for the merged TSV")
    parser.add_argument("--plot_output", default="model_ppl.png", help="Output image file for the combined plot")
    args = parser.parse_args()

    df_model = merge_model_files(args.model, args.input_dir)
    if df_model is None:
        return

    save_df_model(df_model, args.output)
    print("The perplexity for one model:")
    print(df_model.head())

    plot_train_valid(df_model, args.plot_output)

if __name__ == "__main__":
    main()
