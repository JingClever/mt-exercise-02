# MT Exercise 2: Pytorch RNN Language Models

This repo shows how to train neural language models using [Pytorch example code](https://github.com/pytorch/examples/tree/master/word_language_model). Thanks to Emma van den Bold, the original author of these scripts. 

# Requirements

- This only works on a Unix-like system, with bash.
- Python 3 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

# Modifications

**Add `download_friends.py` Script to Download the Corpus for Task 1**:
- Adding a new script `download_friends.py` to download Friends from the Cornell Movie-Dialogs Corpus used for model training. Key changes include:
  - **Install the package**: Install `convokit` package to download Friends corpus.
  - **Download the data to txt file**: Write a short script to write the corpus to a txt file.

**Change `download_data.sh` Script for Task 1**:
- Updated the `download_data.sh` to manage data preprocessing for the Friends Corpus:
  - Make a new directory to store the raw and processed data for Friends corpus.
  - Run the python script `download_friends.py` to download the raw data.
  - Cut the segments to 8,000 to make sure the proper vocabulary size with 5000.
  - Change the previous code to preprocess the raw data.
  - Split the preprocessed dataset to training, valid and testing with 9297, 1200, 1200 segments respectively.
  - During preprocessing, `sacremoses.MosesTokenizer`  escapes characters like `&`, `'`, `"` into `&amp;`, `&apos;`, `&quot;`. If you want to convert them into origincal characters, just set `t = tokenizer.tokenize(line, escape = False)`

**Change `train.sh` Script for Task 1**:
- Adapted `train.sh` to accommodate the infrastructure and dataset changes:
  - Changed the data directory in the script to `$data/friends` to align with the new dataset.
  - Added the `--mps` flag to enable GPU support on macOS with Apple Silicon, enhancing training performance on compatible systems.

**Change `train.sh` Script for Task 2**:
- Added a `logs` directory to store training logs under different dropout rates.
- Training epochs increased from 40 to 50.
- Increased embedding size and number of hidden units from 200 to 256.
- Added logging for each dropout rate training session with specific files.
- Models are saved with names that includes model name and dropout, e.g., `model_ppl_dp_0.pt`.

**Change`generate.sh` Script for Task 1**:
- Updated the script to ensure compatibility and functionality with the trained model and data:
  - Modified directory paths to correctly point to the Friends dataset and model files.

**Change`main.py` Script for Task 2**
- Copy `main.py` as `main_ex2.py` to make a modification based on this script.
- Added `--ppl-log` argument to log train/validation/test perplexities to a `.tsv` file.
- Adjusted the `train()` function to return epoch-level training loss so that training perplexity can be computed and logged.
- Used `.tsv` format with columns `epoch`, `train_ppl_{dropout}`, `valid_ppl_{dropout}`, and a test line at the end.

**Add`plot_ppl.sh` Script to Create Tables for the Three Perplexities for Task 2**
- This bash file is to run plot python file.
- A new directory called results is created to store the final table file and plot results.
- Just run the code in the script.

**Add`plot_ppl.py` Script to Create Tables for the Three Perplexities for Task 2**
- This script firstly combine all models-generated result into one `.tsv` file excluding test perplexity.
- Then this script draw the line plot for the train and validation perplexity change. 

**Add `plot_ppl_test.py` Script to Visualize Test Perplexity for Task 2**
- This script extracts the last line of log file to build a bar plot for test perplexity change.


# Steps

Clone this repository in the desired place:

    git clone https://github.com/JingClever/mt-exercise-02.git
    cd mt-exercise-02

Create a new virtualenv that uses Python 3. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh

**Important**: Then activate the env by executing the `source` command that is output by the shell script above.

Download and install required software:

    ./scripts/install_packages.sh

Download and preprocess data:

    ./scripts/download_data.sh

Train a model:

    ./scripts/train.sh

The training process can be interrupted at any time, and the best checkpoint will always be saved.

Generate (sample) some text from a trained model with:

    ./scripts/generate.sh

Plot the train, valid and test perplexity:

    ./scripts/plot_ppl.sh

    


