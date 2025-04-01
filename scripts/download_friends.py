from convokit import Corpus, download

corpus = Corpus(filename=download("friends-corpus"))
file_path = './data/friends/raw/friends.txt'

with open(file_path, 'w') as f:
    for utt in corpus.iter_utterances():
        f.write(utt.text + '\n')

print('Done!')

