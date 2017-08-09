import numpy as np
import pickle
from collections import defaultdict
from functools import partial

with open('vocabs.pkl', 'rb') as f:
    vocab = {v: k for k, v in pickle.load(f)['text'].items()}
goh = np.load('goh_text.npy').flatten()

x = []
START_IX = 1
END_IX = 2
d = {
    '': 0,
    '<start>': START_IX,
    '<end>': END_IX
}
SEQ_LEN = 1000


def buffered_sequences():
    l = 0
    while l < len(goh):
        seq = ' '.join(vocab[ix] for ix in goh[l])
        yield from (seq[j:j + SEQ_LEN] for j in range(0, len(seq), SEQ_LEN))
        l += 1

max_len = 0

for i, wseq in enumerate(buffered_sequences()):
    if i % 1000 == 0:
        print(i, '/', len(goh))
    cseq = [START_IX]
    for c in wseq:
        if c not in d:
            d[c] = len(d)
        cseq.append(d[c])
    cseq.append(END_IX)
    x.append(np.array(cseq, dtype=np.uint8))
    max_len = max(max_len, len(cseq))

print(max_len)
np.save('c_x.npy', np.array(x))
with open('cvoc.pkl', 'wb') as f:
    pickle.dump(d, f)