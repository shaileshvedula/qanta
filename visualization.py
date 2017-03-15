import numpy as np
import pandas as pd
from dependencyRNN import DependencyRNN as DRNN
from sklearn.manifold import TSNE

x = DRNN.load("random_init.npz")

dictionary = x.answers

words = []
embeddings = []

for key, value in dictionary.iteritems():
    words.append(key)
    embeddings.append(value)

embeddings_matrix = np.array(embeddings)

tsne = TSNE(n_components=2, perplexity=30.0)
X_reduced = tsne.fit_transform(embeddings_matrix)

fileout = open("reduced_embeddings.tsv", 'w')
for row in X_reduced:
    fileout.write("{0}\t{1}".format(str(row[0]), str(row[1])) + '\n')
fileout.close()

fileout = open("words.txt", 'w')
for word in words:
    fileout.write(word+ '\n')
fileout.close()

