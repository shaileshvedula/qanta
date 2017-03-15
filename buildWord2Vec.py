import gensim
import json
import numpy as np

if __name__ == '__main__':

	filein = open("hist_split.json", 'r')

        data = json.loads(filein.read())

	train_data = data['train']

	sentences = []	

	for x in train_data:
		temp = []
		for y in x[0]:
			if y[0]:
				temp.append(y[0])

		sentences.append(temp)

	model = gensim.models.Word2Vec(size =100, window = 5, min_count=1)

	model.build_vocab(sentences)

	alpha, min_alpha, passes = (0.025, 0.001, 20) 

	alpha_delta = (alpha - min_alpha) / passes 

	for epoch in range(passes): 

		model.alpha, model.min_alpha = alpha, alpha 

		model.train(sentences)

		print('completed pass %i at alpha %f' % (epoch + 1, alpha)) 

		alpha -= alpha_delta

		np.random.shuffle(sentences)

	model.save("word2vec.model")
