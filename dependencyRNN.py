import json

from math import sqrt
from collections import OrderedDict, defaultdict

import theano
import theano.tensor as T

import numpy as np

from adagrad import Adagrad

class DependencyRNN:
    '''
    class for dependency RNN for QANTA
    '''
    def __init__(self, d, V, r, answer_idxs, embeddings=None, seed=0):
        '''
        d = dimensionality of embeddings
        V = size of vocabulary
        r = number of dependency relations
        answer_idxs = list of indices into the embeddings matrix for all the answers
        embeddings = pre-trained word embeddings
        seed = for random number generator for reproducivility
        '''
        
        self.d = d

        rnge = sqrt(6) / sqrt(201)
        rnge_we = sqrt(6) / sqrt(51)

        np.random.seed(seed)
        
        #|V| x d embedding matrix
        if embeddings is None:
            self.We = theano.shared(name='embeddings',
                                    value=np.random.rand(V, d) * 2 * rnge_we - rnge_we
                                    ).astype(theano.config.floatX)
        else:
            self.We = theano.shared(name='embeddings',
                                    value=embeddings
                                    ).astype(theano.config.floatX)
            
        #r x d x d tensor (matrix for each dependency relation)
        self.Wr = theano.shared(name='dependencies',
                                value=np.random.rand(r, d, d) * 2 * rnge - rnge
                                ).astype(theano.config.floatX)
        
        #d x d map from embedding to hidden vector
        self.Wv = theano.shared(name='Wv',
                                value=np.random.rand(d, d) * 2 * rnge - rnge
                                ).astype(theano.config.floatX)
        
        #d long bias vector
        self.b = theano.shared(name='b',
                               value=np.zeros(d, dtype=theano.config.floatX))
        
        self.params =  [self.We, self.Wr, self.Wv, self.b]
        
        self.answer_idxs = np.array(answer_idxs, dtype=np.int32)
        self.ans_probabilities = np.ones(self.answer_idxs.shape[0])/(self.answer_idxs.shape[0]-1)
        self.ans_lookup = {j:i for i,j in enumerate(self.answer_idxs)}
        self._answers = {}
        
        self.descender = Adagrad(self.params)

        def normalized_tanh(x):
            '''returns tanh(x) / ||tanh(x)||'''
            return T.tanh(x)/T.sqrt(T.sum(T.tanh(x)**2))
            
        self.f = normalized_tanh

        #need to calculate both the input to its parent node and the error at this step
        def recurrence(n, hidden_states, hidden_sums, cost, x, r, p, wrong_ans, corr_ans):
            '''
            function called below by scan over the nodes in the dependency parse
            
            n - this is the index of the current node
            hidden_states - a list of hidden_states for every node, to be updated
            hidden_sums - sum over the children of dot product of the hidden nodes and the relation matrix
            cost - the total cost so far for this tree
            x - a list of word embeddings (x[n] will access the embedding for the current word)
            r - a list of relation matrices (r[n] will access the current matrix)
            p - a list of parent node indices
            wrong_ans - a list of randomly sampled word embeddings for wrong answers
            corr_ans - the word embedding for the correct answer
            '''
            h_n = self.f(T.dot(self.Wv, x[n]) + self.b + hidden_sums[n])

            update_new = T.set_subtensor(hidden_sums[p[n]], hidden_sums[p[n]]+ T.dot(r[n], h_n))

            update_hidden_new = T.set_subtensor(hidden_states[n], h_n)

            output, updates = theano.scan(fn = lambda xz, xc, h_n: T.maximum(0., 1.-T.dot(xc, h_n) + T.dot(xz, h_n)),
                                         outputs_info = None, sequences = wrong_ans, non_sequences = [corr_ans, h_n])

            output_sum = T.as_tensor_variable(output.sum())

            new_cost = cost + output_sum

            return update_hidden_new, update_new, new_cost

        idxs = T.ivector('idxs')
        x = self.We[idxs]

        rel_idxs = T.ivector('rel_idxs')
        r = self.Wr[rel_idxs]

        p = T.ivector('parents')

        wrong_idxs = T.ivector('wrong_idxs')
        wrong_ans = self.We[wrong_idxs]

        corr_idx = T.iscalar('corr_idx') # index of answer
        corr_ans = self.We[corr_idx]

        hidden_states = T.zeros((idxs.shape[0], d), dtype=theano.config.floatX)
        #needs to be sent_length + 1 to store final sum
        hidden_sums = T.zeros((idxs.shape[0]+1, d), dtype=theano.config.floatX)

        [h, s, cost], updates = theano.scan(fn=recurrence,
                                            sequences=T.arange(x.shape[0]),
                                            outputs_info=[hidden_states,
                                                          hidden_sums,
                                                          T.as_tensor_variable(np.asarray(0, theano.config.floatX))],
                                            non_sequences=[x, r, p, wrong_ans, corr_ans])
        final_states = h[-1]
        self.states = theano.function(inputs=[idxs, rel_idxs, p, wrong_idxs, corr_idx], outputs=final_states)

        final_cost = cost[-1] #no regularization
        gradients = T.grad(final_cost, self.params)
        self.cost_and_grad = theano.function(inputs=[idxs, rel_idxs, p, wrong_idxs, corr_idx], outputs=[final_cost] + gradients)

    def gradient_descent(self, new_gradients):
        self.descender.gradient_descent(*new_gradients)

    #batch consists of tuples of word indices, relation indices, parent indices, and an answer index
    def train(self, batch, num_wrong_ans=100):
        total_cost_and_grad = None
        total_nodes = 0.

        #split data into batches, then into minibatches for multiprocessing

        for datum in batch:
            idxs, rel_idxs, p, corr_idx = datum

            #sample new wrong answers for every point (make sure not to sample the correct answer)
            self.ans_probabilities[self.ans_lookup[corr_idx]] = 0
            wrong_idxs = self.answer_idxs[np.random.choice(self.answer_idxs.shape[0],
                                                           num_wrong_ans,
                                                           False,
                                                           self.ans_probabilities)]
            self.ans_probabilities[self.ans_lookup[corr_idx]] = 1./(self.ans_probabilities.shape[0]-1)

            cost_and_grad = self.cost_and_grad(idxs, rel_idxs, p, wrong_idxs, corr_idx)
            if total_cost_and_grad is None:
                total_cost_and_grad = [0] + [np.zeros(i.shape) for i in cost_and_grad[1:]]
            for i in range(len(cost_and_grad)):
                total_cost_and_grad[i] += cost_and_grad[i]
            total_nodes += len(idxs)

        #update gradients from total_cost_and_grad[1:]
        self.gradient_descent([i/total_nodes for i in total_cost_and_grad[1:]])

        return total_cost_and_grad[0]/total_nodes

    def reset_weights(self):
        self.descender.reset_weights()

    def transform(self, batch, stop_indices=None):
        features = []
        for idxs,rel_idxs,p in batch:
            h = self.states(idxs, rel_idxs, p, [], 0)
            x = np.zeros(self.d)
            count = 0.0
            for i,s in enumerate(h):
                if stop_indices is None or idxs[i] not in stop_indices:
                    x += s
                    count += 1
            features.append(x / count)
            
        return(np.array(features))

    def save(self, filename, answers):
        '''save all the weights and hyperparameters to a file'''
        kwds = {}
        for param in self.params:
            kwds[param.name] = param.get_value()
        kwds['answer_idxs'] = self.answer_idxs
        
        with open(filename, 'wb') as f:
            np.savez(f, **kwds)

        embeddings = self.We.get_value()
        for answer in answers:
            self._answers[answer] = embeddings[answers[answer]].tolist()

        with open(filename + '.json', 'w') as f:
            json.dump(self._answers, f)

    @classmethod
    def load(cls, filename):
        '''load pre-trained weights from a file'''
        with open(filename) as f:
            npzfile = np.load(f)
            
            d = npzfile['embeddings'].shape[1]
            V = npzfile['embeddings'].shape[0]
            r = npzfile['dependencies'].shape[0]

            d = cls(d, V, r, npzfile['answer_idxs'])

            for param in d.params:
                param.set_value(npzfile[param.name])

        with open(filename + '.json') as f:
            d._answers = json.load(f)

        return d

    @property
    def answers(self):
        return self._answers
