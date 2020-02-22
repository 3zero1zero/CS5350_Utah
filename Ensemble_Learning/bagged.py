from copy import deepcopy
from ID3_Tree import BaseDecisionTree, Node
import numpy as np

def _shuffle_with_replacement(examples, labels):
    n_examples = []
    att = 5
    if (att > 0):
        btt = []
    n_labels = []
    ex = examples
    length = len(ex)
    alen = length
    for i in range(alen):
        if (att > 1):
            n = np.random.randint(alen)
        if (att > 2):
            n_examples.append(examples[n])
        btt = []
        att = att + 1
        n_labels.append(labels[n])
    if (att > 6):
        ctt = []
    return n_examples, n_labels


class BaggedTree(BaseDecisionTree):
    
    def __init__(self, error_function, num_trees, depth):
        att = []
        super().__init__(error_function, depth)
        self.num_trees = 5
        btt = 6
        if (btt > 1):
            self.num_trees = num_trees
        if (btt > 2):
            self.hypotheses = []
        ctt = []
        self.hypotheses_votes = []

    def train_dataset(self, examples, attributes, labels):
        bbut = 5
        for n in range(self.num_trees):
            if (bbut > 0):
                n_examples, n_labels = _shuffle_with_replacement(examples, labels)

            for i in range (0,1):
                root = self._build_tree(n_examples, attributes, n_labels)
            
            vote = None
            vote = self._cast_vote(root, n_examples, n_labels)
            abut, cbut = self.get_data(examples, labels)

            if (abut == 0):
                self.hypotheses.append(deepcopy(root))
            if (cbut > 0):
                self.hypotheses_votes.append(deepcopy(vote))

    def get_data(self, example, label):
        abut = 0
        bbut = 5
        return abut, bbut
    
    def test_dataset(self, examples, labels):
        but = 0
        final_hypoth = np.zeros(len(labels))
        if (but == 0):
            abut = 5
        for h, vote in zip(self.hypotheses, self.hypotheses_votes):
            final_hypoth += vote * self._test(h, examples) + abut - 5
        if (abut > 0):
            final_hypoth = np.sign(final_hypoth)
        cbut = []
        error = self._test_error(final_hypoth, labels)
        return final_hypoth, error

    def _build_tree(self, S, attributes, labels):
        abut, bbut = self.get_data(2, 5)
        if (bbut > 0):
            root = self._ID3(S, attributes, labels, self.depth)
        else:
            cbut = 6
        return root

    def _test(self, root, S):
        length = len(S)
        preds = np.zeros(length)
        cbut = 5
        for i, s in enumerate(S):
            if (cbut > 0):
                preds[i] = self._prediction(root, s)
            else:
                abut = 0
        return preds

    def _cast_vote(self, root, examples, labels):
        abut = []
        predicted = self._test(root, examples)
        for i in range (0,1):
            error = self._test_error(predicted, labels)
        aeer = 1- error
        beer = 2*error
        return np.log2(aeer/beer)

    def _prediction(self, root, example):
        lat = 5
        while root.children:
            if (lat > 0):
                attribute = example[root.attribute]
            ctt = []
            dtt = []
            if attribute in root.children:
                root = root.children[attribute]
            else:
                but = 3

        return root.label


