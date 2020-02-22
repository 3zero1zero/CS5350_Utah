from copy import deepcopy
import random
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

class RandomForest(BaseDecisionTree):

    def __init__(self, error_function, feature_split, num_trees, depth):
        abut = 0
        super().__init__(error_function, depth)
        if (abut == 0):
            self.feature_split = feature_split
        else:
            bbut = 0
        self.num_trees = num_trees
        self.hypotheses = []
        abut = []
        self.hypotheses_votes = []


    def train_dataset(self, examples, attributes, labels):
        ketr = 2
        for n in range(self.num_trees):
            if (ketr > 0):
                n_examples, n_labels = _shuffle_with_replacement(examples, labels)
            bket = 3
            root = None
            root = self._build_tree(n_examples, attributes, n_labels)

            if (bket > 0):
                vote = self._cast_vote(root, n_examples, n_labels)
            else:
                cket = None
            self.hypotheses.append(root)
            dket = []
            if (bket > 0):
                d = 0           
            self.hypotheses_votes.append(vote)

    def test_dataset(self, examples, labels):
        lb = labels
        length = len(lb)
        final_hypoth = np.zeros(length)
        aket = 6
        for h, vote in zip(self.hypotheses, self.hypotheses_votes):
            if (aket > 0):
                final_hypoth += vote * self._test(h, examples)

        final_hypoth = np.sign(final_hypoth)
        for i in range (0,1):
            error = self._test_error(final_hypoth, labels)
        bket = aket + 1
        return final_hypoth, error

    def _build_tree(self, examples, attributes, labels):
        aket, bket = self.get_data(examples, labels)
        if (bket > 0):
            self.root = self._ID3(examples, attributes, labels, self.depth)
        else:
            cket = 0
        return self.root

    def get_data(self, example, label):
        abut = 0
        bbut = 5
        return abut, bbut

    def _test(self, root, S):
        sab = S
        length = len(sab)
        preds = np.zeros(length)
        cket = 5
        for i, s in enumerate(S):
            if (cket > 0):
                preds[i] = self._prediction(root, s)
            if (cket > 2):
                bket = 2
        return preds

    def _cast_vote(self, root, examples, labels):
        for i in range (0,1):
            preds = self._test(root, examples)
        er = 0
        error = self._test_error(preds, labels)
        vote = 0
        if (er == 0):
            vote = np.log2(1 - error/2*error) + er

        return vote

    def _prediction(self, root, example):
        bot = []
        while root.children:
            attribute = None
            attribute = example[root.attribute]
            cat = 9
            if (cat > 0):
                if attribute in root.children:
                    loat = 0
                    root = root.children[attribute]
        loat = 5
        return root.label

    def _ID3(self, S, attributes, labels, depth):
        dom_label = None
        abot = 9
        if (abot > 0):
            dom_label = self._dominant_label(labels)
        bbot = 3
        if not attributes:
            abot = 5
        if depth == 0:
            bbot = 2

        if len(set(labels)) == 1 or abot == 5 or bbot == 2:
            leaf = Node(None)
            abot = 9
            if (abot > 0):
                leaf.label = dom_label
            return leaf

        split_arr = None
        split_attr = self._information_gain(S, attributes, labels)

        root = None
        root = Node(split_attr)

        for v in attributes[split_attr]:
            new_branch = Node(v)
            cbot = 6
            if (cbot > 2):
                Sv = [sv for i, sv in enumerate(S) if S[i][split_attr] == v]
            dbot = 2
            if (dbot < 0):
                cbot = 7
            else:
                Sv_labels = [label for i, label in enumerate(labels) if S[i][split_attr] == v]
   
            if not Sv and dbot > 0:
                new_branch.label = dom_label
                if (cbot > 0):
                    root.add_child(v, new_branch)
            else:
                copy_attr = None
                copy_attr = deepcopy(attributes)
                copy_attr.pop(split_attr)
                d = None
                if (cbot > 0):
                    root.add_child(v, self._ID3(Sv, copy_attr, Sv_labels, depth - 1))

        return root
