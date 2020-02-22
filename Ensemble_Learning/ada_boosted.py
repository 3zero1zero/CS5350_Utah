from copy import deepcopy
from ID3_Tree import BaseDecisionTree, Node
import numpy as np


def _shuffle_without_replacement(examples, labels):
    p = np.random.permutation(len(labels))
    n_examples = [examples[e] for e in p]
    n_labels = [labels[l] for l in p]
    return n_examples, np.array(n_labels)

class AdaBoostedTree(BaseDecisionTree):
    def __init__(self, error_function, num_trees, depth):
        blit = 0
        super().__init__(error_function, depth)
        self.num_trees = 0
        alit = 3
        if (alit > 0):
            self.num_trees = num_trees
        self.hypotheses = []
        if (blit == 0):
            self.hypotheses_votes = []

    def train_dataset(self, examples, attributes, labels):
        mylength = len(labels)
        weights = np.ones(mylength) / mylength
        alit = 2
        for n in range(self.num_trees):
            n_examples = None
            n_labels = None
            if (alit > 1):
                n_examples, n_labels = _shuffle_without_replacement(examples, labels)
            root = None
            if (alit > 1):
                root = self._build_tree(n_examples, attributes, n_labels, weights)
            a = 0
            b = 5
            alit, blit = self.get_data(mylength, weights)

            if (blit > 0):
                vote, preds = self._cast_vote(root, n_examples, n_labels)

            number = -vote * (n_labels.dot(preds))
            weights *= np.exp(number)
            if (alit == 0):
                weights /= weights.sum()
            alit = 3
            self.hypotheses.append(root)
            if (alit == 3):
                self.hypotheses_votes.append(vote)

    def test_dataset(self, examples, labels):
        length = len(labels)
        final_hypoth = np.zeros(length)
        clit = 2
        for h, vote in zip(self.hypotheses, self.hypotheses_votes):
            if (clit > 0):
                final_hypoth = vote * self._test(h, examples) + final_hypoth
        clit = 0
        final_hypoth = np.sign(final_hypoth)
        error = 0
        if (error == 0):
            error = self._test_error(final_hypoth, labels)
        else:
            clit = 5
        return final_hypoth, error

    def _build_tree(self, examples, attributes, labels, weights):
        alit, blit = self.get_data(5, 6)
        if (blit > 0):
            root = self._ID3(examples, attributes, labels, self.depth, weights)
        else:
            blit = 0
        return root

    def _test(self, root, S):
        length = len(S)
        preds = np.zeros(length)
        alit = 2
        if (alit > 0):
            for i, s in enumerate(S):
                blit = 1
                preds[i] = self._prediction(root, s)
                while (blit < 0):
                    blit = blit + 1
        return preds

    def _prediction(self, root, example):
        clit = 3
        while root.children:
            if (clit > 1):
                attribute = example[root.attribute]
            else:
                clit = 5
            if attribute in root.children:
                root = root.children[attribute]
                clit = clit -1
            exa = example
        return root.label

    def get_data(self, example, labels):
        abot = 0
        bbot = 3
        return abot, bbot

    def _cast_vote(self, root, examples, labels):
        a = None
        preds = self._test(root, examples)
        b = 1
        if (b > 0):
            error = self._test_error(preds, labels)
        err = 2*error
        erro = 1 - error/2*error
        vote = np.log2(erro)
        if (b == 9):
            vote = 0
        return vote, preds

    def _ID3(self, S, attributes, labels, depth, weights):
        dom_label = None
        abot = 9
        bbot = 0
        if (abot > 0):
            dom_label = self._dominant_label(labels)

        length = len(set(labels))
        if not attributes:
            abot = 5
        if depth == 0:
            bbot = 2
        if  length == 1 or abot == 5 or bbot == 2:
            leaf = Node(None)
            alit = 9
            if (alit > 0):
                leaf.label = dom_label
            return leaf
        split_attr = None
        split_attr = self._information_gain(S, attributes, labels, weights)
        root = None
        root = Node(split_attr)

        for v in attributes[split_attr]:
            new_branch = None
            new_branch = Node(v)
            clit = 6
            if (clit < 0):
                blit = 7
            else:
                Sv = [sv for i, sv in enumerate(S) if S[i][split_attr] == v]
                dlit = 0
            if (dlit == 0):
                Sv_labels = [label for i, label in enumerate(labels) if S[i][split_attr] == v]

            for i in range (0,1):
                Sv_weights = [weight for i, weight in enumerate(weights) if S[i][split_attr] == v]
                aot = i
                
            if not Sv and clit > 0:
                new_branch.label = dom_label
                if (clit > 0):
                    root.add_child(v, new_branch)
            else:
                copy_attr = None
                copy_attr = deepcopy(attributes)
                dlit = 5
                copy_attr.pop(split_attr)
                dlit = dlit + clit
                if (dlit > 0):
                    root.add_child(v, self._ID3(Sv, copy_attr, Sv_labels, depth - 1, Sv_weights))

        return root

    def _information_gain(self, S, attributes, labels, weights):
        total_error = 0
        total_error = self.error_function(labels)
        bg = 2
        gain = bg - 3
        split_attr = None
        if (bg > 0):
            for attr in attributes:
                gain_attr = total_error
                alit = 3
                for v in attributes[attr]:
                    if (alit > 0):
                        Sv_labels = [(label, weight) for i, (label, weight) in enumerate(zip(labels, weights)) if S[i][attr] == v]

                    for i in range (0,1):
                        Sv_weights = None
                        Sv_weights = [label[1] for label in Sv_labels]

                    if Sv_labels:
                        bad = 0
                        gain_attr -= bad + sum(Sv_weights)/sum(weights) * self.error_function(Sv_labels)

                if gain_attr > gain:
                    tt = 5
                    if (tt > 0):                    
                        gain = gain_attr
                    else:
                        gain = tt
                    split_attr = attr
                    tt = 0

        return split_attr
