import numpy as np


class HiddenMarkovModel:

    hmm_dict = []                   # List of dictionaries
    word_list = []                  # All the considered words
    count_tags = np.zeros(17)       # Count global tags
    priori = np.zeros((17, 17))     # A-priori prob matrix

    def train_hmm(self, train_data):
        print("Begin HMM training")
        for sent in train_data:
            previous_tag = ""
            for tupleOfSent in sent:
                self.update_tag_count(tupleOfSent[1])
                self.get_prior(tupleOfSent[1], previous_tag)
                if self.word_list.__contains__(tupleOfSent[0]):
                    for rec in self.hmm_dict:
                        if rec["lemma"] == tupleOfSent[0]:
                            rec.update(
                                dict(likelihood=self.get_likelihood(tupleOfSent[1], rec["likelihood"]))
                            )
                else:
                    self.hmm_dict.append(
                        dict(lemma=tupleOfSent[0], likelihood=self.get_likelihood(tupleOfSent[1]))
                    )
                    self.word_list.append(tupleOfSent[0])
                previous_tag = tupleOfSent[1]
        print("Count tags: " + str(self.count_tags))

        # Normalize counts and get prior and likelihood prob
        self.normalize_prior()
        i = 0
        for item in self.hmm_dict:
            likelihood = self.normalize_likelihood(item["likelihood"])
            self.hmm_dict[i]["likelihood"] = likelihood
            i = i + 1
        return

    def tag(self, splitted_sent=[]):
        # create data structures
        # N rows: states (pos tag),  T columns: observations (words)
        N = 17
        T = len(splitted_sent)
        viterbi = np.zeros((N, T))
        viterbi_backpointer = np.zeros((N, T))

        # Viterbi
        for observation_i in range(0, T):
            current_likelihood, current_prior = self.get_params_hmm(splitted_sent[observation_i])

            if observation_i == 0:
                viterbi[:, 0] = current_likelihood  # Add the probability that a sentence start with this lemma
                viterbi_backpointer[observation_i] = 0
            else:
                for current_state in range(0, N - 1):
                    vector_prod = viterbi[:, observation_i - 1] * current_prior[current_state]
                    viterbi[current_state, observation_i] = np.max(vector_prod)
                    viterbi_backpointer[current_state, observation_i] = np.argmax(vector_prod)

        # print("\n::: viterbi :::")
        # print(viterbi)
        # print("\n::: viterbi_back pointer :::")
        # print(viterbi_backpointer)

        last_index = np.argmax(viterbi[:, T - 1])

        # list of string
        tags = self.get_viterbi_path(viterbi_backpointer, last_index, T - 1)
        tags.reverse()

        result = []
        k = 0
        for word in splitted_sent:
            result.append((word, tags[k]))
            k = k+1

        return result

    def get_viterbi_path(self, viterbi_backpointer, last_index, column):
        result = []
        while column >= 0:
            result.append(self.get_tag_from_index(last_index))
            last_index = viterbi_backpointer[int(last_index), column]
            column = column - 1
        return result

    def update_tag_count(self, tag):
        index_count_tag = self.get_tag_index(tag)
        self.count_tags[index_count_tag] = self.count_tags[index_count_tag] + 1
        return

    def get_likelihood(self, tag, params=None):
        if params is None:
            params = np.zeros(17)
        tag_index = self.get_tag_index(tag)
        params[tag_index] = params[tag_index] + 1
        return params

    def get_prior(self, tag, previous_tag):
        current_tag = self.get_tag_index(tag)
        previous_tag = self.get_tag_index(previous_tag)
        self.priori[previous_tag, current_tag] = self.priori[previous_tag, current_tag] + 1
        return

    def get_params_hmm(self, word):
        for hmm_item in self.hmm_dict:
            # retrieve probabilities for current word
            if hmm_item["lemma"] == word:
                current_likelihood = hmm_item["likelihood"]
                return current_likelihood, self.priori

        return np.zeros(17), np.zeros((17, 17))

    @staticmethod
    def get_tag_index(tag):
        if tag == "PROPN": return 0
        if tag == "PUNCT": return 1
        if tag == "NOUN": return 2
        if tag == "ADP": return 3
        if tag == "DET": return 4
        if tag == "ADJ": return 5
        if tag == "AUX": return 6
        if tag == "VERB": return 7
        if tag == "PRON": return 8
        if tag == "CCONJ": return 9
        if tag == "NUM": return 10
        if tag == "ADV": return 11
        if tag == "INTJ": return 12
        if tag == "SCONJ": return 13
        if tag == "X": return 14
        if tag == "SYM": return 15
        if tag == "PART": return 16

    @staticmethod
    def get_tag_from_index(index):
        if index == 0: return "PROPN"
        if index == 1: return "PUNCT"
        if index == 2: return "NOUN"
        if index == 3: return "ADP"
        if index == 4: return "DET"
        if index == 5: return "ADJ"
        if index == 6: return "AUX"
        if index == 7: return "VERB"
        if index == 8: return "PRON"
        if index == 9: return "CCONJ"
        if index == 10: return "NUM"
        if index == 11: return "ADV"
        if index == 12: return "INTJ"
        if index == 13: return "SCONJ"
        if index == 14: return "X"
        if index == 15: return "SYM"
        if index == 16: return "PART"

    def normalize_likelihood(self, array):
        for i in range(0, 16):
            array[i] = array[i] / self.count_tags[i]
        return array

    def normalize_prior(self):
        return self.priori / self.count_tags.reshape(17, 1)

