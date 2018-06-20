import numpy as np
import HMM.utils.filesystem_utils as utility


class HiddenMarkovModel:

    word_list = []                      # All the considered words
    likelihood_list = []
    count_tags = np.zeros(17)           # Count global tags
    priori = np.zeros((17, 17))         # A-priori prob matrix
    ending_prob = np.zeros(17)          # Ending prob for each tag
    starting_prob = np.zeros(17)        # Ending prob for each tag

    # List of used tag
    tag_list = (("PROPN", 0), ("PUNCT", 1), ("NOUN", 2), ("ADP", 3), ("DET", 4), ("ADJ", 5), ("AUX", 6),
                ("VERB", 7), ("PRON", 8), ("CCONJ", 9), ("NUM", 10), ("ADV", 11),
                ("INTJ", 12), ("SCONJ", 13), ("X", 14), ("SYM", 15), ("PART", 16))

    def train_hmm(self, train_data, save_model=True):
        print("::: Begin HMM training :::")
        for sent in train_data:
            previous_tag = ""
            for sent_i in range(0, len(sent)):
                tupleOfSent = sent[sent_i]
                if sent_i != 0:
                    self.count_tags = self.update_tag_count(tupleOfSent[1], self.count_tags)
                self.update_prior(tupleOfSent[1], previous_tag)
                if tupleOfSent[0] in self.word_list:
                    # Update statistics for an already present in dict word
                    index = self.word_list.index(tupleOfSent[0])
                    self.likelihood_list[index] = self.update_likelihood(tupleOfSent[1], self.likelihood_list[index])
                    if sent_i == 0:
                        self.starting_prob = self.update_tag_count(tupleOfSent[1], self.starting_prob)
                    if sent_i == (len(sent) - 1):
                        self.ending_prob = self.update_tag_count(tupleOfSent[1], self.ending_prob)
                else:

                    # If the considered word is not present in the dictionary insert it in the dictionary
                    self.word_list.append(tupleOfSent[0])
                    self.likelihood_list.append(self.update_likelihood(tupleOfSent[1]))
                    if sent_i == 0:
                        self.starting_prob = self.update_tag_count(tupleOfSent[1], self.starting_prob)
                    if sent_i == (len(sent) - 1):
                        self.ending_prob = self.update_tag_count(tupleOfSent[1], self.ending_prob)
                previous_tag = tupleOfSent[1]

        print("The dictionary contains " + str(len(self.word_list)) + " elements")

        # Normalize counts and get prior and likelihood prob
        self.normalize_prior()
        self.starting_prob = self.normalize_vect_count_tags(self.starting_prob)
        self.ending_prob = self.normalize_vect_count_tags(self.ending_prob)

        for i in range(0, len(self.likelihood_list)):
            self.likelihood_list[i] = self.normalize_vect_count_tags(self.likelihood_list[i])

        print("::: HMM trained successfully :::")
        if save_model:
            utility.save_model(self.word_list, self.likelihood_list, self.count_tags, self.priori, self.ending_prob, self.starting_prob)
        return

    def load_model(self):
        self.word_list, self.likelihood_list, self.count_tags, self.priori, self.ending_prob, self.starting_prob = utility.load_model()
        return

    def tag(self, splitted_sent=[]):
        # create data structures
        # N rows: states (pos tag),  T columns: observations (words)
        N = 17
        T = len(splitted_sent)
        viterbi = np.zeros((N, T))
        viterbi_backpointer = np.zeros((N, T))

        # Viterbi
        print("Start tagging sent: " + str(splitted_sent))

        # Initialization Step - prob first word start the sent
        current_likelihood = self.get_likelihood_vect(splitted_sent[0])
        viterbi[:, 0] = current_likelihood * self.starting_prob

        for observation_i in range(1, T):
            current_likelihood = self.get_likelihood_vect(splitted_sent[observation_i])
            print("obs " + str(observation_i) + " :" + str(splitted_sent[observation_i]))
            print(current_likelihood)

            for current_state in range(0, N - 1):
                vect = viterbi[:, observation_i - 1] * self.priori[:, current_state]  # Paths to the new position
                vector_prod = vect * current_likelihood
                viterbi[current_state, observation_i] = np.max(vector_prod)
                viterbi_backpointer[current_state, observation_i] = np.argmax(vect)

        # Termination Step - prob last word ending sent with a tag
        viterbi[:, T - 1] = np.max(viterbi[:, T - 1] * self.ending_prob)
        last_index = np.argmax(viterbi[:, T - 1] * self.ending_prob)

        print("\n::: viterbi :::")
        print(viterbi)
        print("\n::: viterbi bp :::")
        print(viterbi_backpointer)

        tags = self.get_viterbi_path(viterbi_backpointer, last_index, T - 1)
        tags.reverse()

        result = []
        k = 0
        for word in splitted_sent:
            result.append((word, tags[k]))
            k = k + 1
        print(result)

        return result

    def get_viterbi_path(self, viterbi_backpointer, last_index, column):
        result = []
        while column >= 0:
            result.append(self.get_tag_from_index(last_index))
            last_index = viterbi_backpointer[int(last_index), column]
            column = column - 1
        print("viterbi path: " + str(result))
        return result

    def update_tag_count(self, tag, array):
        index_count_tag = self.get_tag_index(tag)
        array[index_count_tag] = array[index_count_tag] + 1
        return array

    def update_likelihood(self, tag, params=None):
        if params is None:
            params = np.zeros(17)
        return self.update_tag_count(tag, params)

    def update_prior(self, tag, previous_tag):
        current_tag = self.get_tag_index(tag)
        previous_tag = self.get_tag_index(previous_tag)
        self.priori[previous_tag, current_tag] = self.priori[previous_tag, current_tag] + 1
        return

    def get_likelihood_vect(self, word):
        if word in self.word_list:
            index = self.word_list.index(word)
            return self.likelihood_list[index]
        else:
            print("Word " + word + " not retrieved!")
            return np.zeros(17)

    def normalize_vect_count_tags(self, array):
        if array is None:
            return np.zeros(17)
        for i in range(0, 16):
            array[i] = array[i] / self.count_tags[i]
        return array

    def normalize_prior(self):
        self.priori = self.priori / self.count_tags.reshape(17, 1)
        return

    def get_tag_index(self, tag):
        for t in self.tag_list:
            if tag in t:
                return t[1]

    def get_tag_from_index(self, index):
        for t in self.tag_list:
            if index in t:
                return t[0]
