import numpy as np
import pickle


class HiddenMarkovModel:

    hmm_dict = []                       # List of dictionaries
    word_list = []                      # All the considered words
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
                self.update_tag_count(tupleOfSent[1])
                self.update_prior(tupleOfSent[1], previous_tag)
                if tupleOfSent[0] in self.word_list:

                    # Update statistics for an already present in dict word
                    for rec in self.hmm_dict:
                        if rec["lemma"] == tupleOfSent[0]:
                            likelihood = rec.get("likelihood")
                            rec.update(dict(likelihood=self.update_likelihood(tupleOfSent[1], likelihood)))
                            if sent_i == 0:
                                self.update_starting_prob(tupleOfSent[1])
                            if sent_i == (len(sent) - 1):
                                self.update_ending_prob(tupleOfSent[1])
                else:

                    # If the considered word is not present in the dictionary insert it in the dictionary
                    self.hmm_dict.append(dict(lemma=tupleOfSent[0], likelihood=self.update_likelihood(tupleOfSent[1])))
                    self.word_list.append(tupleOfSent[0])

                    if sent_i == 0:
                        self.update_starting_prob(tupleOfSent[1])
                    if sent_i == (len(sent) - 1):
                        self.update_ending_prob(tupleOfSent[1])

                previous_tag = tupleOfSent[1]

        print("The dictionary contains " + str(len(self.word_list)) + " elements")

        # Normalize counts and get prior and likelihood prob
        self.normalize_prior()
        self.normalize_ending_prob()
        self.normalize_starting_prob()

        i = 0
        for item in self.hmm_dict:
            self.hmm_dict[i]["likelihood"] = self.normalize_vect_count_tags(item["likelihood"])
            i = i + 1
        print("::: HMM trained successfully :::")
        if save_model:
            with open("models/hmm_dict.file", "wb") as f:
                pickle.dump(self.hmm_dict, f, pickle.HIGHEST_PROTOCOL)
            with open("models/count_tags.file", "wb") as f:
                pickle.dump(self.count_tags, f, pickle.HIGHEST_PROTOCOL)
            with open("models/priori.file", "wb") as f:
                pickle.dump(self.priori, f, pickle.HIGHEST_PROTOCOL)
            with open("models/ending_prob.file", "wb") as f:
                pickle.dump(self.ending_prob, f, pickle.HIGHEST_PROTOCOL)
            with open("models/starting_prob.file", "wb") as f:
                pickle.dump(self.starting_prob, f, pickle.HIGHEST_PROTOCOL)
            print("::: HMM Model saved :::")
        return

    def load_model(self):
        with open("models/hmm_dict.file", "rb") as f:
            self.hmm_dict = pickle.load(f)
        with open("models/count_tags.file", "rb") as f:
            self.count_tags = pickle.load(f)
        with open("models/priori.file", "rb") as f:
            self.priori = pickle.load(f)
        with open("models/ending_prob.file", "rb") as f:
            self.ending_prob = pickle.load(f)
        with open("models/starting_prob.file", "rb") as f:
            self.starting_prob = pickle.load(f)
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
        print("current_likelihood" + str(current_likelihood))
        print("starting_prob" + str(self.starting_prob))

        viterbi[:, 0] = current_likelihood * self.starting_prob
        print("viterbi[:, 0]" + str(viterbi[:, 0]))

        for observation_i in range(1, T):
            current_likelihood = self.get_likelihood_vect(splitted_sent[observation_i])
            print("obs " + str(observation_i)+ " :"+str(splitted_sent[observation_i]))
            print(current_likelihood)

            for current_state in range(0, N - 1):
                vect = viterbi[:, observation_i - 1] * self.priori[:, current_state]  # Paths to the new position
                vector_prod = vect * current_likelihood
                viterbi[current_state, observation_i] = np.max(vector_prod)
                viterbi_backpointer[current_state, observation_i] = np.argmax(vect)
                print("LIKELI")
                print(vect)
                print("Viterbi i -1")

        # Termination Step - prob last word ending sent with a tag
        viterbi[:, T - 1] = np.max(viterbi[:, T - 1] * self.ending_prob)
        last_index = np.argmax(viterbi[:, T - 1] * self.ending_prob)

        print("ending prob " + str(self.ending_prob))
        print(viterbi[:, T - 1])
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
        print("::: VITERBI PATH :::")
        print(viterbi_backpointer)
        print("Last indx: "+str(last_index))
        print("column: " + str(column))
        while column >= 0:
            result.append(self.get_tag_from_index(last_index))
            last_index = viterbi_backpointer[int(last_index), column]
            column = column - 1

        print("viterbi path: " + str(result))
        return result

    def update_tag_count(self, tag):
        index_count_tag = self.get_tag_index(tag)
        self.count_tags[index_count_tag] = self.count_tags[index_count_tag] + 1
        return

    def update_ending_prob(self, tag):
        tag_index = self.get_tag_index(tag)
        self.ending_prob[tag_index] = self.ending_prob[tag_index] + 1
        return

    def update_starting_prob(self, tag):
        tag_index = self.get_tag_index(tag)
        self.starting_prob[tag_index] = self.starting_prob[tag_index] + 1
        return

    def update_likelihood(self, tag, params=None):
        if params is None:
            params = np.zeros(17)
        tag_index = self.get_tag_index(tag)
        params[tag_index] = params[tag_index] + 1
        return params

    def update_prior(self, tag, previous_tag):
        current_tag = self.get_tag_index(tag)
        previous_tag = self.get_tag_index(previous_tag)
        self.priori[previous_tag, current_tag] = self.priori[previous_tag, current_tag] + 1
        return

    def get_likelihood_vect(self, word):
        for hmm_item in self.hmm_dict:
            # retrieve probabilities for current word
            if hmm_item["lemma"] == word:
                current_likelihood = hmm_item.get("likelihood")
                return current_likelihood
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

    def normalize_ending_prob(self):
        for i in range(0, 16):
            self.ending_prob[i] = self.ending_prob[i] / self.count_tags[i]
        return

    def normalize_starting_prob(self):
        for i in range(0, 16):
            self.starting_prob[i] = self.starting_prob[i] / self.count_tags[i]
        return

    def get_tag_index(self, tag):
        for t in self.tag_list:
            if tag in t:
                return t[1]

    def get_tag_from_index(self, index):
        for t in self.tag_list:
            if index in t:
                return t[0]
