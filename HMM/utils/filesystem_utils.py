import pickle


def save_model(word_list, likelihood_list, count_tags, priori, ending_prob, starting_prob):
    with open("models/word_list.file", "wb") as f:
        pickle.dump(word_list, f, pickle.HIGHEST_PROTOCOL)
    with open("models/likelihood_list.file", "wb") as f:
        pickle.dump(likelihood_list, f, pickle.HIGHEST_PROTOCOL)
    with open("models/count_tags.file", "wb") as f:
        pickle.dump(count_tags, f, pickle.HIGHEST_PROTOCOL)
    with open("models/priori.file", "wb") as f:
        pickle.dump(priori, f, pickle.HIGHEST_PROTOCOL)
    with open("models/ending_prob.file", "wb") as f:
        pickle.dump(ending_prob, f, pickle.HIGHEST_PROTOCOL)
    with open("models/starting_prob.file", "wb") as f:
        pickle.dump(starting_prob, f, pickle.HIGHEST_PROTOCOL)


def load_model():
    with open("models/word_list.file", "rb") as f:
        word_list = pickle.load(f)
    with open("models/likelihood_list.file", "rb") as f:
        likelihood_list = pickle.load(f)
    with open("models/count_tags.file", "rb") as f:
        count_tags = pickle.load(f)
    with open("models/priori.file", "rb") as f:
        priori = pickle.load(f)
    with open("models/ending_prob.file", "rb") as f:
        ending_prob = pickle.load(f)
    with open("models/starting_prob.file", "rb") as f:
        starting_prob = pickle.load(f)
    return word_list, likelihood_list, count_tags, priori, ending_prob, starting_prob