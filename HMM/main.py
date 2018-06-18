from HMM.utils import conllu_utils, test_hmm
from HMM.hmm import HiddenMarkovModel

# Get training data
train_data = conllu_utils.get_train_data()

# Train HMM
hmm = HiddenMarkovModel()
hmm.train_hmm(train_data)

#
# tagged = hmm.tag("ciao .".replace(",", "").split(" "))
# tagged2 = hmm.tag("se tale decisione essere prendere anteriormente a il data".replace(",", "").split(" "))
# tagged3 = hmm.tag("portare il mio cane".replace(",", "").split(" "))
# print(tagged)
# print(tagged2)
# print(tagged3)

# Get Test Data
test_data = conllu_utils.get_test_data()

hmm_output = []
for test_sent in test_data:
    hmm_output.append(hmm.tag(test_sent))


tagged_test_data = conllu_utils.get_test_data(True)

accuracy = test_hmm.get_accuracy(hmm_output, tagged_test_data)
print("accuracy: " + str(accuracy))