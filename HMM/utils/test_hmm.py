def get_accuracy(hmm_output, correct_tagged_data):
    tp_tn = 0
    total = 0
    n_sent = len(hmm_output)

    for s in range(0, n_sent):
        print("::: hmm_output :::")
        print(hmm_output[s])
        print("::: correct_tagged_data :::")
        print(correct_tagged_data[s])
        n_word = len(hmm_output[s])
        for n in range(0, n_word):
            total = total + 1
            if hmm_output[s][n] == correct_tagged_data[s][n]:
                tp_tn = tp_tn + 1
        print("count tp+tn->" + str(tp_tn))
        print("count tot->" + str(total))

    return tp_tn / total
