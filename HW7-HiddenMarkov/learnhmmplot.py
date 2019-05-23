import numpy as np
import sys

# train_input=sys.argv[1]
# index_to_word=sys.argv[2]
# index_to_tag=sys.argv[3]
# hmmprior=sys.argv[4]
# hmmemit=sys.argv[5]
# hmmtrans=sys.argv[6]

train_input = "trainwords.txt"
index_to_word = "index_to_word.txt"
index_to_tag = "index_to_tag.txt"
hmmprior = "hmmprior.txt"
hmmemit = "hmmemit.txt"
hmmtrans = "hmmtrans.txt"
sequence = 10000


## store data tag and word into dictionary
def getDictionary(filename):
    with open(filename, "r") as doc:
        i = 0  # index start from 0
        dictionary = {}
        for line in doc:
            line = line.rstrip()
            dictionary[line] = i
            i = i + 1
        return dictionary


## read train file as input
def read_train(filename):
    with open(filename, "r") as doc:
        file = []
        i = 1
        for line in doc:  # for each line
            if (i <= sequence):
                subline = []
                newline = line.rstrip()
                word = newline.split(" ")  # store each word in the line
                for word in word:  # for each word element
                    text = word.split("_")[0]
                    tag = word.split("_")[1]
                    element = [text, tag]
                    subline.append(element)
                file.append(subline)
                print(i)
            i = i + 1
        return file


def getPrior(trainfile, dictionary):
    prior = [0] * len(dictionary)
    for line in trainfile:
        first_word = line[0]
        tag = first_word[1]
        index = dictionary[tag]
        prior[index] = prior[index] + 1
    new_prior = [x + 1 for x in prior]
    return new_prior


def getHmmPrior(prior):
    SUM = sum(prior)
    hmm_prior = []
    for line in prior:
        hmm_prior.append(line / SUM)
    return hmm_prior


def getTrans(trainfile, dictionary):
    result = [[0] * len(dictionary)] * len(dictionary)
    result = np.array(result)
    for line in trainfile:
        length = len(line)
        for i in range(0, length - 1):
            front_word = line[i]
            front_tag = front_word[1]
            front_index = dictionary[front_tag]
            back_word = line[i + 1]
            back_tag = back_word[1]
            back_index = dictionary[back_tag]
            result[front_index][back_index] = result[front_index][back_index] + 1
    new_result = np.add(result, 1)
    return new_result


def getHmmTrans(trans_list):
    result = []
    for line in trans_list:
        SUM = sum(line)
        subline = []
        for element in line:
            subline.append(element / SUM)
        result.append(subline)
    return result


def getEmit(trainfile, tag_dic, word_dic):
    result = [[0] * len(word_dic)] * len(tag_dic)
    result = np.array(result)
    for line in trainfile:
        for word in line:
            text = word[0]
            tag = word[1]
            text_index = word_dic[text]
            tag_index = tag_dic[tag]
            result[tag_index][text_index] = result[tag_index][text_index] + 1
    new_result = np.add(result, 1)
    return new_result


def getHmmEmit(emit_list):
    result = []
    for line in emit_list:
        SUM = sum(line)
        subline = []
        for element in line:
            subline.append(element / SUM)
        result.append(subline)
    return result


# export files into txt
def export(listname, exportname):
    file = open(exportname, "w")
    for line in listname:
        content = ""
        for i in range(0, len(line)):
            if i < len(line) - 1:
                content = content + str(line[i]) + " "
            else:
                content = content + str(line[i])
        file.write(str(content) + '\n')
        # print(content)
    file.close()


def export2(listname, exportname):
    file = open(exportname, "w")
    for line in listname:
        file.write(str(line) + '\n')
        # print(line)
    file.close()


tag_dic = getDictionary(index_to_tag)
word_dic = getDictionary(index_to_word)
train_file = read_train(train_input)

# get the prior
prior_list = getPrior(train_file, tag_dic)
Hmm_prior = getHmmPrior(prior_list)

# get the trans
trans_list = getTrans(train_file, tag_dic)
Hmm_trans = getHmmTrans(trans_list)

# get the emit
emit_list = getEmit(train_file, tag_dic, word_dic)
Hmm_emit = getHmmEmit(emit_list)

# export 3 files
export2(Hmm_prior, hmmprior)
export(Hmm_trans, hmmtrans)
export(Hmm_emit, hmmemit)
