import os, sys, optparse
import tqdm
import pymagnitude
from copy import deepcopy
import re
import numpy
import math

class LexSub:

    def __init__(self, wvecs, topn=10):
        self.wvecs = pymagnitude.Magnitude(wvecs)
        self.topn = topn

    def substitutes(self, index, sentence):
        "Return ten guesses that are appropriate lexical substitutions for the word at sentence[index]."
        return(list(map(lambda k: k[0], self.wvecs.most_similar(sentence[index], topn=self.topn))))

def build_wvec_dict(file):
    wordVectors = {}
    for word, vec in file:
        wordVectors[word] = numpy.zeros(len(vec), dtype=float)
        for idx, elem in enumerate(vec):
            wordVectors[word][idx] = float(elem)
        wordVectors[word] /= math.sqrt((wordVectors[word]**2).sum() + 1e-6)
    return wordVectors

isNumber = re.compile(r"\d+.*")
def norm_word(word):
    # Handle numbers
    if isNumber.search(word.lower()):
        return "---num---"
    # Handle punctuation
    elif re.sub(r"\W+", "", word) == "":
        return "---punc---"
    # Handle words
    else:
        return word.lower()

def build_lexicon_dict(filename):
    lexicon = {}
    for line in open(filename, "r"):
        # Handle words
        words = line.lower().strip().split()
        lexicon[norm_word(words[0])]=[norm_word(word) for word in words[1:]]
    return lexicon

def retrofit_vector(wordVecs, lexicon, alpha, beta, iteration):
    # wordVecs is q_bar, newwordVecs is q
    newwordVecs = deepcopy(wordVecs)
    wvVocab = set(wordVecs.keys())

    loopVocab = wvVocab.intersection(set(lexicon.keys()))
    for i in range(iteration):
        for word in loopVocab:
            # Seek neighbours
            wordNeighbours = set(lexicon[word]).intersection(wvVocab)
            num = len(wordNeighbours)
            # If there is no neighbour found, skip it.
            if num == 0:
                continue
            # Numerator is equal to sum(beta * q) + alpha * q_bar
            numerator = alpha * wordVecs[word]
            for neighbour in wordNeighbours:
                numerator += beta * newwordVecs[neighbour]
            # Denominator is euqal to beta * n + alpha
            denominator = num * beta + alpha

            newwordVecs[word]= numerator / denominator
    return newwordVecs

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="input file with target word in context")
    optparser.add_option("-w", "--wordvecfile", dest="wordvecfile", default=os.path.join('data', 'glove.6B.100d.magnitude'), help="word vectors file")
    optparser.add_option("-n", "--topn", dest="topn", default=10, help="produce these many guesses")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    optparser.add_option("-a", "--lexiconfile", dest="lexiconfile",default=os.path.join("data","lexicons","ppdb-xl.txt"), help="lexicon file")
    optparser.add_option("-b", "--retrofittedfile", dest="retrofittedfile",default=os.path.join("data","glove.6B.100d.retrofit.magnitude"), help="retrofitted file")
    (opts, _) = optparser.parse_args()
    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    iteration = 15
    alpha = 1
    beta = 1
    if os.path.exists(opts.retrofittedfile) == False:
        # Build dictionaries
        lexicon_dict = build_lexicon_dict(opts.lexiconfile)
        wvec_dict = build_wvec_dict(pymagnitude.Magnitude(opts.wordvecfile))
        # Retrofit vector
        new_vector = retrofit_vector(wvec_dict, lexicon_dict, alpha, beta, iteration)
        # Write txt
        new_txt = os.path.join("data","glove.6B.100d.retrofit.txt")
        output = open(new_txt, "w")
        for word, vec in new_vector.items():
            output.write(word)
            for elem in vec:
                output.write(" " + "%.4f"%(elem))
            output.write("\n")
        output.close()
        # Convert txt to magnitude
        os.system("python3 -m pymagnitude.converter -i data/glove.6B.100d.retrofit.txt -o data/glove.6B.100d.retrofit.magnitude")


    lexsub = LexSub(opts.retrofittedfile, int(opts.topn))
    num_lines = sum(1 for line in open(opts.input,'r'))
    with open(opts.input) as f:
        for line in tqdm.tqdm(f, total=num_lines):
            fields = line.strip().split('\t')
            print(" ".join(lexsub.substitutes(int(fields[0].strip()), fields[1].strip().split())))