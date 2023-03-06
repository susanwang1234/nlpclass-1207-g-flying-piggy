import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math
from functools import reduce
from collections import defaultdict
from math import log10
import heapq




class Entry:
    def __init__(self, word, start_pos, log_prob, back_ptr):
        self.word = word
        self.start_pos = start_pos
        self.log_prob = log_prob
        self.back_ptr = back_ptr


def sortbyprob(entry):
    "Function for sorting by probability"
    return entry.log_prob


class Segment:

    def __init__(self, Pu, Pb, lam):
        self.Pu = Pu
        self.Pb = Pb
        self.lam = lam   # lambda for Jelinek-Mercer Smoothing

    def segment(self, text):
        "Return a list of words that is the best segmentation of text."
        if not text: return []

        # initialize the chart
        chart = [ None for i in range(len(text))]
        heap = []
        maxlen = 10 # max length of words (avoid too long words)
        

        # initialize the heap
        # push words start from position 0
        for i in range(len(text)):
            if i == maxlen:
                break
            word = text[: (i + 1)]
            if word in self.Pu.keys() or len(word) <= 3:
                entry = Entry(word, 0, log10(self.Pu(word)), None)
                heap.append(entry)

        # iteratively fill in chart[i] for i
        while heap:
            # sort in probability descending order and pop from heap
            heap.sort(key=sortbyprob, reverse=True)
            entry = heap.pop(0)
            endindex = entry.start_pos + len(entry.word) - 1
            # compute argmax
            if chart[endindex] is not None:
                preventry = chart[endindex]
                if entry.log_prob > preventry.log_prob:
                    chart[endindex] = entry
                else:
                    continue 
            else:
                chart[endindex] = entry
            # push new entry to the heap (starting from (endindex + 1))
            for i in range((endindex + 1), len(text)):
                if i - endindex - 1 == maxlen:
                    break
                newword = text[(endindex + 1): (i + 1)]
                bigram = entry.word + " " + newword
                if newword in self.Pu.keys() or len(newword) <= 3:
                    # compute conditional probability (cpr)
                    if bigram in self.Pb.keys() and entry.word in self.Pu.keys():
                        cpr = self.Pb[bigram] / self.Pu[entry.word]
                    else:
                        cpr = self.Pu(newword)
                    # apply Jelinek-Mercer Smoothing
                    newentry = Entry(newword, endindex + 1, entry.log_prob + log10(self.lam * cpr + (1 - self.lam) * self.Pu(newword)), entry)
                    if newentry not in heap:
                        heap.append(newentry)

        # get the best segmentation
        segmentation = []
        finalindex = len(text) - 1
        finalentry = chart[finalindex]
        while finalentry != None:
            segmentation.append(finalentry.word)
            finalentry = finalentry.back_ptr


        segmentation.reverse()

        return segmentation

    def Pwords(self, words): 
        "The Naive Bayes probability of a sequence of words."
        return product(self.Pw(w) for w in words)

#### Support functions (p. 224)

def product(nums):
    "Return the product of a sequence of numbers."
    return reduce(operator.mul, nums, 1)

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."
    def __init__(self, data=[], N=None, missingfn=None):
        for key,count in data:
            self[key] = self.get(key, 0) + int(count)
        self.N = float(N or sum(self.values()))
        self.missingfn = missingfn or (lambda k, N: 1./N)
    def __call__(self, key): 
        if key in self: return self[key]/self.N  
        else: return self.missingfn(key, self.N)

def datafile(name, sep='\t'):
    "Read key,value pairs from file."
    with open(name) as fh:
        for line in fh:
            (key, value) = line.split(sep)
            yield (key, value)


def avoid_long_words(key, N):
    log_prob = log10(10.) - log10(N * 10000 ** len(key)) #equivalent to log(10./(N * 10000 ** len(key))) , used to avoid arithmentic underflow for small probabilities
    return 10**log_prob #convert log back to its probability



if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts [default: data/count_1w.txt]")
    optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts [default: data/count_2w.txt]")
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    Pu = Pdist(data=datafile(opts.counts1w), missingfn=avoid_long_words)
    Pb = Pdist(data=datafile(opts.counts2w), missingfn=lambda x, y: 0)
    lam = 0.2425
    segmenter = Segment(Pu, Pb, lam)
    with open(opts.input) as f:
        for line in f:
            print(" ".join(segmenter.segment(line.strip())))
