"""
First run `python3 zipout.py` to create your output zipfile `output.zip` and output directory `./output`

Then run:

    python3 check.py

It will print out a score of all your outputs that matched the
testcases with a reference output file (typically `./references/dev/*.out`).
In some cases the output is supposed to fail in which case it
compares the `*.ret` files instead.

To customize the files used by default, run:

    python3 check.py -h
"""

import sys, os, optparse, subprocess
import kaggle
import time


class Check:

    def __init__(self, out_dir, competition):
        self.out_dir = out_dir
        self.competition = competition
        self.suffix = ["svc_imbalanced", "svc_balanced", "logistic", "CNN", "MLP", "LSTM"]
        
    def submit_file(self):
        for suf in self.suffix:
            filename = "submission_" + suf + ".csv"
            if os.path.exists(filename) and os.access(filename, os.X_OK):
                argv = ["kaggle", "competitions", "submit", "-c", self.competition, "-f", filename, "-m", "Evaluate " + suf]
            else:
                print("Did not find {}. Please create the output file by running source code bufore evaluation.".format(filename), file=sys.stderr)
                continue

            try:
                prog = subprocess.Popen(argv)
                prog.wait()
            except:
                print("ERROR: something went wrong when trying to run the following command: ", file=sys.stderr)
                print(argv, file=sys.stderr)
                raise

    def list_submissions(self, file):
        out_file = open(file, 'w')
        argv = ["kaggle", "competitions", "submissions", self.competition]
        try:
            prog = subprocess.Popen(argv, stdout=out_file)
            prog.wait()
        except:
            print("ERROR: something went wrong when trying to run the following command: ", file=sys.stderr)
            print(argv, file=sys.stderr)
            raise


if __name__ == '__main__':
    #check_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    optparser = optparse.OptionParser()
    optparser.add_option("-o", "--output", dest="out_dir", default=os.path.join('.'), help="output file directory")
    optparser.add_option("-c", "--competition", dest="competition", default='jigsaw-toxic-comment-classification-challenge', help="the name of the competition")
    optparser.add_option("-f", "--file", dest="scorefile", default=os.path.join('scores.txt'), help="the output file for scores of models")
    (opts, _) = optparser.parse_args()

    check = Check(out_dir=opts.out_dir, competition=opts.competition)
    check.submit_file()
    print()
    print("Loading the score list...")
    time.sleep(5)
    check.list_submissions(file=opts.scorefile)
    print("Press ENTER to continue...")
    
    

