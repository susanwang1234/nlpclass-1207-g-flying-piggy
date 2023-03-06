import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import svm
from tqdm.notebook import tqdm
from scipy.sparse import vstack
from sklearn.decomposition import TruncatedSVD
import optparse

class mySVC:
    def __init__(self, train_file, test_file, class_labels):
        self.train = pd.read_csv(train_file)
        self.test = pd.read_csv(test_file)
        self.class_labels = class_labels
    
    def generate_input_vectors(self):
        vec = TfidfVectorizer()
        train_tokens = vec.fit_transform(self.train['comment_text'])
        test_tokens = vec.transform(self.test['comment_text'])
        global_tokens = vstack([train_tokens, test_tokens])

        return global_tokens

    def preprocess(self, global_tokens):
        svd = TruncatedSVD(n_components=20)
        svd.fit(global_tokens)
        global_svd = svd.transform(global_tokens)
        train_svd = global_svd[:len(self.train)]
        test_svd = global_svd[len(self.train):]

        return train_svd, test_svd
    
    def fit_predict(self, train_svd, test_svd, class_weight=None):
        predict_on_test = np.zeros((len(self.test), len(self.class_labels)))
        predict_on_train = np.zeros((len(self.train), len(self.class_labels)))
        for idx, label in tqdm(enumerate(self.class_labels)):
            print("Training %s"%(label))
            m = svm.SVC(probability=True, class_weight=class_weight).fit(train_svd, self.train[label])
            predict_on_test[:,idx] = m.predict_proba(test_svd)[:,1]
            predict_on_train[:,idx] = m.predict_proba(train_svd)[:,1]
        return predict_on_train, predict_on_test

    def output(self, predict, file_name):
        out = pd.concat([self.test["id"], pd.DataFrame(predict)], axis=1, ignore_index = True)
        output_columns = ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        out.columns = output_columns
        submission_file = open(file_name, "w")
        out.to_csv(file_name, index=False)
        submission_file.close()
    
if __name__ == '__main__':
    #check_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--train", dest="train_file", default=os.path.join('..', 'data', 'train.csv'), help="path to train.csv")
    optparser.add_option("-e", "--test", dest="test_file", default=os.path.join('..', 'data', 'test.csv'), help="path to test.csv")
    optparser.add_option("-b", "--balanced", dest="balanced", default=False, help="balance the weight or not")
    (opts, _) = optparser.parse_args()

    class_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    model = mySVC(opts.train_file, opts.test_file, class_labels)

    global_tokens = model.generate_input_vectors()
    train_svd, test_svd = model.preprocess(global_tokens)
    filename = None
    if opts.balanced:
        print("Training a balanced SVC model. It may take a long time...")
        _, predict = model.fit_predict(train_svd=train_svd, test_svd=test_svd, class_weight='balanced')
        filename = "../output/submission_svc_balanced.csv"
        model.output(predict, filename)        
    else:
        print("Training an imbalanced SVC model. It may take a long time...")
        _, predict = model.fit_predict(train_svd=train_svd, test_svd=test_svd, class_weight=None)
        filename = "../output/submission_svc_imbalanced.csv"
        model.output(predict, filename)

    print("Done! Please check", filename)