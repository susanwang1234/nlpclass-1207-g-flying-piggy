import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm.notebook import tqdm
from scipy.sparse import vstack
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MaxAbsScaler
import optparse

class myLR:
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

    def apply_svd(self, global_tokens):
        svd = TruncatedSVD(n_components=100, n_iter=10)
        svd.fit(global_tokens)
        global_svd = svd.transform(global_tokens)
        return global_svd
    
    def apply_scale(self, global_svd):
        scaler = MaxAbsScaler()
        scaler.fit(global_svd)
        global_scaled = scaler.transform(global_svd)
        train_scaled = global_scaled[:len(self.train)]
        test_scaled = global_scaled[len(self.train):]
        return train_scaled, test_scaled
    
    def fit_predict(self, train, test):
        predict_on_test = np.zeros((len(self.test), len(self.class_labels)))
        predict_on_train = np.zeros((len(self.train), len(self.class_labels)))
        for idx, label in tqdm(enumerate(self.class_labels)):
            print("Training %s"%(label))
            m = LogisticRegression().fit(train, self.train[label])
            predict_on_test[:,idx] = m.predict_proba(test)[:,1]
            predict_on_train[:,idx] = m.predict_proba(train)[:,1]
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
    (opts, _) = optparser.parse_args()

    class_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    model = myLR(opts.train_file, opts.test_file, class_labels)

    global_tokens = model.generate_input_vectors()
    global_svd = model.apply_svd(global_tokens)
    train_scaled, test_scaled = model.apply_scale(global_svd)

    print("Training a balanced Logistic Regression model...")
    _, predict = model.fit_predict(train=train_scaled, test=test_scaled)
    filename = "../output/submission_logistic.csv"
    model.output(predict, filename)        

    print("Done! Please check", filename)