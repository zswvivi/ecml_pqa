""" Preprocessing Amazon QA and Review data"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gzip
import json
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from nltk.tokenize import sent_tokenize
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", None,
    "The data directory where all the data kept.")


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def generate_QA_pairs(t):
    temp=[]
    for i in range(len(t['questions'])):
        temp.append([t['asin'],t['questions'][i]])   
    return temp

def tokenizer_sentence(t):
    sentences = sent_tokenize(t)
    sentences = sorted(sentences, key=len)
    sentences = list(dict.fromkeys(sentences))
    return sentences
 
def main(_):
    categories = [
              'Tools_and_Home_Improvement', 
              'Patio_Lawn_and_Garden',
              'Electronics',
              'Baby',
    ]
    
    for category in categories:
        reviews=os.path.join(FLAGS.data_dir,'reviews_'+category+'.json.gz') # review data
        qa=os.path.join(FLAGS.data_dir,'QA_'+category+'.json.gz') # QA data
        
        if os.path.exists(os.path.join(FLAGS.data_dir, category+'.txt')):
            continue
            
        da = getDF(qa)
        dr = getDF(reviews)

        # Generate qa pairs
        qa_pairs = da.apply(generate_QA_pairs,axis=1)
        qa_pairs = [item for sublist in qa_pairs for item in sublist]
        qa_pairs =pd.DataFrame(qa_pairs,columns=['asin','QA'])
        asins=da.asin.unique()

        # Only keep reviews whose product has QA pairs
        dr=dr[dr.asin.isin(asins)]
        dr=dr.reset_index(drop=True)
        dr = dr[['asin','reviewText']]
    
        # split review into sentences for each product
        review_agg=dr.groupby('asin')['reviewText'].apply(lambda x: "%s" % ' '.join(x)).reset_index()
        review_agg['reviewText']=review_agg['reviewText'].apply(tokenizer_sentence)
    
        qa_withReivews = pd.merge(qa_pairs, review_agg, how='inner', on=['asin'])
        qa_withReivews = shuffle(qa_withReivews,random_state=30)
        qa_withReivews = qa_withReivews.reset_index(drop=True)
        if(qa_withReivews.isnull().values.any()):
            qa_withReivews = qa_withReivews.replace(np.nan, '[PAD]', regex=True)
        qa_withReivews.to_csv(os.path.join(FLAGS.data_dir, category+'.txt'), index=None, sep='\t', mode='w')
        
if __name__ == '__main__':
    flags.mark_flag_as_required('data_dir')
    tf.app.run(main)   