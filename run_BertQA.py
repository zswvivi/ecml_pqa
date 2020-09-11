""" Create, train and test BertQA"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import ast
import copy
import collections
import numpy as np
import pandas as pd
from datetime import datetime

import modeling
import optimization
import tokenization
import run_classifier

import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

flags = tf.flags
FLAGS = flags.FLAGS


## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .txt files"
)

flags.DEFINE_string(
    "category_name", None,
    "Which category should be processed, e.g. Patio_Lawn_and_Garden"
    "For cross domain pretraining, which is to train model on all categories, use All_Categories"
)


flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture."
)

flags.DEFINE_string(
    "vocab_file", None,
    "The vocabulary file that the BERT model was trained on."
)

flags.DEFINE_string(
    "model_output_dir", None,
    "The output directory where the model checkpoints will be written."
)

flags.DEFINE_integer(
    "max_seq_length", 40,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded."
)

flags.DEFINE_integer(
    "num_top_reviews", 10,
    "The number of top N reviews (ranked by FLTR) are used in training BertQA."
)

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set."
)

flags.DEFINE_integer(
    "train_batch_size", 180, 
    "Total batch size for training, has to be 30*n, n is the number of questions per batch."
    "e.g. 180 equals 6 questions, each question has 30 paris ( 30*6=180 )" 
    "30 pairs include 10 Question/Reviews, 10 answer/reviews and 10 non-answer/reviews."
)


flags.DEFINE_float(
    "learning_rate", 5e-6, 
    "The initial learning rate for Adam."
)

flags.DEFINE_float(
    "num_train_epochs", 6.0,
    "Total number of training epochs to perform."
)

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training."
)

flags.DEFINE_integer(
    "save_checkpoints_steps", 1000,
     "How often to save the model checkpoint."
)

flags.DEFINE_integer(
    "save_summary_steps", 2000,
     "How often to save the summary"
)


def input_fn_builder(features, seq_length, is_training, drop_remainder):
    
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
            })
        
        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        d = d.prefetch(batch_size)
        if is_training:
            d = d.repeat()
        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """ Creates a BertQA. """
    
    batch_size = FLAGS.train_batch_size
    k = FLAGS.num_top_reviews*3
    
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value
    output_weights = tf.get_variable("BertQA_output_weights", 
                                     [1, hidden_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable("BertQA_output_bias", 
                                  [1], 
                                  initializer=tf.zeros_initializer())
    
    if is_training:
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.8)
        
    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    logits = tf.reshape(logits,[int(batch_size/k),k])
    #logits = tf.Print(logits,[logits,tf.shape(logits)],'logits:')
    
    def scoring_fn(temp):
        rq = temp[0:FLAGS.num_top_reviews]
        rq = tf.reshape(rq,[FLAGS.num_top_reviews])
        #rq = tf.nn.softmax(rq, axis=-1)
        #rq = tf.Print(rq,[rq],"Score of each review to question:")
        rq = tf.sigmoid(rq)

        ar = temp[FLAGS.num_top_reviews:FLAGS.num_top_reviews*2]
        ar = tf.reshape(ar,[FLAGS.num_top_reviews])
        ar = tf.sigmoid(ar)
        #ar = tf.Print(ar,[ar],"Score of each review to true answer:")
    
        nar = temp[FLAGS.num_top_reviews*2:FLAGS.num_top_reviews*3]
        nar = tf.reshape(nar,[FLAGS.num_top_reviews]) 
        nar = tf.sigmoid(nar)
        #nar = tf.Print(nar,[nar],"Score of each review to non-answer:")
    
        ars = tf.tensordot(rq, ar, 1)
        nars = tf.tensordot(rq, nar, 1)
    
        #ars = tf.Print(ars,[ars,tf.shape(ars)],'Score of ture answer: ')
        #nars = tf.Print(nars,[nars,tf.shape(nars)],'Score of non-answer: ')
        
        score = tf.subtract(ars,nars)
        #score = tf.Print(score,[score],'Score: ')
        return score,rq
    
    scores,rqs = tf.map_fn(scoring_fn,logits,dtype=(tf.float32, tf.float32))
    #scores = tf.Print(scores,[scores,tf.shape(scores)],'scores: ')
    
    with tf.variable_scope("loss"):    
        #rqs = tf.Print(rqs,[rqs],"p(review | question):")
        predicted_labels = tf.squeeze(tf.round(tf.to_float(scores)+0.5))         
        delta = 0.5       
        per_example_loss = tf.maximum(0.0, delta - scores)
        #per_example_loss = tf.Print(per_example_loss,[per_example_loss],'per_example_loss: ')
        loss = tf.reduce_mean(per_example_loss)
        loss = tf.Print(loss,[loss],'loss: ')
        #if is_predicting:
        #      return (predicted_labels, scores,rqs)
        return (loss, predicted_labels, scores,rqs)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,
                     use_one_hot_embeddings=False):
    
    def model_fn(features, labels, mode, params): 
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
 
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (loss, predicted_labels, scores, rqs) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
            ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        print("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
                print("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

        output_spec = None
        
        if is_training:
            train_op = optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
            
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op)
        else:
            predictions = {
                          'scores': scores
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        
    return model_fn


def FLTR_Top10(t):
    reviews = copy.deepcopy(t['reviewText'])
    FLTR_scores = copy.deepcopy(t['FLTR_scores'])
    temp=[]
    for i in range(0,FLAGS.num_top_reviews):
        if i<len(reviews):
            index = FLTR_scores.index(max(FLTR_scores))
            temp.append(reviews[index])
            reviews.pop(index)
            FLTR_scores.pop(index)
        else:
            temp.append('[PAD]')
    return temp

def qar_pair(t):
    x1=[]
    x2=[]
    x3=[]
    for i in range(FLAGS.num_top_reviews):
        x1.append([t['question'],t['FLTR_Top10'][i]])
        x2.append([t['answer'],t['FLTR_Top10'][i]])    
        x3.append([t['non_answer'],t['FLTR_Top10'][i]])
    return x1+x2+x3

def main(_):
    """For current work, we only use the following four categories, 
    but you can add others if you would like to."""
    categories = [
              'Tools_and_Home_Improvement', 
              'Patio_Lawn_and_Garden',
              'Electronics',
              'Baby',
    ]
    
    data = None
    
    """Cross-domain pre-training (All_Categories) to boost the performance."""
    if FLAGS.category_name == "All_Categories":
        for category in categories:
            category_data_path = os.path.join(FLAGS.data_dir,category+'.txt')
            category_data = pd.read_csv(category_data_path,sep='\t',encoding='utf-8',nrows=10000,
                  converters={'reviewText':ast.literal_eval,'FLTR_scores':ast.literal_eval})
            if data is None:
                data = category_data
            else:
                data = pd.concat([data,category_data],axis=0)
        data = data.sample(n=len(data))
    else:
        data_path = os.path.join(FLAGS.data_dir,FLAGS.category_name+'.txt')
        data = pd.read_csv(data_path,sep='\t',encoding='utf-8',#nrows=10000,
                  cconverters={'reviewText':ast.literal_eval,'FLTR_scores':ast.literal_eval})
    
    #data['len_questions'] = data["question"].apply(lambda x: len(x.split()))
    #data = data[data['len_questions']<=10]
    
    data['FLTR_Top10'] = data.apply(FLTR_Top10,axis=1)
    list_of_answers = list(data['answer'])
    list_of_answers=shuffle(list_of_answers)
    data['non_answer']= list_of_answers
    
    train = data[:int(len(data)*0.8)]
    train = train.sample(n=min(20000,len(train)))
    test = data[int(len(data)*0.8):]
    print(train.shape,test.shape)

    DATA_COLUMN_A = 'senA'
    DATA_COLUMN_B = 'senB'
    LABEL_COLUMN = 'Label'
    label_list = [0, 1]
         
    train = train.apply(qar_pair,axis=1)    
    test = test.apply(qar_pair,axis=1)
    
    temp = train.tolist()
    flat_list = [item for sublist in temp for item in sublist]
    train =pd.DataFrame(flat_list,columns=['senA','senB'])
    train['Label'] =1
    train['senA']=train['senA'].apply(str)
    train['senB']=train['senB'].apply(str)
    
    temp = test.tolist()
    flat_list = [item for sublist in temp for item in sublist]
    test = pd.DataFrame(flat_list,columns=['senA','senB'])
    test['Label'] = 1
    test['senA'] = test['senA'].apply(str)
    test['senB'] = test['senB'].apply(str)
    print(train.shape,test.shape)
    
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, 
                                           do_lower_case=True)
    
    train_InputExamples = train.apply(lambda x: run_classifier.InputExample(guid=None, 
                                                                        text_a = x[DATA_COLUMN_A],
                                                                        text_b = x[DATA_COLUMN_B],
                                                                        label = x[LABEL_COLUMN]), 
                                       axis = 1)

    test_InputExamples = test.apply(lambda x: run_classifier.InputExample(guid=None,
                                                                           text_a = x[DATA_COLUMN_A],
                                                                           text_b = x[DATA_COLUMN_B],
                                                                           label = x[LABEL_COLUMN]), 
                                     axis = 1)
                                            
    train_features = run_classifier.convert_examples_to_features(train_InputExamples, 
                                                  label_list, 
                                                  FLAGS.max_seq_length, 
                                                  tokenizer)
    test_features = run_classifier.convert_examples_to_features(test_InputExamples, 
                                                 label_list, 
                                                 FLAGS.max_seq_length, 
                                                 tokenizer)

    
    OUTPUT_DIR = os.path.join(FLAGS.model_output_dir,FLAGS.category_name+"_BertQA")
    tf.gfile.MakeDirs(OUTPUT_DIR)
    
    run_config = tf.estimator.RunConfig(
                                    model_dir=OUTPUT_DIR,
                                    keep_checkpoint_max=2,
                                    save_summary_steps=FLAGS.save_summary_steps,
                                    save_checkpoints_steps=FLAGS.save_checkpoints_steps)

    num_train_steps = int(len(train_features) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    
    
    model_fn = model_fn_builder(
                            bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file),
                            num_labels = len(label_list),
                            init_checkpoint = FLAGS.init_checkpoint,
                            learning_rate = FLAGS.learning_rate,
                            num_train_steps = num_train_steps,
                            num_warmup_steps = num_warmup_steps)

    estimator = tf.estimator.Estimator(
                                   model_fn = model_fn,
                                   config = run_config,
                                   params = {"batch_size": FLAGS.train_batch_size})

    train_input_fn = run_classifier.input_fn_builder(
                                                 features = train_features,
                                                 seq_length = FLAGS.max_seq_length,
                                                 is_training = True,
                                                 drop_remainder = True)

    print("Beginning Training!")
    current_time = datetime.now()
    #early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(
    #                 estimator,metric_name='loss',max_steps_without_decrease=1000,min_steps=100)

    estimator.train(input_fn = train_input_fn, max_steps = num_train_steps) #,hooks=[early_stopping]
    print("Training took time ", datetime.now() - current_time)
    
    test_input_fn = run_classifier.input_fn_builder(
                                                features = test_features,
                                                seq_length = FLAGS.max_seq_length,
                                                is_training = False,
                                                drop_remainder = True)
                                                

    predictions = estimator.predict(test_input_fn)
    x=[prediction['scores'] for prediction in predictions]
    print('\n')
    print("The accuracy of BertQA on "+FLAGS.category_name+" is: "+str(sum(i > 0 for i in x)/len(x)))
    print('\n')

if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("category_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("model_output_dir")
    tf.app.run()
