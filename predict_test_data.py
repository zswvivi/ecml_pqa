"""Test FLTR on test data"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import os
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
    "train_batch_size", 180, 
    "Total batch size for training."
)


flags.DEFINE_float(
    "learning_rate", 5e-5, 
    "The initial learning rate for Adam."
)

flags.DEFINE_float(
    "num_train_epochs", 3.0,
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


def create_BertQA(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """ Creates a BertQA. """
    
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
    
    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    logits = tf.sigmoid(logits)
      
    predicted_labels = tf.squeeze(tf.round(tf.to_float(logits)+0.5))         

    return (predicted_labels, logits)


def model_fn_builder_BertQA(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,
                     use_one_hot_embeddings=False):
    
    def model_fn(features, labels, mode, params): 
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
 
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (predicted_labels, logits) = create_BertQA(
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
                          'probabilities': logits
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        
    return model_fn


def create_FLTR(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a FLTR."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

    
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        
        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder_FLTR(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,
                     use_one_hot_embeddings=False):
    
    def model_fn(features, labels, mode, params): 
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
 
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (loss, per_example_loss, logits, probabilities) = create_FLTR(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
            ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
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
                          'probabilities': probabilities
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        
    return model_fn



def main(_):
    """For current work, we only use the following four categories, 
    but you can add others if you would like to."""
    categories = [
              'Tools_and_Home_Improvement', 
              'Patio_Lawn_and_Garden',
              'Electronics',
              'Baby',
    ]
    
    models = ['FLTR','BertQA']
    
    data_path = os.path.join(FLAGS.data_dir,'Annotated_Data.txt')
    data = pd.read_csv(data_path,sep='\t',encoding='utf-8',
                       converters={'annotation_score':ast.literal_eval, 'reviews':ast.literal_eval})
    data = data.reset_index()
    data['qr'] = data[['index','question','reviews']].apply(lambda x: [[x['index'],x['question'],i] 
                                                                        for i in x['reviews']],axis=1)
    
    d = []
    for category in categories:
        qr = data[data['category']==category]['qr'].tolist()
        qr = [item for sublist in qr for item in sublist]
        qr = pd.DataFrame(columns=['index','question','review'],data=qr)
        qr['label'] = 1

        temp = qr.copy()
        temp['question'] = temp['question'].apply(str)
        temp['review'] = temp['review'].apply(str)
        DATA_COLUMN_A = 'question'
        DATA_COLUMN_B = 'review'
        LABEL_COLUMN = 'label'
        label_list=[0,1]
        
        tokenizer = tokenization.FullTokenizer(
                                           vocab_file=FLAGS.vocab_file, 
                                           do_lower_case=True)
   

        test_InputExamples = temp.apply(lambda x: run_classifier.InputExample(guid=None,
                                                                           text_a = x[DATA_COLUMN_A],
                                                                           text_b = x[DATA_COLUMN_B],
                                                                           label = x[LABEL_COLUMN]), 
                                     axis = 1)
                                            
        test_features = run_classifier.convert_examples_to_features(test_InputExamples, 
                                                 label_list, 
                                                 FLAGS.max_seq_length, 
                                                 tokenizer)

        t = data[data['category']==category]
        t = t.reset_index(drop=True)
        for model in models:
            
            OUTPUT_DIR = os.path.join(FLAGS.model_output_dir, category+'_'+model)
            run_config = tf.estimator.RunConfig(
                                                model_dir=OUTPUT_DIR,
                                                save_summary_steps=100,
                                                save_checkpoints_steps=100)

            
            model_fn = None
            if model == 'BertQA':
                model_fn = model_fn_builder_BertQA(
                                        bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file),
                                        num_labels=len(label_list),
                                        init_checkpoint=OUTPUT_DIR,
                                        learning_rate=FLAGS.learning_rate,
                                        num_train_steps=100,
                                        num_warmup_steps=100)
            else:
                model_fn = model_fn_builder_FLTR(
                                        bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file),
                                        num_labels=len(label_list),
                                        init_checkpoint=OUTPUT_DIR,
                                        learning_rate=FLAGS.learning_rate,
                                        num_train_steps=100,
                                        num_warmup_steps=100)
                

                
            estimator = tf.estimator.Estimator(
                                               model_fn=model_fn,
                                               config=run_config,
                                               params={"batch_size": FLAGS.train_batch_size})
    
    
            test_input_fn = run_classifier.input_fn_builder(
                                                features=test_features,
                                                seq_length=FLAGS.max_seq_length,
                                                is_training=False,
                                                drop_remainder=False)
                                                

            predictions = estimator.predict(test_input_fn)
            probabilities = [prediction['probabilities'] for prediction in  predictions]
            probabilities = [list(item) for item in probabilities]
            
            if model == 'FLTR':        
                probabilities = [item[1] for item in probabilities]
            else:
                probabilities = [item[0] for item in probabilities]
                
            print(model,' :',probabilities[:10])
            temp[model+'_score'] = probabilities
            temp_groupby = temp.groupby(['index','question'],
                                sort=False)[model+'_score'].apply(list).reset_index(name=model+'_score')
            t = pd.concat([t,temp_groupby[model+'_score']],axis=1)
    
        if len(d)==0:
              d = t
        else:
              d = pd.concat([d,t],axis=0,ignore_index=True)
                
    d.to_csv(os.path.join(FLAGS.data_dir, 'test_predictions.txt'), index=None, sep='\t', mode='w') 
                
if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("model_output_dir")
    tf.app.run()
