""" Create, train and test FLTR"""

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


flags.DEFINE_bool(
    "do_predict", False,
    "Whether to predict FLTR scores for each review."
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
    "num_train_epochs", 8.0,
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


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
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


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,
                     use_one_hot_embeddings=False):
    
    def model_fn(features, labels, mode, params): 
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
 
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (loss, per_example_loss, logits, probabilities) = create_model(
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

def QR_pair(t):
    """ Generating Question and Review pairs"""
    temp=[]
    num_reviews = t['num_reviews']
    for i in range(num_reviews):
        temp.append([t['question'],t['reviewText'][i]])
    return temp

def FLTR_Prediction(d,tokenizer,estimator):
    """ Ranking all of reviews accoring to how they are relevant to a given quetion."""
    
    d = d.reset_index(drop=True)
    test_t = d.apply(QR_pair,axis=1)
    test_t = test_t.tolist()
    flat_list = [item for sublist in test_t for item in sublist]
    test_t = pd.DataFrame(flat_list,columns=['question','review'])
    test_t['question'] = test_t['question'].apply(str)
    test_t['review'] = test_t['review'].apply(str)
    DATA_COLUMN_A = 'question'
    DATA_COLUMN_B = 'review'
    label_list = [0, 1]
    max_inputs = 1000000
    probs = []
    temp_test = test_t.copy()
    
    while len(temp_test)>0:
        line = min(max_inputs,len(temp_test))
        temp = temp_test[:line]
        
        inputExamples = temp.apply(lambda x: run_classifier.InputExample(guid=None,
                                                                         text_a = x[DATA_COLUMN_A],
                                                                         text_b = x[DATA_COLUMN_B],
                                                                         label = 0), 
                                   axis = 1)
                                    
        input_features = run_classifier.convert_examples_to_features(inputExamples, 
                                                                     label_list, 
                                                                      FLAGS.max_seq_length, 
                                                                      tokenizer)  
        
        predict_input_fn = run_classifier.input_fn_builder(features=input_features, 
                                                           seq_length=FLAGS.max_seq_length,
                                                           is_training=False, 
                                                           drop_remainder=False)
        
        predictions = estimator.predict(predict_input_fn)
        probabilities = [prediction['probabilities'] for prediction in  predictions]
        probs = probs+[item.tolist()[1] for item in probabilities]
        
        if len(temp_test)>max_inputs:
            temp_test = temp_test[line:]
            temp_test = temp_test.reset_index(drop=True)
        else:
            temp_test = []

    test_t['probilities']=probs
    num_reviews = d['num_reviews'].tolist()
    d['FLTR_scores'] = ''
    for i in range(0,len(d)):
        n = num_reviews[i]
        #print(probs[:n])
        d.at[i,'FLTR_scores'] = probs[:n]
        #print(d.at[i,'FLTR_scores'])
        if i!=len(d)-1:
            probs=probs[n:]
            
    return d


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
    
    if FLAGS.category_name == "All_Categories":
        """Cross-domain pre-training (All_Categories)."""
        for category in categories:
            category_data_path = os.path.join(FLAGS.data_dir,category+'.txt')
            category_data = pd.read_csv(category_data_path,sep='\t',encoding='utf-8',#nrows=5000,
                  converters={'QA':ast.literal_eval,'reviewText':ast.literal_eval})
            
            category_data = category_data[:int(len(data)*0.8)]
            if data is None:
                data = category_data
            else:
                data = pd.concat([data,category_data],axis=0)
        data = data.sample(n=min(len(data),60000))
    else:
        data_path = os.path.join(FLAGS.data_dir,FLAGS.category_name+'.txt')
        data = pd.read_csv(data_path,sep='\t',encoding='utf-8',#nrows=10000,
                  converters={'QA':ast.literal_eval,'reviewText':ast.literal_eval})
    
    data['question'] = data['QA'].apply(lambda x: x['questionText'])
    data['answer'] = data['QA'].apply(lambda x: x['answers'][0]['answerText'] if len(x['answers'])>0 else PAD_WORD)
    data['num_reviews']= data['reviewText'].apply(lambda x: len(x))

    train = data[:int(len(data)*0.8)]
    
    list_of_answers = list(train['answer'])
    list_of_answers=shuffle(list_of_answers)
    qa = train[['question','answer']]
    nqa =  pd.DataFrame({'question': train['question'].tolist(),'answer':list_of_answers})
    qa['label']=1
    nqa['label']=0

    d = pd.concat([qa,nqa],axis=0)
    d=shuffle(d)
    d['question']=d['question'].apply(str)
    d['answer']=d['answer'].apply(str)
    split = int(len(d)*0.9)
    dtrain = d[0:split]
    dtest = d[split:]

    DATA_COLUMN_A = 'question'
    DATA_COLUMN_B = 'answer'
    LABEL_COLUMN = 'label'
    label_list = [0, 1]

    tokenizer = tokenization.FullTokenizer(
                                           vocab_file=FLAGS.vocab_file, 
                                           do_lower_case=True)
    
    train_InputExamples = dtrain.apply(lambda x: run_classifier.InputExample(guid=None, 
                                                                        text_a = x[DATA_COLUMN_A],
                                                                        text_b = x[DATA_COLUMN_B],
                                                                        label = x[LABEL_COLUMN]), 
                                       axis = 1)

    test_InputExamples = dtest.apply(lambda x: run_classifier.InputExample(guid=None,
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
   

    OUTPUT_DIR = os.path.join(FLAGS.model_output_dir,FLAGS.category_name+'_FLTR')
    tf.gfile.MakeDirs(OUTPUT_DIR)
    
    run_config = tf.estimator.RunConfig(
                                    model_dir=OUTPUT_DIR,
                                    save_summary_steps=FLAGS.save_summary_steps,
                                    save_checkpoints_steps=FLAGS.save_checkpoints_steps)

    num_train_steps = int(len(train_features) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    
    
    model_fn = model_fn_builder(
                            bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file),
                            num_labels=len(label_list),
                            init_checkpoint=FLAGS.init_checkpoint,
                            learning_rate=FLAGS.learning_rate,
                            num_train_steps=num_train_steps,
                            num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(
                                   model_fn=model_fn,
                                   config=run_config,
                                   params={"batch_size": FLAGS.train_batch_size})

    train_input_fn = run_classifier.input_fn_builder(
                                                 features=train_features,
                                                 seq_length=FLAGS.max_seq_length,
                                                 is_training=True,
                                                 drop_remainder=True)

    print("Beginning Training!")
    current_time = datetime.now()
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print("Training took time ", datetime.now() - current_time)
    
    test_input_fn = run_classifier.input_fn_builder(
                                                features=test_features,
                                                seq_length=FLAGS.max_seq_length,
                                                is_training=False,
                                                drop_remainder=False)
                                                

    predictions = estimator.predict(test_input_fn)
    x=[np.argmax(prediction['probabilities']) for prediction in predictions]
    dtest['prediction']=x
    print("The accuracy of FLTR on "+FLAGS.category_name+" is: "+str(accuracy_score(dtest.label,dtest.prediction)))
    
    if FLAGS.do_predict:
        print("Beginning Prediction!")
        data_with_FLTR_predictions = FLTR_Prediction(data,tokenizer,estimator)
        
        if(data_with_FLTR_predictions.isnull().values.any()):
            data_with_FLTR_predictions = data_with_FLTR_predictions.replace(np.nan, "[PAD]", regex=True)
            
        data_with_FLTR_predictions.to_csv(os.path.join(FLAGS.data_dir, 
                                                       FLAGS.category_name+'.txt'), 
                                          index=None, sep='\t', mode='w') 
        print("Prediction End!")


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("category_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("model_output_dir")
    tf.app.run()
