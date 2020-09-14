This repository holds data and source code for our ECML-PKDD 2020 paper titled: 
# Less is More: Rejecting Unreliable Reviews for Product Question Answering 
[Paper link](https://arxiv.org/abs/2007.04526)


# Deep Learning Packages Requirements
- python3.6
- tensorflow-gpu 1.11

(Note: We built our models upon the [Bert Model](https://github.com/google-research/bert) and pretrained Bert weights [BERT-Base, Uncased: 12-layer, 768-hidden, 12-heads, 110M parameters](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip))

# Data
1. The Amazon PQA dataset (including reviews and QA data) can be downloaded from [dataset link](http://cseweb.ucsd.edu/~jmcauley/datasets.html).
2. In this work, we tested our models on data in the following four categories, namely Tools_and_Home_Improvement, Patio_Lawn_and_Garden, Electronics and Baby.
3. We released our annotated data which have 200 questions and 4k reviews, located in data folder.


# Environment Variables
1. export DATA_DIR=/path/to/data
2. export MODEL_DIR=/path/for/keeping/trained/models
3. export BERT_DIR=/path/to/bert/uncased_L-12_H-768_A-12

# Dataset preprocessing
python run_preproceesing_data.py --data_dir=$DATA_DIR 

# Training PQA Models

1. FTLR is basically a binary classifier, but trained on QA pairs and tested on Question/Reviews pairs.
2. BertQA is a Mixture-of-Expert based model, which is trying to identify the correct answer. 
In BertQA, question does not meet answers. Instead, reviews meet question and answers, and then score each answer. 
The higher score means more possible to be the correct answer.
The loss function is actually a margin loss, where we maxmize the distance of score of correct answer and non-answer.
As you see from BertQA model, the input size is massive. It is impossible to fit all of reviews into the model.
What we do here is to filter reviews. In practice, we choose to use the top 10 reivews ranked by FLTR.
Even top 10 reviews are still a large input, since 10 reviews will be paired with question, answer and non-answer, so 30 pairs.
As you know, BERT is a massive model, we are not allowed to have a big batch size for training because of GPU memory.
In practice, we can only have 6 questions per batch, 6 questions will be 180 pairs (30*6), and our GPU is Nvidia V100.

![alt text](https://github.com/zswvivi/ecml_pqa/blob/master/figures/PQA_Models.png)

# FLTR Training 
1. Cross-domian pretraining:

python run_FLTR.py --data_dir=$DATA_DIR --model_output_dir=$MODEL_DIR --init_checkpoint=$BERT_DIR/bert_model.ckpt --bert_config_file=$BERT_DIR/bert_config.json --vocab_file=$BERT_DIR/vocab.txt --category_name="All_Categories"

2. Training on each category, e.g. 'Baby':

python run_FLTR.py --data_dir=$DATA_DIR --do_predict=True --model_output_dir=$MODEL_DIR --init_checkpoint=$MODEL_DIR /All_Categories_FLTR --bert_config_file=$BERT_DIR/bert_config.json --vocab_file=$BERT_DIR/vocab.txt --category_name="Baby"

# BertQA Training
1. Cross-domian pretraining:

python run_BertQA.py --data_dir=$DATA_DIR --model_output_dir=$MODEL_DIR --init_checkpoint=$BERT_DIR/bert_model.ckpt --bert_config_file=$BERT_DIR/bert_config.json --vocab_file=$BERT_DIR/vocab.txt --category_name="All_Categories"

2. Training on each category, e.g. 'Baby': 

python run_BertQA.py --data_dir=$DATA_DIR --model_output_dir=$MODEL_DIR --init_checkpoint=$MODEL_DIR /All_Categories_BertQA --bert_config_file=$BERT_DIR/bert_config.json --vocab_file=$BERT_DIR/vocab.txt --category_name="Baby"

# Predict annotated data (test data) using both trained FLTR and BertQA
python predict_test_data.py --data_dir=$DATA_DIR  --model_output_dir=$MODEL_DIR --bert_config_file=$BERT_DIR/bert_config.json --vocab_file=$BERT_DIR/vocab.txt

# Run rejection framework
python run_rejection_framework.py --data_dir=$DATA_DIR 
