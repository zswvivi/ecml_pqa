This repository holds data and source code for our ECML-PKDD 2020 paper titled: 
# Less is More: Rejecting Unreliable Reviews for Product Question Answering 
[Paper link](https://arxiv.org/abs/2007.04526)

# Deep Learning Packages Requirements
- python3.6
- tensorflow-gpu 1.11

(Note: We use the [Bert Model](https://github.com/google-research/bert) and pretrained Bert weights [BERT-Base, Uncased: 12-layer, 768-hidden, 12-heads, 110M parameters](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip))

# Data
1. The Amazon PQA dataset (including reviews and QA data) are provided by [dataset link](http://cseweb.ucsd.edu/~jmcauley/datasets.html).
2. In this work, we tested our models on data in the following four categories, namely Tools_and_Home_Improvement, Patio_Lawn_and_Garden, Electroincis and Baby.
3. We released our annotated data which have 200 questions and 4k reviews, in data folder.


# Environment Varibles
1. export DATA_DIR=/path/to/data
2. export MODEL_DIR=/path/for/keeping/trained/models
3. export BERT_DIR=/path/to/bert/uncased_L-12_H-768_A-12

# Dataset preprocessing
python run_preproceesing_data.py --data_dir=$DATA_DIR 

# FLTR Training 
1. Cross-domian pretraining:  python run_FLTR.py --data_dir=$DATA_DIR --model_output_dir=$MODEL_DIR --init_checkpoint=$BERT_DIR/bert_model.ckpt --bert_config_file=$BERT_DIR/bert_config.json --vocab_file=$BERT_DIR/vocab.txt --category_name="All_Categories"
2. Training on each category, e.g. 'Baby': python run_FLTR.py --data_dir=$DATA_DIR --do_predict=True --model_output_dir=$MODEL_DIR --init_checkpoint=$MODEL_DIR /All_Categories_FLTR --bert_config_file=$BERT_DIR/bert_config.json --vocab_file=$BERT_DIR/vocab.txt --category_name="Baby"

# BertQA Training
1. Cross-domian pretraining: python run_BertQA.py --data_dir=$DATA_DIR --model_output_dir=$MODEL_DIR --init_checkpoint=$BERT_DIR/bert_model.ckpt --bert_config_file=$BERT_DIR/bert_config.json --vocab_file=$BERT_DIR/vocab.txt --category_name="All_Categories"
2. Training on each category, e.g. 'Baby': python run_BertQA.py --data_dir=$DATA_DIR --model_output_dir=$MODEL_DIR --init_checkpoint=$MODEL_DIR /All_Categories_BertQA --bert_config_file=$BERT_DIR/bert_config.json --vocab_file=$BERT_DIR/vocab.txt --category_name="Baby"

# Predict annotated data (test data) using both trained FLTR and BertQA
python predict_test_data.py --data_dir=$DATA_DIR  --model_output_dir=$MODEL_DIR --bert_config_file=$BERT_DIR/bert_config.json --vocab_file=$BERT_DIR/vocab.txt

# Run rejection framework
python run_rejection_framework.py --data_dir=$DATA_DIR 
