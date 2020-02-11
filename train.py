import os
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from modules.utils import dataset_generator, ModelWrapper

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
NUM_GPUS = 4

TRUE_MODEL_TYPE = 'AETN'
MODEL_TYPE = 'AETN'

DATA_DIR = '../app_unsupervised/'
SOFTMAP_DIR = DATA_DIR + 'softmap_20200105/'
SOFTMAP_PLUS_DIR = DATA_DIR + 'softmap2_20200105/'
TRAIN_DIR = DATA_DIR + 'train_20200105'
VAL_DIR = DATA_DIR + 'valid_20200105'
TEST_DIR = DATA_DIR + 'test_wifi_20200105'

SOFTMAP = pd.read_parquet(SOFTMAP_DIR + [file for file in os.listdir(SOFTMAP_DIR) if 'parquet' in file][0])
SOFTMAP = SOFTMAP.sort_values('soft_id', ascending=True)
print(SOFTMAP[['type','type_idx']].drop_duplicates())

SOFTMAP_PLUS = pd.read_parquet(SOFTMAP_PLUS_DIR + [file for file in os.listdir(SOFTMAP_PLUS_DIR) if 'parquet' in file][0])
print(SOFTMAP_PLUS[['type', 'type_idx']].drop_duplicates())
SOFTMAP_PLUS.drop_duplicates('soft_id', inplace=True)
SOFTMAP_PLUS = SOFTMAP_PLUS.sort_values('soft_id', ascending=True)

NUM_TRAIN_SAMPLE = 1.6e+7
NUM_SOFTID = len(SOFTMAP['soft_id'].unique())
NUM_LABEL = len(SOFTMAP['type_idx'].unique()) - 1 # exclude type 'others'
LENGTH_HIS = 25
BERT_MASK_NUM = 3
BATCH_SIZE = int(1000/NUM_GPUS)
print('Batch_Size is:', BATCH_SIZE * NUM_GPUS)

data_map = {
    'uid': 0,
    'image': 1,
    'label': 2,
    'new_time_list': 3,
    'new_soft_list': 4,
    'loss_time_list': 5,
    'loss_soft_list': 6,
}

input_keys = [key for key in data_map.keys() if key != 'label']
input_keys.extend(['new_mask_list','loss_mask_list',
                    'new_bert_mask_index','new_bert_mask',
                    'loss_bert_mask_index','loss_bert_mask'])

train_data_params = {
    'data_dir': TRAIN_DIR,
    'file_stamp': 'csv',
    'header': True,

    'shuffle': True,
    'batch_size': BATCH_SIZE,
    'buffer_batch_num': 100,

    'data_map': data_map,
    'input_keys': input_keys,

    'num_softid': NUM_SOFTID,
    'num_label': NUM_LABEL,
    'length_his': LENGTH_HIS,
    'bert_mask_num': BERT_MASK_NUM
}

val_data_params = {
    'data_dir': VAL_DIR,
    'file_stamp': 'csv',
    'header': True,

    'shuffle': False,
    'batch_size': BATCH_SIZE,

    'data_map': data_map,
    'input_keys': input_keys,

    'num_softid': NUM_SOFTID,
    'num_label': NUM_LABEL,
    'length_his': LENGTH_HIS,
    'bert_mask_num': BERT_MASK_NUM
}

test_data_params = {
    'data_dir': TEST_DIR,
    'file_stamp': 'csv',
    'header': True,

    'shuffle': False,
    'batch_size': BATCH_SIZE,

    'data_map': data_map,
    'input_keys': input_keys,

    'num_softid': NUM_SOFTID,
    'num_label': NUM_LABEL,
    'length_his': LENGTH_HIS,
    'bert_mask_num': BERT_MASK_NUM
}

model_params = {
    'model_type': MODEL_TYPE,
    'num_train_sample': NUM_TRAIN_SAMPLE,
    'batch_size': BATCH_SIZE,

    'train_mlm': True,
    'p_dims': [256, 512, NUM_SOFTID],
    'q_dims': None,
    'u_dims': 128,
    'l_dims': [1024, NUM_LABEL],
    'ffn_dim': 1024,
    'num_translayer': 2,
    'num_header': 8,
    'num_date': 52,
    'length_his': LENGTH_HIS,
    'softmap': SOFTMAP_PLUS['type_idx'].values,
    'softmap_vocab': len(SOFTMAP_PLUS[['type', 'type_idx']].drop_duplicates()),

    'image_dropout_rate': 0.05,
    'classifier_dropout_rate': 0.05,
    'attention_dropout_rate': 0.1,
    'ffn_dropout_rate': 0.1,

    'lambda_L2': 1.5e-7,

    'loss_type_dae': 'logistic',
    'weight_positive_dae': 100,
    'alpha_dae': 1.0,

    'loss_type_image': 'logistic',
    'weight_positive_image': 100,

    'loss_type_his': 'softmax',
    'weight_loss_his': 1.0,

    'alpha_main': 1.0,

    'loss_type_classifier': 'logistic',
    'weight_positive_classifier': 100,
    'alpha_classifier': 0,

    'optimizer': 'Adam',
    'init_lr': 1e-4,
    'decay_steps': 16000,
    'decay_rate': 0.8,

    'monitor_metric': 'loss_main',

    'image': [tf.feature_column.numeric_column(key='image',
                                                shape=(NUM_SOFTID,),
                                                dtype=tf.int8)],
    'new_time_list': [tf.feature_column.numeric_column(key='new_time_list',
                                                        shape=(LENGTH_HIS,),
                                                        dtype=tf.int32)],
    'new_soft_list': [tf.feature_column.numeric_column(key='new_soft_list',
                                                        shape=(LENGTH_HIS,),
                                                        dtype=tf.int32)],
    'new_mask_list': [tf.feature_column.numeric_column(key='new_mask_list',
                                                        shape=(LENGTH_HIS,),
                                                        dtype=tf.int32)],
    'new_bert_mask_index': [tf.feature_column.numeric_column(key='new_bert_mask_index',
                                                        shape=(BERT_MASK_NUM,),
                                                        dtype=tf.int8)],
    'new_bert_mask': [tf.feature_column.numeric_column(key='new_bert_mask',
                                                        shape=(LENGTH_HIS,),
                                                        dtype=tf.int8)],
    'loss_time_list': [tf.feature_column.numeric_column(key='loss_time_list',
                                                        shape=(LENGTH_HIS,),
                                                        dtype=tf.int32)],
    'loss_soft_list': [tf.feature_column.numeric_column(key='loss_soft_list',
                                                        shape=(LENGTH_HIS,),
                                                        dtype=tf.int32)],
    'loss_mask_list': [tf.feature_column.numeric_column(key='loss_mask_list',
                                                        shape=(LENGTH_HIS,),
                                                        dtype=tf.int32)],
    'loss_bert_mask_index': [tf.feature_column.numeric_column(key='loss_bert_mask_index',
                                                        shape=(BERT_MASK_NUM,),
                                                        dtype=tf.int8)],
    'loss_bert_mask': [tf.feature_column.numeric_column(key='loss_bert_mask',
                                                        shape=(LENGTH_HIS,),
                                                        dtype=tf.int8)]
}

def build_session(gpu_fraction=0.95, allow_growth=True):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=allow_growth)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


sess = build_session()
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
sess.run(init_op)

tf.logging.set_verbosity(tf.logging.INFO)

train_input_fn = lambda: dataset_generator(train_data_params, cycle_length=2, num_parallel_calls=5, prefetch=8)
val_input_fn = lambda: dataset_generator(val_data_params, cycle_length=2, num_parallel_calls=5, prefetch=8)
test_input_fn = lambda: dataset_generator(test_data_params, cycle_length=1, num_parallel_calls=5, prefetch=8)

model_stamp = TRUE_MODEL_TYPE + '_Model@%s' % time.strftime('%m%d_%H%M%S',
                            time.localtime(time.time())) + '_%04d' % np.random.randint(0, 10000)
model = ModelWrapper(model_params, model_stamp, num_gpus=NUM_GPUS)

model.train(train_input_fn, val_input_fn, n_epochs=50, steps_per_epoch=None, early_stopping=1)
