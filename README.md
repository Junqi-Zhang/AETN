# AETN
The AutoEncoder-coupled Transformer Network is a novel model for generating General-Purpose User Embeddings based on Mobile App Usage.
We regret that the dataset we use to train and evaluate the models cannot to made public because of user data privacy policy at Tencent.
To help researchers reproduce our model on their own datasets, we make the implementation details public through this code.

This implementation is a TensorFlow version, supporting multi-GPUs training, and employs the tf.data.Dataset and tf.estimator APIs. You can train the AETN or the V-AETN with optional task of auxiliary retention reconstruction and masked app prediction by run the train.py. You can also fine-tune the models with downstream tasks through the codes.

## System requirements
* Python >= 3.6.5
* Numpy >= 1.14.3
* Pandas >= 0.23.0
* TensorFlow-gpu = 1.12
* Total memory of GPUs >= 16G

Attention! The code won't work without GPUs, even if you modify the os.environ["CUDA_VISIBLE_DEVICES"]='' in train.py.

## Components
1. 'modules/losses.py' for loss functions
2. 'modules/models.py' where AETN and V-AETN are defined
3. 'modules/model_fn.py' where the model_fn for the tf.estimator are defined
4. 'modules/utils.py' including dataset_generator and ModelWrapper
5. 'train.py' the main code for training models
6. 'softmap_plus_sample/' a sample folder where the information of apps considered in training process are provided
7. 'train/valid/test_sample/' sample folders where the training/validation/testing dataset are located

## Settings in train.py
### Dataset Path
```Python
DATA_DIR = './'
SOFTMAP_PLUS_DIR = DATA_DIR + 'softmap_plus_sample/'
TRAIN_DIR = DATA_DIR + 'train_sample'
VAL_DIR = DATA_DIR + 'valid_sample'
TEST_DIR = DATA_DIR + 'test_sample'

NUM_TRAIN_SAMPLE = 1.6e+7 # the number of users for training
NUM_SOFTID = len(SOFTMAP_PLUS) # number of apps considered in training process
NUM_LABEL = 6 # for the fine-tuning tasks
```
### The column index for the .csv source data files
```Python
data_map = {
    'uid': 0, # user id
    'image': 1, # the retention
    'label': 2, # the label for fine-tuning tasks
    'new_time_list': 3, # app sequence for installation
    'new_soft_list': 4, # date sequence for installation
    'loss_time_list': 5, # app sequence for uninstallation
    'loss_soft_list': 6, # date sequence for uninstallation
}
```
### Keys for the feature dict generated by tf.data.Dataset()
```Python
input_keys = [key for key in data_map.keys() if key != 'label']
input_keys.extend(['new_mask_list','loss_mask_list',
                    'new_bert_mask_index','new_bert_mask',
                    'loss_bert_mask_index','loss_bert_mask'])
```
### Configuration for train/valid/test dataset
```Python
train_data_params = {
    'data_dir': TRAIN_DIR,
    'file_stamp': 'csv',
    'header': True, # True for .csv files with headers

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
```
### Model Settings
1. MODEL_TYPE = AETN or MODEL_TYPE = VAETN
2. For training with the masked app prediction task, 'train_mlm' should be set to True.
3. For training with the auxiliary retention reconstruction task, 'alpha_dae' should be set to 1.0.
4. For fine-tuning models, 'alpha_classifier' should be set to 1.0.
5. For training without the main reconstruction, 'alpha_main' should be set to 0.
```Python
LENGTH_HIS = 25 # the length of (un)installation sequences
BERT_MASK_NUM = 3 # the number of masked apps
BATCH_SIZE = int(1000/NUM_GPUS) # batch_size for each GPU
print('Batch_Size is:', BATCH_SIZE * NUM_GPUS) # show the real batch_size for training

model_params = {
    'model_type': MODEL_TYPE,
    'num_train_sample': NUM_TRAIN_SAMPLE,
    'batch_size': BATCH_SIZE,

    'train_mlm': True, # True for training with the BERT-like masked app prediction task
    'p_dims': [256, 512, NUM_SOFTID], # the structure of the autoencoder part
    'q_dims': None,
    'u_dims': 128, # the size of the bottleneck layer
    'l_dims': [1024, NUM_LABEL], # MLP for the fine_tuning tasks
    'ffn_dim': 1024, # the size of inner layer of the FFN in transformers
    'num_translayer': 2, # the number of layers in transformer encoder
    'num_header': 8, # the number of attention headers
    'num_date': 52, # the number of dates
    'length_his': LENGTH_HIS,
    'softmap': SOFTMAP_PLUS['type_idx'].values, # the categories for every app in consideration
    'softmap_vocab': len(SOFTMAP_PLUS[['type', 'type_idx']].drop_duplicates()), # the number of categories

    'image_dropout_rate': 0.05,
    'classifier_dropout_rate': 0.05,
    'attention_dropout_rate': 0.1,
    'ffn_dropout_rate': 0.1,

    'lambda_L2': 1.5e-7,

    'loss_type_dae': 'logistic',
    'weight_positive_dae': 100,
    'alpha_dae': 1.0, # 1.0 for training with the auxiliary retention reconstruction task

    'loss_type_image': 'logistic',
    'weight_positive_image': 100,

    'loss_type_his': 'softmax',
    'weight_loss_his': 1.0,

    'alpha_main': 1.0, # 1.0 for training with the main reconstruction task

    'loss_type_classifier': 'logistic',
    'weight_positive_classifier': 100,
    'alpha_classifier': 0, # 1.0 for fine-tuning

    'optimizer': 'Adam',
    'init_lr': 1e-4,
    'decay_steps': 16000,
    'decay_rate': 0.8,

    'monitor_metric': 'loss_main',

    # retention
    'image': [tf.feature_column.numeric_column(key='image',
                                                shape=(NUM_SOFTID,),
                                                dtype=tf.int8)],
    # dates for installation
    'new_time_list': [tf.feature_column.numeric_column(key='new_time_list',
                                                        shape=(LENGTH_HIS,),
                                                        dtype=tf.int32)],
    # apps for installation
    'new_soft_list': [tf.feature_column.numeric_column(key='new_soft_list',
                                                        shape=(LENGTH_HIS,),
                                                        dtype=tf.int32)],
    # masks for installation
    'new_mask_list': [tf.feature_column.numeric_column(key='new_mask_list',
                                                        shape=(LENGTH_HIS,),
                                                        dtype=tf.int32)],
    # index for masked apps in installation
    'new_bert_mask_index': [tf.feature_column.numeric_column(key='new_bert_mask_index',
                                                        shape=(BERT_MASK_NUM,),
                                                        dtype=tf.int8)],
    # bert masks for installation
    'new_bert_mask': [tf.feature_column.numeric_column(key='new_bert_mask',
                                                        shape=(LENGTH_HIS,),
                                                        dtype=tf.int8)],
    # dates for uninstallation
    'loss_time_list': [tf.feature_column.numeric_column(key='loss_time_list',
                                                        shape=(LENGTH_HIS,),
                                                        dtype=tf.int32)],
    # apps for uninstallation
    'loss_soft_list': [tf.feature_column.numeric_column(key='loss_soft_list',
                                                        shape=(LENGTH_HIS,),
                                                        dtype=tf.int32)],
    # masks for uninstallation
    'loss_mask_list': [tf.feature_column.numeric_column(key='loss_mask_list',
                                                        shape=(LENGTH_HIS,),
                                                        dtype=tf.int32)],
    # index for masked apps in uninstallation
    'loss_bert_mask_index': [tf.feature_column.numeric_column(key='loss_bert_mask_index',
                                                        shape=(BERT_MASK_NUM,),
                                                        dtype=tf.int8)],
    # bert masks for uninstallation
    'loss_bert_mask': [tf.feature_column.numeric_column(key='loss_bert_mask',
                                                        shape=(LENGTH_HIS,),
                                                        dtype=tf.int8)]
}
```
