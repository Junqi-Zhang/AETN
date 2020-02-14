import os
import time
import shutil
import numpy as np
import tensorflow as tf

from pathlib import Path
from .model_fn import get_model

def dataset_generator(params, cycle_length, num_parallel_calls, prefetch):

    data_dir = params.get('data_dir')
    file_stamp = params.get('file_stamp')
    skip = int(params.get('header'))

    shuffle = params.get('shuffle')
    batch_size = params.get('batch_size')
    buffer_batch_num = params.get('buffer_batch_num', 1)

    data_map = params.get('data_map')
    input_keys = params.get('input_keys')

    num_softid = params.get('num_softid')
    num_label = params.get('num_label')
    length_his = params.get('length_his')
    bert_mask_num = params.get('bert_mask_num')


    def _parse_line(line):

        input_data = dict.fromkeys(input_keys)
        line_split = tf.strings.split([line], '|').values

        # process uid
        input_data['uid'] = tf.strings.to_number(line_split[data_map['uid']], out_type=tf.int64)


        def _string_to_sparse(string, dense_shape):
            number = tf.strings.to_number(
                            tf.strings.split([string], sep=',').values,
                            out_type=tf.int64)
            return tf.sparse.SparseTensor(
                        indices=tf.expand_dims(number, axis=1),
                        values=tf.ones_like(number, dtype=tf.int8),
                        dense_shape=dense_shape)

        # process retention
        input_data['image'] = tf.sparse.to_dense(tf.cond(tf.strings.length(line_split[data_map['image']]) > 0,
                                true_fn=lambda: _string_to_sparse(line_split[data_map['image']], dense_shape=[num_softid]),
                                false_fn=lambda: tf.sparse.SparseTensor(
                                                indices=np.empty((0,1), dtype=np.int64),
                                                values=tf.constant([], dtype=tf.int8),
                                                dense_shape=[num_softid])))

        # process label
        label = tf.sparse.to_dense(tf.cond(tf.strings.length(line_split[data_map['label']]) > 0,
                        true_fn=lambda: _string_to_sparse(line_split[data_map['label']], dense_shape=[num_label]),
                        false_fn=lambda: tf.sparse.SparseTensor(
                                            indices=np.empty((0,1), dtype=np.int64),
                                            values=tf.constant([], dtype=tf.int8),
                                            dense_shape=[num_label])))


        # process installation and uninstallation
        def _string_to_list(string, padding):
            number = tf.strings.to_number(
                            tf.strings.split([string], sep=',').values,
                            out_type=tf.int32)
            return tf.concat([tf.tile(tf.constant([padding], dtype=tf.int32),
                                        tf.math.maximum(length_his-tf.shape(number), 0)),
                                number[-length_his:]], axis=0)

        input_data['new_time_list'] = tf.cond(tf.strings.length(line_split[data_map['new_time_list']]) > 0,
                                        true_fn=lambda: _string_to_list(line_split[data_map['new_time_list']], padding=-7) // 7,
                                        false_fn=lambda: tf.tile(tf.constant([-1], dtype=tf.int32), [length_his]))

        input_data['new_soft_list'] = tf.cond(tf.strings.length(line_split[data_map['new_soft_list']]) > 0,
                                        true_fn=lambda: _string_to_list(line_split[data_map['new_soft_list']], padding=-1),
                                        false_fn=lambda: tf.tile(tf.constant([-1], dtype=tf.int32), [length_his]))

        input_data['new_mask_list'] = tf.to_int32(tf.greater_equal(input_data['new_time_list'],
                                                        tf.tile(tf.constant([0], dtype=tf.int32), [length_his])))

        input_data['new_bert_mask_index'] = tf.reverse(tf.math.top_k(
                                        tf.random.shuffle(tf.constant(np.asarray(range(length_his)), dtype=tf.int8))[:bert_mask_num],
                                        k=bert_mask_num).values, axis=[0])

        input_data['new_bert_mask'] = tf.sparse.to_dense(tf.sparse.SparseTensor(
                                        indices=tf.expand_dims(tf.to_int64(input_data['new_bert_mask_index']), axis=1),
                                        values=tf.ones_like(input_data['new_bert_mask_index'], dtype=tf.int8),
                                        dense_shape=[length_his]))

        input_data['loss_time_list'] = tf.cond(tf.strings.length(line_split[data_map['loss_time_list']]) > 0,
                                        true_fn=lambda: _string_to_list(line_split[data_map['loss_time_list']], padding=-7) // 7,
                                        false_fn=lambda: tf.tile(tf.constant([-1], dtype=tf.int32), [length_his]))

        input_data['loss_soft_list'] = tf.cond(tf.strings.length(line_split[data_map['loss_soft_list']]) > 0,
                                        true_fn=lambda: _string_to_list(line_split[data_map['loss_soft_list']], padding=-1),
                                        false_fn=lambda: tf.tile(tf.constant([-1], dtype=tf.int32), [length_his]))

        input_data['loss_mask_list'] = tf.to_int32(tf.greater_equal(input_data['loss_time_list'],
                                                        tf.tile(tf.constant([0], dtype=tf.int32), [length_his])))

        input_data['loss_bert_mask_index'] = tf.reverse(tf.math.top_k(
                                        tf.random.shuffle(tf.constant(np.asarray(range(length_his)), dtype=tf.int8))[:bert_mask_num],
                                        k=bert_mask_num).values, axis=[0])

        input_data['loss_bert_mask'] = tf.sparse.to_dense(tf.sparse.SparseTensor(
                                        indices=tf.expand_dims(tf.to_int64(input_data['loss_bert_mask_index']), axis=1),
                                        values=tf.ones_like(input_data['loss_bert_mask_index'], dtype=tf.int8),
                                        dense_shape=[length_his]))

        return input_data, label

    files = tf.data.Dataset.list_files(data_dir + '/*.' + file_stamp, shuffle=shuffle)
    dataset = files.interleave(lambda x: tf.data.TextLineDataset(x).skip(skip),
                                cycle_length=cycle_length, block_length=1, num_parallel_calls=None)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_batch_num * batch_size)

    dataset = dataset.map(_parse_line,
                num_parallel_calls=num_parallel_calls).batch(batch_size).prefetch(prefetch)

    return dataset


class ModelWrapper(object):

    def __init__(self, params, model_stamp, num_gpus=None, checkpoint_dir=None):
        self.params = params
        self.model_stamp = model_stamp
        self.num_gpus = num_gpus

        self.best_model_dir = './models/best_model/%s' % model_stamp
        self.checkpoint_dir = './models/checkpoints/%s' % model_stamp if checkpoint_dir is None else checkpoint_dir
        self.export_dir = './models/export/%s' % model_stamp

        self.mirrored_strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=self.num_gpus)
        self.config = tf.estimator.RunConfig(train_distribute=self.mirrored_strategy,
                                                eval_distribute=self.mirrored_strategy)
        self.model = tf.estimator.Estimator(model_fn=get_model,
                                            model_dir=self.checkpoint_dir,
                                            config=self.config,
                                            params=self.params)

        self.monitor_metric = params.get('monitor_metric')
        if 'loss' in self.monitor_metric:
            self.best_metric = 1e+7
        else:
            self.best_metric = 0
        self.best_epoch = 0

    def save(self):
        path = Path('./models/best_model')
        if not path.exists():
            path.mkdir()
        path = Path(self.best_model_dir)
        if path.exists():
            shutil.rmtree(self.best_model_dir)
        shutil.copytree(self.checkpoint_dir, self.best_model_dir)

    def train(self, train_input_fn, eval_input_fn=None, n_epochs=10,
              steps_per_epoch=None, early_stopping=None, hooks=None):
        print('\nTraining %s ...' % self.model_stamp)

        for epoch in range(1, n_epochs + 1):
            print('\nEpoch %d/%d' % (epoch, n_epochs))

            time0 = time.time()
            self.model.train(train_input_fn, steps=steps_per_epoch, hooks=hooks)
            train_time = time.time() - time0

            if eval_input_fn is None or (steps_per_epoch is not None and epoch < np.ceil(
                                self.params['num_train_sample']/self.params['batch_size']/steps_per_epoch)):
                print('    Training done. Cost time: %.2fs' % train_time)
                self.best_epoch = epoch
                self.save()
            else:
                time0 = time.time()
                eval_results = self.model.evaluate(eval_input_fn)
                print('    Training / Evaluation done. Cost time: %.2fs / %.2fs' %
                      (train_time, time.time() - time0))
                print('    Evaluation metric: %s' % eval_results)

                epoch_metric = eval_results[self.monitor_metric]

                if 'loss' in self.monitor_metric:
                    if epoch_metric < self.best_metric:
                        print('    New best epoch with epoch_metric =', epoch_metric)
                        self.best_metric = epoch_metric
                        self.best_epoch = epoch
                        self.save()
                else:
                    if epoch_metric > self.best_metric:
                        print('    New best epoch with epoch_metric =', epoch_metric)
                        self.best_metric = epoch_metric
                        self.best_epoch = epoch
                        self.save()

            if early_stopping is not None:
                if epoch - self.best_epoch >= early_stopping:
                    break

    def evaluate(self, input_fn, eval_or_test='Evaluation'):
        eval_results = self.model.evaluate(input_fn, checkpoint_path=tf.train.latest_checkpoint(self.best_model_dir))
        print(eval_or_test+' metric: %s' % eval_results)
        return eval_results

    def predict(self, input_fn):
        print('    Predicting...')
        return self.model.predict(input_fn, checkpoint_path=tf.train.latest_checkpoint(self.best_model_dir),
                                    yield_single_examples=False)
