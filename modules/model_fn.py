import tensorflow as tf
from .losses import *
from .models import *

def get_optimizer(optimizer):
    switch = {
        'adam': tf.train.AdamOptimizer,
        'rmsprop': tf.train.RMSPropOptimizer,
        'sgd': tf.train.GradientDescentOptimizer
    }

    try:
        opt = switch[optimizer.lower()]
    except KeyError:
        raise NotImplementedError
    else:
        return opt


def get_loss(loss_type):
    switch = {
        'logistic': logistic_loss, # (labels, logits, a=100, label_smoothing=0.0)
        'softmax': softmax_loss, # (onehot_labels, logits, weights=1.0, label_smoothing=0.0)
        'KL': kullback_leibler_divergence_loss, # (labels, logits, a(useless), label_smoothing=0.0)
        'multinomial': multinomial_loss # (labels, logits, a(useless), label_smoothing=0.0)
    }

    try:
        loss = switch[loss_type]
    except KeyError:
        raise NotImplementedError
    else:
        return loss


def get_metric(metric_type):
    switch = {
        'auc': tf.metrics.auc
    }

    try:
        metric = switch[metric_type]
    except KeyError:
        raise NotImplementedError
    else:
        return metric


def get_forward_pass(model_type):
    switch = {
        'AETN': AETN,
        'VAETN': VAETN
    }

    try:
        forward_pass = switch[model_type]
    except KeyError:
        raise NotImplementedError
    else:
        return forward_pass


def get_model(features, labels, mode, params):
    # parse hyper-parameters
    model_type = params.get('model_type', 'AETN')

    NUM_SOFTID = params['p_dims'][-1]

    lambda_L2 = params.get('lambda_L2', 1.5e-7)

    loss_type_dae = params.get('loss_type_dae', 'logistic')
    weight_positive_dae = params.get('weight_positive_dae', 100)
    alpha_dae = params.get('alpha_dae', 1)

    loss_type_image = params.get('loss_type_image', 'logistic')
    weight_positive_image = params.get('weight_positive_image', 100)

    loss_type_his = params.get('loss_type_his', 'softmax')
    weight_loss_his = params.get('weight_loss_his', 1)

    alpha_main = params.get('alpha_main', 1)

    loss_type_classifier = params.get('loss_type_classifier', 'logistic')
    weight_positive_classifier = params.get('weight_positive_classifier', 100)
    alpha_classifier = params.get('alpha_classifier', 0)

    optimizer = params.get('optimizer', 'Adam')
    init_lr = params.get('init_lr', 1e-4)
    decay_steps = params.get('decay_steps', 3000)
    decay_rate = params.get('decay_rate', 0.8)

    train_mlm = params.get('train_mlm', True)

    NUM_LABEL = params['l_dims'][-1]

    # fetch inputs
    inputs = []
    inputs.append(tf.feature_column.input_layer(features, params['image']))
    inputs.append(tf.feature_column.input_layer(features, params['new_time_list']))
    inputs.append(tf.feature_column.input_layer(features, params['new_soft_list']))
    inputs.append(tf.feature_column.input_layer(features, params['new_mask_list']))
    inputs.append(tf.feature_column.input_layer(features, params['new_bert_mask_index']))
    inputs.append(tf.feature_column.input_layer(features, params['new_bert_mask']))
    inputs.append(tf.feature_column.input_layer(features, params['loss_time_list']))
    inputs.append(tf.feature_column.input_layer(features, params['loss_soft_list']))
    inputs.append(tf.feature_column.input_layer(features, params['loss_mask_list']))
    inputs.append(tf.feature_column.input_layer(features, params['loss_bert_mask_index']))
    inputs.append(tf.feature_column.input_layer(features, params['loss_bert_mask']))

    # get_model
    model = get_forward_pass(model_type)(params, inputs, mode)
    # forward pass
    model_outputs = model.forward_pass()

    logits_dae = model_outputs[0]
    user_embeddings = model_outputs[1]
    logits_bert_new = model_outputs[2]
    logits_bert_loss = model_outputs[3]
    logits_image = model_outputs[4]
    logits_new = model_outputs[5]
    logits_loss = model_outputs[6]
    logits_classifier = model_outputs[7]

    # predictions
    predictions = {
        'uid': features['uid'],
        'user_embeddings': user_embeddings
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode,
            predictions=predictions,
            export_outputs={'outputs': tf.estimator.export.PredictOutput(predictions)})

    # define loss
    loss_dae = get_loss(loss_type_dae)(inputs[0], logits_dae, weight_positive_dae) # loss for auxiliary retention reconsrtuction
    loss_image = get_loss(loss_type_image)(inputs[0], logits_image, weight_positive_image) # loss for main retention reconstruction

    new_one_hot = tf.one_hot(tf.to_int32(inputs[2]), NUM_SOFTID, dtype=tf.int32)
    loss_new = get_loss(loss_type_his)(new_one_hot, logits_new, weight_loss_his*inputs[3]) # loss for installation reconstruction

    loss_one_hot = tf.one_hot(tf.to_int32(inputs[7]), NUM_SOFTID, dtype=tf.int32)
    loss_loss = get_loss(loss_type_his)(loss_one_hot, logits_loss, weight_loss_his*inputs[8]) # loss for uninstallation reconstruction

    new_bert_one_hot = tf.one_hot(tf.batch_gather(tf.to_int32(inputs[2]), tf.to_int32(inputs[4])), NUM_SOFTID, dtype=tf.int32)
    # loss for masked app prediction in installtion
    loss_bert_new = get_loss(loss_type_his)(new_bert_one_hot, logits_bert_new, weight_loss_his*tf.batch_gather(inputs[3],tf.to_int32(inputs[4])))

    loss_bert_one_hot = tf.one_hot(tf.batch_gather(tf.to_int32(inputs[7]), tf.to_int32(inputs[9])), NUM_SOFTID, dtype=tf.int32)
    # loss for masked app prediction in uninstalltion
    loss_bert_loss = get_loss(loss_type_his)(loss_bert_one_hot, logits_bert_loss, weight_loss_his*tf.batch_gather(inputs[8],tf.to_int32(inputs[9])))

    # loss for fine-tuning tasks
    loss_classifier = get_loss(loss_type_classifier)(tf.to_float(labels), logits_classifier, weight_positive_classifier)

    if alpha_main > 0:
        loss = alpha_main * (loss_image + loss_new + loss_loss) + \
                model.compute_bottleneck_L2() + model.compute_transformer_L2()
        if alpha_dae > 0:
            loss += alpha_dae * loss_dae
            loss += model.compute_dae_L2()
        else:
            loss += model.compute_dae_L2(only_part=True)
        if train_mlm:
            loss += train_mlm * (loss_bert_new + loss_bert_loss)
        if alpha_classifier > 0:
            loss += alpha_classifier * loss_classifier
            loss += model.compute_classifier_L2()
    else:
        loss = alpha_dae * loss_dae + model.compute_dae_L2()
        if train_mlm:
            loss += train_mlm * (loss_bert_new + loss_bert_loss)
            loss += model.compute_transformer_L2(only_part=True)
            if alpha_classifier > 0:
                loss += alpha_classifier * loss_classifier
                loss += model.compute_classifier_L2()
        elif alpha_classifier > 0:
            loss += alpha_classifier * loss_classifier
            loss += model.compute_transformer_L2(only_part=True)
            loss += model.compute_classifier_L2()

    # define ops
    global_step = tf.train.get_global_step()
    lr = tf.train.exponential_decay(
        init_lr, global_step=global_step, decay_steps=decay_steps, decay_rate=decay_rate)
    optimizer = get_optimizer(optimizer)(lr)

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_op = optimizer.minimize(
            loss=loss, global_step=global_step)

    # define metrics
    eval_metric_ops = dict()
    eval_metric_ops['loss_dae'] = tf.metrics.mean(loss_dae)
    eval_metric_ops['loss_main'] = tf.metrics.mean(loss_image + loss_new + loss_loss)
    eval_metric_ops['loss_image'] = tf.metrics.mean(loss_image)
    eval_metric_ops['loss_new'] = tf.metrics.mean(loss_new)
    eval_metric_ops['loss_loss'] = tf.metrics.mean(loss_loss)
    eval_metric_ops['loss_bert_new'] = tf.metrics.mean(loss_bert_new)
    eval_metric_ops['loss_bert_loss'] = tf.metrics.mean(loss_bert_loss)
    eval_metric_ops['loss_classifier'] = tf.metrics.mean(loss_classifier)

    for label in range(NUM_LABEL):
        eval_metric_ops['auc_'+str(label)] = get_metric('auc')(labels[:,label],
                                                tf.sigmoid(logits_classifier[:,label]),
                                                num_thresholds=10000)

    return tf.estimator.EstimatorSpec(
        mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )
