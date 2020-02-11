import tensorflow as tf


def kullback_leibler_divergence_loss(labels, logits, a=100, label_smoothing=0.0):
    if label_smoothing > 0:
        num_classes = tf.cast(tf.shape(labels)[1], logits.dtype)
        smooth_positives = 1.0 - label_smoothing
        smooth_negatives = label_smoothing / num_classes
        labels = labels * smooth_positives + smooth_negatives

    labels = labels / (tf.reduce_sum(labels, axis=1, keepdims=True) + 1e-5)
    logits = tf.nn.softmax(logits, axis=1)

    labels = tf.clip_by_value(labels, 1e-7, 1)
    logits = tf.clip_by_value(logits, 1e-7, 1)

    return tf.reduce_mean(tf.reduce_sum(labels * tf.log(labels / logits), axis=1))


def logistic_loss(labels, logits, a=100, label_smoothing=0.0):
    weights = labels * (a - 1) + 1
    loss = tf.losses.sigmoid_cross_entropy(
        labels, logits, weights=weights, label_smoothing=label_smoothing)
    return loss


def softmax_loss(onehot_labels, logits, weights=1.0, label_smoothing=0.0):
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels, logits, weights=weights, label_smoothing=label_smoothing)
    return loss


def multinomial_loss(labels, logits, a=100, label_smoothing=0.0):
    if label_smoothing > 0:
        num_classes = tf.cast(tf.shape(labels)[1], logits.dtype)
        smooth_positives = 1.0 - label_smoothing
        smooth_negatives = label_smoothing / num_classes
        labels_smoothed = labels * smooth_positives + smooth_negatives
    else:
        labels_smoothed = labels

    log_softmax_var = tf.nn.log_softmax(logits, axis=1)

    loss = -tf.reduce_mean(tf.reduce_sum(
        log_softmax_var * labels_smoothed, axis=1) / (tf.reduce_sum(labels, axis=1) +
                                                      1e-5))
    return loss
