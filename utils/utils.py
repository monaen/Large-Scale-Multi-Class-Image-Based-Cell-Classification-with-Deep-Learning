import numpy as np

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


def reformat_label(labels, num_labels):
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return labels