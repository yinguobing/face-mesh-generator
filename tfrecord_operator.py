"""This module provides an abstract class for TensorFlow Record file reading and
writting."""
from abc import ABC, abstractmethod

import tensorflow as tf

# The following functions can be used to convert a value to a type compatible
# with tf.Example.


def to_strings(value):
    """Returns strings from a tensor."""
    return tf.io.serialize_tensor(value)


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class RecordOperator(ABC):

    def __init__(self, filename):
        # Construct a reader if the user is trying to read the record file.
        self.dataset = None
        self._writer = None
        if tf.io.gfile.exists(filename):
            self.dataset = tf.data.TFRecordDataset(filename)
        else:
            # Construct a writer in case the user want to write something.
            self._writer = tf.io.TFRecordWriter(filename)

        # Set the feature description. This should be provided before trying to
        # parse the record file.
        self.set_feature_description()

    @abstractmethod
    def make_example(self):
        """Returns a tf.train.example from values to be saved."""
        pass

    def write_example(self, tf_example):
        """Create TFRecord example from a data sample."""
        if self._writer is None:
            raise IOError("Record file already exists.")
        else:
            self._writer.write(tf_example.SerializeToString())

    @abstractmethod
    def set_feature_description(self):
        """Set the feature_description to parse TFRecord file."""
        pass

    def parse_dataset(self):
        # Create a dictionary describing the features. This dict should be
        # consistent with the one used while generating the record file.
        if self.dataset is None:
            raise IOError("Dataset file not found.")

        def _parse_function(example_proto):
            # Parse the input tf.Example proto using the dictionary above.
            return tf.io.parse_single_example(example_proto, self.feature_description)

        parsed_dataset = self.dataset.map(_parse_function)
        return parsed_dataset
