import tensorflow as tf


# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float_feature_list(value):
    """Returns a float_list from a numpy array."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class RecordWriter(object):

    def __init__(self, filename):
        self._writer = tf.io.TFRecordWriter(filename)

    def write_example(self, image_np, mesh, score, example_name):
        """create TFRecord example from a data sample."""
        # Encode the image.
        image_encoded = tf.image.encode_jpeg(image_np)

        # Get required features ready.
        image_shape = image_np.shape

        # Flat the mesh.
        mesh = mesh.flatten()

        # After getting all the features, time to generate a TensorFlow example.
        feature = {
            'image/height': _int64_feature(image_shape[0]),
            'image/width': _int64_feature(image_shape[1]),
            'image/depth': _int64_feature(image_shape[2]),
            'image/filename': _bytes_feature(example_name.encode('utf8')),
            'image/encoded': _bytes_feature(image_encoded),
            'label/mesh': _float_feature_list(mesh),
            'label/score': _float_feature(score)
        }
        tf_example = tf.train.Example(
            features=tf.train.Features(feature=feature))

        self._writer.write(tf_example.SerializeToString())


class RecordReader(object):

    def __init__(self, record_file):
        self.dataset = tf.data.TFRecordDataset(record_file)

    def parse_dataset(self):
        # Create a dictionary describing the features. This dict should be
        # consistent with the one used while generating the record file.
        feature_description = {
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/depth': tf.io.FixedLenFeature([], tf.int64),
            'image/filename': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'label/mesh': tf.io.FixedLenFeature([468*3], tf.float32),
            'label/score': tf.io.FixedLenFeature([], tf.float32),
        }

        def _parse_function(example_proto):
            # Parse the input tf.Example proto using the dictionary above.
            return tf.io.parse_single_example(example_proto, feature_description)

        parsed_dataset = self.dataset.map(_parse_function)
        return parsed_dataset
