import tensorflow as tf

from tfrecord_operator import (RecordOperator, bytes_feature, float_feature,
                               int64_feature, to_strings)


class MeshRecordOperator(RecordOperator):

    def make_example(self, image_np, mesh, score, example_name):
        # Encode the image.
        image_encoded = tf.image.encode_jpeg(image_np)

        # Get required features ready.
        image_shape = image_np.shape

        # Flat the mesh.
        mesh = to_strings(mesh)

        # After getting all the features, time to generate a TensorFlow example.
        feature = {
            'image/height': int64_feature(image_shape[0]),
            'image/width': int64_feature(image_shape[1]),
            'image/depth': int64_feature(image_shape[2]),
            'image/filename': bytes_feature(example_name.encode('utf8')),
            'image/encoded': bytes_feature(image_encoded),
            'label/mesh': bytes_feature(mesh),
            'label/score': float_feature(score)
        }

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=feature))

        return tf_example

    def set_feature_description(self):
        self.feature_description = {
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/depth': tf.io.FixedLenFeature([], tf.int64),
            'image/filename': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'label/mesh': tf.io.FixedLenFeature([], tf.string),
            'label/score': tf.io.FixedLenFeature([], tf.float32),
        }
