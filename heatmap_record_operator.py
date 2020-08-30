import tensorflow as tf

from tfrecord_operator import (RecordOperator, bytes_feature, float_feature,
                               int64_feature, to_strings)


class HeatmapRecordOperator(RecordOperator):

    def make_example(self, image_np, marks, heatmaps, example_name):
        # Encode the image.
        image_encoded = tf.image.encode_jpeg(image_np)

        # Get required features ready.
        image_shape = image_np.shape
        heatmap_size = heatmaps.shape
        n_marks = marks.shape[0]
        marks = to_strings(marks)
        heatmaps = to_strings(heatmaps)

        # After getting all the features, time to generate a TensorFlow example.
        feature = {
            'image/height': int64_feature(image_shape[0]),
            'image/width': int64_feature(image_shape[1]),
            'image/depth': int64_feature(image_shape[2]),
            'image/filename': bytes_feature(example_name.encode('utf8')),
            'image/encoded': bytes_feature(image_encoded),
            "label/marks": bytes_feature(marks),
            "label/n_marks": int64_feature(n_marks),
            'heatmap/map': bytes_feature(heatmaps),
            'heatmap/height': int64_feature(heatmap_size[0]),
            'heatmap/width': int64_feature(heatmap_size[1]),
            'heatmap/depth': int64_feature(heatmap_size[2])
        }

        tf_example=tf.train.Example(
            features=tf.train.Features(feature=feature))

        return tf_example

    def set_feature_description(self):
        self.feature_description={
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/depth': tf.io.FixedLenFeature([], tf.int64),
            'image/filename': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'label/marks': tf.io.FixedLenFeature([], tf.string),
            'label/n_marks': tf.io.FixedLenFeature([], tf.int64),
            'heatmap/map': tf.io.FixedLenFeature([], tf.string),
            'heatmap/height': tf.io.FixedLenFeature([], tf.int64),
            'heatmap/width': tf.io.FixedLenFeature([], tf.int64),
            'heatmap/depth': tf.io.FixedLenFeature([], tf.int64)
        }
