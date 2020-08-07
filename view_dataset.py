import cv2
import numpy as np
import tensorflow as tf

from tfrecord_operator import RecordReader
from fmd.mark_dataset.util import draw_marks

if __name__ == "__main__":
    dataset = RecordReader(
        "/home/robin/Desktop/face-mesh-generator/tfrecord/300w.record")

    for sample in dataset.parse_dataset():

        image_decoded = tf.image.decode_image(sample['image/encoded']).numpy()
        height = sample['image/height'].numpy()
        width = sample['image/width'].numpy()
        depth = sample['image/depth'].numpy()
        filename = sample['image/filename'].numpy()
        marks = sample['label/mesh'].numpy()

        print(filename, width, height, depth)

        # Use OpenCV to preview the image.
        image = np.array(image_decoded, np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw the landmark on image
        landmark = np.reshape(marks, (-1, 3)) * width
        draw_marks(image, landmark, 1)

        # Show the result
        cv2.imshow("image", cv2.resize(image, (512, 512)))
        if cv2.waitKey() == 27:
            break
