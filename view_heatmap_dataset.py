import cv2
import numpy as np
import tensorflow as tf

from heatmap_record_operator import HeatmapRecordOperator
from fmd.mark_dataset.util import draw_marks

if __name__ == "__main__":
    dataset = HeatmapRecordOperator(
        "/home/robin/data/facial-marks/wflw/tfrecord/wflw_train.record")

    for sample in dataset.parse_dataset():

        image_decoded = tf.image.decode_image(sample['image/encoded']).numpy()
        height = sample['image/height'].numpy()
        width = sample['image/width'].numpy()
        depth = sample['image/depth'].numpy()
        filename = sample['image/filename'].numpy()
        marks = tf.io.parse_tensor(sample['label/marks'], tf.double).numpy()
        n_marks = sample['label/n_marks'].numpy()
        heatmaps = tf.io.parse_tensor(
            sample['heatmap/map'], tf.double).numpy()

        print(filename, width, height, depth, heatmaps.shape)

        # Use OpenCV to preview the image.
        image = np.array(image_decoded, np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw the landmark on image
        draw_marks(image, marks*width, 1)

        # View the heatmaps as a whole..
        heatmaps_merged = np.sum(heatmaps, axis=0) * 0.1

        # ..or individually.
        heatmap_idvs = np.hstack(heatmaps[:8])
        for row in range(1, 12, 1):
            heatmap_idvs = np.vstack(
                [heatmap_idvs, np.hstack(heatmaps[row:row+8])])

        # Show the result
        cv2.imshow("image", cv2.resize(image, (512, 512)))
        cv2.imshow("Heatmap", cv2.resize(
            heatmaps_merged, (512, 512), interpolation=cv2.INTER_AREA))
        cv2.imshow("Heatmap_idvs", heatmap_idvs)
        if cv2.waitKey() == 27:
            break
