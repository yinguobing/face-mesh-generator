import argparse
import logging
import traceback

import cv2
import numpy as np
from tqdm import tqdm

import fmd
from mark_guardians import check_mark_location
from mark_operator import MarkOperator

# Get the command line argument.
parser = argparse.ArgumentParser()
parser.add_argument("--loglevel", type=str, default="info",
                    help="The logging level.")
args = parser.parse_args()

# Data processing is a long time job, which makes logging a essential part.
log_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
numeric_level = getattr(logging, args.loglevel.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: {}'.format(args.loglevel))

# Setup logs in the console.
console_hdlr = logging.StreamHandler()
console_hdlr.setFormatter(log_formatter)

# Setup logs in the log file.
file_hdlr = logging.FileHandler('data_generation.log')
file_hdlr.setFormatter(log_formatter)

# Setup the logger.
logger = logging.getLogger(__name__)
logger.addHandler(console_hdlr)
logger.addHandler(file_hdlr)
logger.setLevel(numeric_level)


def process(dataset, index_start_from=0):
    """Process the dataset as required, including rotating the face, crop the
    face area.

    Args:
        dataset: a MarkDataset object.
        start_from: the sample index to start from.

    Returns:
        None
    """
    logger.info("Starting to process dataset: {}".format(dataset.meta['name']))

    # Keep a record of the current location.
    current_sample_index = -1

    # Count the samples considered invalid.
    num_invalid_samples = 0

    # Construct a mark operator to transform the marks.
    mo = MarkOperator()

    # Some dataset contains enormous samples, in which some may be corrupted
    # and cause processing error. Catch these errors to avoid restarting over
    # from the start.
    try:
        # Enumerate all the samples in dataset.
        for sample in tqdm(dataset):
            # In case the job is interrupted, we can start from somwhere in
            # between rather than starting over from the very begining.
            current_sample_index += 1
            if current_sample_index < index_start_from:
                continue

            # Safety check, invalid samples will be discarded.
            image = sample.read_image()
            img_height, img_width, _ = image.shape
            marks = sample.marks

            # Security check passed, the image is ready for transformation. Here
            # the face area is our region of interest, and will be cropped.
            fmd.mark_dataset.util.draw_marks(image, marks)

            # First, move the face to the center.
            face_center = mo.get_center(marks)[:2]
            translation_mat = np.array([[1, 0, img_width / 2 - face_center[0]],
                                        [0, 1, img_height / 2 - face_center[1]]])
            translated_image = cv2.warpAffine(
                image, translation_mat, (img_width, img_height))

            # Second, align the face. This happens in the 2D space.
            key_marks = sample.get_key_marks()[:, :2]
            vector_eye = (key_marks[3] - key_marks[0])
            degrees = mo.get_angle(vector_eye, np.array([100, 0]))
            rotation_mat = cv2.getRotationMatrix2D(
                ((img_width-1)/2.0, (img_height-1)/2.0), -degrees, 1)
            image_rotated = cv2.warpAffine(
                translated_image, rotation_mat, (img_width, img_height))

            # Third, try to crop the face area out.
            x_min, y_min, _ = np.amin(marks, 0)
            x_max, y_max, _ = np.amax(marks, 0)
            scale = 1.5
            side_length = max((x_max - x_min, y_max - y_min)) * scale
            start_x = int(img_width / 2 - side_length / 2)
            start_y = int(img_height / 2 - side_length / 2)
            end_x = int(img_width / 2 + side_length / 2)
            end_y = int(img_height / 2 + side_length / 2)

            # In case the new bbox is out of image bounding.
            border_width = 0
            border_x = min(start_x, start_y)
            border_y = max(end_x - img_width, end_y - img_height)
            if border_x < 0 or border_y > 0:
                border_width = max(abs(border_x), abs(border_y))
                start_x += border_width
                start_y += border_width
                end_x += border_width
                end_y += border_width
                image_with_border = cv2.copyMakeBorder(image, border_width,
                                                       border_width,
                                                       border_width,
                                                       border_width,
                                                       cv2.BORDER_CONSTANT,
                                                       value=[0, 0, 0])
                image_cropped = image_with_border[start_y:end_y,
                                                  start_x:end_x]
            else:
                image_cropped = image_rotated[start_y:end_y, start_x:end_x]

            # Last, resize the face area. I noticed Google is using 192px.
            image_resized = cv2.resize(image_cropped, (192, 192))

            cv2.imshow("Preview", image_resized)
            if cv2.waitKey(30) == 27:
                break
    except Exception:
        logger.error(
            "Error {}. sample index: {}".format(traceback.format_exc(), current_sample_index))
    finally:
        # Summary
        logger.info("Dataset done. Processed samples: {}, invalid samples: {}".format(
            current_sample_index+1, num_invalid_samples))


if __name__ == "__main__":
    # Set the dataset directory you are going to use.
    ds300w_dir = "/home/robin/data/facial-marks/300W"
    ds300vw_dir = "/home/robin/data/facial-marks/300VW_Dataset_2015_12_14"
    afw_dir = "/home/robin/data/facial-marks/afw"
    helen_dir = "/home/robin/data/facial-marks/helen"
    ibug_dir = "/home/robin/data/facial-marks/ibug"
    lfpw_dir = "/home/robin/data/facial-marks/lfpw"
    wflw_dir = "/home/robin/data/facial-marks/wflw/WFLW_images"
    aflw2000_3d_dir = "/home/robin/data/facial-marks/3DDFA/AFLW2000-3D"

    # Construct the datasets.

    # 300W
    ds_300w = fmd.ds300w.DS300W("300w")
    ds_300w.populate_dataset(ds300w_dir)

    process(ds_300w)

    # # 300VW
    # ds_300vw = fmd.ds300vw.DS300VW("300vw")
    # ds_300vw.populate_dataset(ds300vw_dir)

    # # AFW
    # ds_afw = fmd.afw.AFW("afw")
    # ds_afw.populate_dataset(afw_dir)

    # # HELEN
    # ds_helen = fmd.helen.HELEN("helen")
    # ds_helen.populate_dataset(helen_dir)

    # # IBUG
    # ds_ibug = fmd.ibug.IBUG("ibug")
    # ds_ibug.populate_dataset(ibug_dir)

    # # LFPW
    # ds_lfpw = fmd.lfpw.LFPW("lfpw")
    # ds_lfpw.populate_dataset(lfpw_dir)

    # # WFLW
    # ds_wflw = fmd.wflw.WFLW("wflw")
    # ds_wflw.populate_dataset(wflw_dir)

    # # AFLW2000-3D
    # ds_aflw2k3d = fmd.AFLW2000_3D("AFLW2000_3D")
    # ds_aflw2k3d.populate_dataset(aflw2000_3d_dir)

    # datasets = [ds_300vw, ds_300w, ds_aflw2k3d,
    #             ds_afw, ds_helen, ds_ibug, ds_lfpw, ds_wflw]

    # # How many samples do we have?
    # print("Total samples: {}".format(
    #     sum(ds.meta["num_samples"] for ds in datasets)))

    # # Process all the data.
    # for ds in datasets:
    #     process(ds)
