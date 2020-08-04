import logging

from tqdm import tqdm

import fmd
from mark_guardians import check_mark_location

# Data processing is a long time job, which makes logging a essential part.
log_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Setup logs in the console.
console_hdlr = logging.StreamHandler()
console_hdlr.setLevel(logging.INFO)
console_hdlr.setFormatter(log_formatter)

# Setup logs in the log file.
file_hdlr = logging.FileHandler('data_generation.log')
file_hdlr.setLevel(logging.INFO)
file_hdlr.setFormatter(log_formatter)

# Setup the logger.
logger = logging.getLogger(__name__)
logger.addHandler(console_hdlr)
logger.addHandler(file_hdlr)


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

    # Some dataset contains enormous samples, in which some may be corrupted
    # and cause processing error. Catch these errors to avoid restarting over
    # from the start.
    try:
        # Enumerate all the samples in dataset.
        for sample in tqdm(dataset):
            current_sample_index += 1
            if current_sample_index < index_start_from:
                # Skip samples.
                continue

            image = sample.read_image()
            marks = sample.marks

            # Safety check, invalid samples will be discarded.

            # Only color images are valid.
            if len(image.shape) != 3:
                num_invalid_samples += 1
                logger.warning("Not a color image: {}, index: {}".format(
                    sample.image_file, current_sample_index))
                continue

            # Check mark locations. Make sure they are all in image.
            img_height, img_width, _ = image.shape
            if not check_mark_location(img_height, img_width, marks):
                num_invalid_samples += 1
                logger.warning("Mark outside of image: {}, index: {}".format(
                    sample.image_file, current_sample_index))
                continue
    except:
        logger.critical(
            "Unexpected error. sample index: {}".format(current_sample_index))

    # Summary
    logger.info("Dataset done. Processed samples: {}, invalid samples: {}".format(
        current_sample_index, num_invalid_samples))


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

    # 300VW
    ds_300vw = fmd.ds300vw.DS300VW("300vw")
    ds_300vw.populate_dataset(ds300vw_dir)

    # AFW
    ds_afw = fmd.afw.AFW("afw")
    ds_afw.populate_dataset(afw_dir)

    # HELEN
    ds_helen = fmd.helen.HELEN("helen")
    ds_helen.populate_dataset(helen_dir)

    # IBUG
    ds_ibug = fmd.ibug.IBUG("ibug")
    ds_ibug.populate_dataset(ibug_dir)

    # LFPW
    ds_lfpw = fmd.lfpw.LFPW("lfpw")
    ds_lfpw.populate_dataset(lfpw_dir)

    # WFLW
    ds_wflw = fmd.wflw.WFLW("wflw")
    ds_wflw.populate_dataset(wflw_dir)

    # AFLW2000-3D
    ds_aflw2k3d = fmd.AFLW2000_3D("AFLW2000_3D")
    ds_aflw2k3d.populate_dataset(aflw2000_3d_dir)

    datasets = [ds_300vw, ds_300w, ds_aflw2k3d,
                ds_afw, ds_helen, ds_ibug, ds_lfpw, ds_wflw]

    # How many samples do we have?
    print("Total samples: {}".format(
        sum(ds.meta["num_samples"] for ds in datasets)))

    # Process all the data.
    for ds in datasets:
        process(ds)
