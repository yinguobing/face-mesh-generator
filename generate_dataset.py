import fmd


# Set the dataset directory you are going to use.
DS300W_DIR = "/home/robin/data/facial-marks/300W"
DS300VW_DIR = "/home/robin/data/facial-marks/300VW_Dataset_2015_12_14"
AFW_DIR = "/home/robin/data/facial-marks/afw"
HELEN_DIR = "/home/robin/data/facial-marks/helen"
IBUG_DIR = "/home/robin/data/facial-marks/ibug"
LFPW_DIR = "/home/robin/data/facial-marks/lfpw"
WFLW_DIR = "/home/robin/data/facial-marks/wflw/WFLW_images"
AFLW2000_3D_DIR = "/home/robin/data/facial-marks/3DDFA/AFLW2000-3D"

if __name__ == "__main__":
    # Construct the datasets.

    # 300W
    ds_300w = fmd.ds300w.DS300W("300w")
    ds_300w.populate_dataset(DS300W_DIR)

    # 300VW
    ds_300vw = fmd.ds300vw.DS300VW("300vw")
    ds_300vw.populate_dataset(DS300VW_DIR)

    # AFW
    ds_afw = fmd.afw.AFW("afw")
    ds_afw.populate_dataset(AFW_DIR)

    # HELEN
    ds_helen = fmd.helen.HELEN("helen")
    ds_helen.populate_dataset(HELEN_DIR)

    # IBUG
    ds_ibug = fmd.ibug.IBUG("ibug")
    ds_ibug.populate_dataset(IBUG_DIR)

    # LFPW
    ds_lfpw = fmd.lfpw.LFPW("lfpw")
    ds_lfpw.populate_dataset(LFPW_DIR)

    # WFLW
    ds_wflw = fmd.wflw.WFLW("wflw")
    ds_wflw.populate_dataset(WFLW_DIR)

    # AFLW2000-3D
    ds_aflw2k3d = fmd.AFLW2000_3D("AFLW2000_3D")
    ds_aflw2k3d.populate_dataset(AFLW2000_3D_DIR)

    # How many samples do we have?
    print("Total samples: {}".format(sum(ds.meta["num_samples"] for ds in [
        ds_300vw, ds_300w, ds_aflw2k3d, ds_afw, ds_helen, ds_ibug, ds_lfpw, ds_wflw
    ])))
