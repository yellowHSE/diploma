from datasets_python.root_dataset import RootDataset


class KSDD2_PerlinDataset(RootDataset):
    ds_name = "ksdd2"

    resize = (224, 672)
    cropsize = 224


class BSDATA_PerlinDataset(RootDataset):
    ds_name = "bsdata"

    resize = (800, 320)
    cropsize = 320


class DAGM_PerlinDataset(RootDataset):
    ds_name = "dagm"

    resize = (512, 512)
    cropsize = 320


class SoftgelPerlinDataset(RootDataset):
    ds_name = "softgel"

    resize = (160, 160)
    cropsize = 128
