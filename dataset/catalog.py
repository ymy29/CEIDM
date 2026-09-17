import os 

class DatasetCatalog:
    def __init__(self, ROOT):
        self.HicoDetHOI = {
            "target": "dataset.hico_dataset.HICODataset",
            "train_params":dict(
                dataset_path=os.path.join(ROOT,'hico_det_clip'),
                #max_images=4000,
            ),
        }

        self.VisualGenome = {
            "target": "dataset.hico_dataset.HICODataset",
            "train_params": dict(
                dataset_path=os.path.join(ROOT, 'vg_clip'),
                #max_images=4000, 
            ),
        }


