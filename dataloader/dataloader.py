import json
from dataclasses import dataclass
from datasets.dataset import Dataset
from tqdm import tqdm
import open3d as o3d
import numpy as np

@dataclass
class DataLoader:

    dataset: Dataset
    classes: list[str]

    def __init__(self, dataset: Dataset, classes: list[str]):
        self.dataset = dataset
        self.classes = classes

    def load(self, split, batch_size):
        self.dataset.load(split)
        for batch in tqdm(range(batch_size)):
            points_in_bboxes, centers_list, bboxes, rot = self.dataset.extract_bounding_boxes(batch)
            dict_bbox = dict()
            for i in range(len(points_in_bboxes)):
                dict_bbox[f'{i}'] = {
                    'points': points_in_bboxes[i].tolist(),
                    'location': bboxes[i].center.tolist(),
                    'dim': bboxes[i].extent.tolist(),
                    'rot': bboxes[i].R.tolist(),
                    'head': rot[i].tolist(),
                    'num_points': len(points_in_bboxes[i].tolist())
                }
                # vis = o3d.visualization.Visualizer()
                # vis.create_window()
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(points_in_bboxes[i])
                # vis.add_geometry(pcd)
                # vis.add_geometry(bboxes[i])
                # vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
                # vis.run()
                # vis.destroy_window()
            with open(f"output/{batch}.json", "w") as f:
                json.dump(dict_bbox, f)

    def pickle(self, split):
        pass # TODO: extract data and pickle