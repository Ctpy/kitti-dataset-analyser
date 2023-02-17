import glob
import os
import numpy as np
import json
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from datasets.dataset import Dataset


class A9Dataset(Dataset):

    def __init__(self, root_path):
        super().__init__(root_path)

        self.point_clouds = []
        self.labels = []
        self.basename = []

    def load(self, split):
        point_clouds = glob.glob(os.path.join(self.root_path, "registered_3_lidars", "s110_lidar_ouster_north", '*'))
        self.point_clouds.extend(point_clouds)
        for file in self.point_clouds:
            basename = os.path.basename(file).replace(".pcd", "")
            self.basename.append(basename)
        for split_type in ["test", "train", "val"]:
            label_paths = glob.glob(os.path.join(self.root_path, "r1_dataset", "point_cloud_split", split_type, "labels", '*'))
            for label in label_paths:
                if os.path.basename(label).replace(".json", "") in self.basename:
                    self.labels.append(label)

    def extract_bounding_boxes(self, idx):
        def find_label(basename):
            for label in self.labels:
                if basename in label:
                    return label

        def rotation_z(heading):
            rotation_matrix_z = np.asarray([[np.cos(heading), -np.sin(heading), 0],
                            [np.sin(heading), np.cos(heading), 0],
                            [0, 0, 1]])
            return rotation_matrix_z

        def object_data_2_o3d_bbox(obj_data):
            heading = R.from_quat(obj_data[3:7]).as_euler('zyx', degrees=False)[0]
            xyz = obj_data[0:3]
            dim = obj_data[7:]  # l h w
            bbox = o3d.geometry.OrientedBoundingBox(xyz, rotation_z(heading), dim)
            bbox.color = np.array([1, 0, 0])
            return bbox

        label = json.load(open(find_label(self.basename[idx])))
        frames = label["openlabel"]["frames"]
        objects = []
        for key in frames:
            for obj in frames[key]["objects"]:
                if frames[key]["objects"][obj]["object_data"]["type"] == 'CAR':
                    objects.append(frames[key]["objects"][obj]["object_data"]["cuboid"]['val'])
        bbox = []
        for obj in objects:
            bbox.append(object_data_2_o3d_bbox(obj))
        return bbox

    def format(self, data):
        pass

    def load_frame(self, idx):
        points = o3d.io.read_point_cloud(self.point_clouds[idx])
        return points

    def visualize(self, frame, with_labels=False):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().background_color = [0.1, 0.1, 0.1]
        vis.get_render_option().point_size = 1
        vis.get_render_option().show_coordinate_frame = True
        pcd = self.load_frame(frame)
        vis.add_geometry(pcd)
        if with_labels:
            bbox = self.extract_bounding_boxes(frame)
            for box in bbox:
                vis.add_geometry(box)
        vis.get_view_control().set_zoom(0.05)
        vis.get_view_control().set_front([-0.940, 0.096, 0.327])
        vis.get_view_control().set_lookat([17.053, 0.544, -2.165])
        vis.get_view_control().set_up([0.327, -0.014, 0.945])
        vis.run()
        vis.destroy_window()

    def __len__(self):
        pass


if __name__ == '__main__':
    a9 = A9Dataset("/mnt/wwn-0x5000c500ecc1920b-part1/a9")
    a9.load("")
    a9.visualize(9, with_labels=True)
