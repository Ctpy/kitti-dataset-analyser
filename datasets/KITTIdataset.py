import glob
import os
from abc import ABC

import numpy as np
import open3d as o3d
import json
from datasets.dataset import Dataset
from tqdm import tqdm
from bbox.bbox import BoundingBox3D


class KITTIDataset(Dataset, ABC):

    def format(self, data):
        pass

    def __init__(self, root_path):
        super(KITTIDataset, self).__init__(root_path)

        self.velodynes = []
        self.labels = []
        self.calib = []
        self.num_nodes = 32

    def __len__(self):
        return len(self.velodynes)

    def load(self, split):
        velodyne_paths = glob.glob(os.path.join(self.root_path, split, "velodyne", "*"))
        self.velodynes.extend(velodyne_paths)

        label_paths = glob.glob(os.path.join(self.root_path, split, "label_2", "*"))
        self.labels.extend(label_paths)

        calib_paths = glob.glob(os.path.join(self.root_path, split, "calib", "*"))
        self.calib.extend(calib_paths)
        self.calib.sort()
        self.velodynes.sort()
        self.labels.sort()

    def load_frame(self, idx):
        points = np.fromfile(self.velodynes[idx], dtype=np.float32).reshape((-1, 4))[:, :3]
        # self.pcd.points = o3d.utility.Vector3dVector(points)
        return points

    def extract_bounding_boxes(self, idx):
        def get_R0_rect(calib_path):
            with open(calib_path, 'r') as f:
                tmp = f.read().split('\n')

            for idx in range(len(tmp)):
                if tmp[idx].split(' ')[0] == "R0_rect:":
                    R0 = tmp[idx]

            for idx in range(len(tmp)):
                if tmp[idx].split(' ')[0] == "Tr_velo_to_cam:":
                    Tr = tmp[idx]

            R0 = np.loadtxt(R0.split(' ')[1:]).reshape((3, 3))
            Tr = np.loadtxt(Tr.split(' ')[1:]).reshape((3, 4))
            return R0, np.append(Tr, np.array([[0, 0, 0, 1]]), axis=0)

        def get_labels(labels_path):
            labels = []
            classes = []
            with open(labels_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if line != '\n':
                        line = line.split()
                        classes.append(line[0])  # get class of each labels
                        line = line[8: 15]
                        line = [float(x) for x in line]
                        labels.append(line)
            labels = np.array(labels)
            return labels, classes

        def get_bbox(cx, cy, cz, h, w, l, rotation, R0_vect, Tr_vel_cam):
            rotation_matrix = np.array([
                [np.cos(rotation), 0, np.sin(rotation)],
                [0, 1, 0.0],
                [-np.sin(rotation), 0.0, np.cos(rotation)]])

            bb_box = o3d.geometry.OrientedBoundingBox(np.array([cx, cy - h / 2, cz]), rotation_matrix,
                                                      np.array([l, h, w]))
            # bb_box.color = np.array([0, 0, 1])
            bb_box.rotate(np.linalg.inv(R0_vect), center=(0, 0, 0))
            Tf_inv = np.linalg.inv(Tr_vel_cam)
            bb_box.rotate(Tf_inv[:3, :3], center=(0, 0, 0))
            bb_box.translate(Tf_inv[:3, 3], relative=True)
            return bb_box, rotation

        pcd_path = self.velodynes[idx]
        labels_path = self.labels[idx]
        calib_path = self.calib[idx]

        pcd = o3d.geometry.PointCloud()
        pcd_all = o3d.geometry.PointCloud()
        points = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)[:, :4]

        pcd_all.points = o3d.utility.Vector3dVector(points)
        pcd.points = o3d.utility.Vector3dVector(points)

        R0_rect, Tr_vel_cam = get_R0_rect(calib_path)
        labels, classes = get_labels(labels_path)

        bboxes = []
        rotations = []
        for i, label in enumerate(labels):
            h, w, l = label[0], label[1], label[2]
            cx, cy, cz = label[3], label[4], label[5]
            rotation = label[6]
            if classes[i] != 'DontCare':
                bbox, rot = get_bbox(cx, cy, cz, h, w, l, rotation, R0_rect, Tr_vel_cam)
                bboxes.append(bbox)
                rotations.append(rot)

        p_in_bb_idx_list = []
        centers_list = []
        filtered_bboxes = []
        filtered_rot = []
        for i in range(len(bboxes)):
            indices_p_in_bb = bboxes[i].get_point_indices_within_bounding_box(pcd.points)
            if classes[i] == 'Car':
                p_in_bb_idx_list.append(indices_p_in_bb)
                filtered_bboxes.append(bboxes[i])
                filtered_rot.append(rotations[i])

        points_in_bboxes = []

        for bbox_idx in p_in_bb_idx_list:
            points_in_bboxes.append(points[bbox_idx])

        return points_in_bboxes, centers_list, filtered_bboxes, filtered_rot

    def visualize(self, frame, with_labels=False):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.load_frame(frame))
        vis.add_geometry(pcd)
        if with_labels:
            # TODO: Add visualization for labels
            bbox, classes = self.extract_bounding_boxes(frame)
            for box in bbox:
                corner_points = box.bbox_center_to_corner()
                vis.add_geometry(box.create_bbox(corner_points.T))
        vis.run()
        vis.destroy_window()

    def inject(self, file, output, split):
        # create kitti folder structure
        os.makedirs(os.path.join(".", output, "kitti"), exist_ok=True)
        os.makedirs(os.path.join(".", output, "kitti", split, "label_2"), exist_ok=True)
        os.makedirs(os.path.join(".", output, "kitti", split, "velodyne"), exist_ok=True)
        # create files
        good_car = json.load(open(file, 'r'))
        counter = 0

        def getCar(i, cars):
            return cars[f"{i % 1416}"]

        for velodyne, label_path, calib_path in tqdm(zip(self.velodynes, self.labels, self.calib), total=len(self.velodynes)):
            def get_R0_rect(calib_path):
                with open(calib_path, 'r') as f:
                    tmp = f.read().split('\n')

                for idx in range(len(tmp)):
                    if tmp[idx].split(' ')[0] == "R0_rect:":
                        R0 = tmp[idx]

                for idx in range(len(tmp)):
                    if tmp[idx].split(' ')[0] == "Tr_velo_to_cam:":
                        Tr = tmp[idx]

                R0 = np.loadtxt(R0.split(' ')[1:]).reshape((3, 3))
                Tr = np.loadtxt(Tr.split(' ')[1:]).reshape((3, 4))
                return R0, np.append(Tr, np.array([[0, 0, 0, 1]]), axis=0)

            def get_labels(labels_path):
                labels = []
                classes = []
                with open(labels_path, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        if line != '\n':
                            line = line.split()
                            classes.append(line[0])  # get class of each labels
                            line = line[1: 15]
                            line = [float(x) for x in line]
                            labels.append(line)
                labels = np.array(labels)
                return labels, classes

            def get_bbox(cx, cy, cz, h, w, l, rotation, R0_vect, Tr_vel_cam):
                rotation_matrix = np.array([
                    [np.cos(rotation), 0, np.sin(rotation)],
                    [0, 1, 0.0],
                    [-np.sin(rotation), 0.0, np.cos(rotation)]])

                bb_box = o3d.geometry.OrientedBoundingBox(np.array([cx, cy - h / 2, cz]), rotation_matrix,
                                                          np.array([l, h, w]))
                # bb_box.color = np.array([0, 0, 1])
                bb_box.rotate(np.linalg.inv(R0_vect), center=(0, 0, 0))
                Tf_inv = np.linalg.inv(Tr_vel_cam)
                bb_box.rotate(Tf_inv[:3, :3], center=(0, 0, 0))
                bb_box.translate(Tf_inv[:3, 3], relative=True)
                return bb_box, rotation


            pcd = o3d.geometry.PointCloud()
            pcd_all = o3d.geometry.PointCloud()
            points = np.fromfile(velodyne, dtype=np.float32).reshape(-1, 4)[:, :3]
            r = np.fromfile(velodyne, dtype=np.float32).reshape(-1, 4)[:, 3]
            pcd_all.points = o3d.utility.Vector3dVector(points)
            pcd.points = o3d.utility.Vector3dVector(points)

            R0_rect, Tr_vel_cam = get_R0_rect(calib_path)
            labels, classes = get_labels(label_path)
            with open(os.path.join(".", output, "kitti", split, "label_2", os.path.basename(label_path)), 'w') as f:
                for i, label in enumerate(labels):
                    if classes[i] == 'Car':
                        l, w, h = getCar(counter, good_car)["dim"]
                        points_box = np.array(getCar(counter, good_car)["points"])
                        # r_p = np.zeros((points_box.shape[0], 1))
                        # points_box = np.append(points_box, r_p, axis=1)
                        counter += 1
                        cx, cy, cz = label[3 + 7], label[4 + 7], label[5 + 7]
                        rotation = label[6 + 7]
                        bbox, rot = get_bbox(cx, cy, cz, h, w, l, rotation, R0_rect, Tr_vel_cam)
                        indices_p_in_bb = bbox.get_point_indices_within_bounding_box(pcd.points)
                        np.delete(points, indices_p_in_bb)
                        np.delete(r, indices_p_in_bb)
                        points = np.vstack((points, points_box + np.array([bbox.center[0], bbox.center[1], bbox.center[2]])))
                        points = points.astype(np.float32)
                        # vis = o3d.visualization.Visualizer()
                        # vis.create_window()
                        # vis.get_render_option().background_color = [0.1, 0.1, 0.1]
                        # vis.get_render_option().point_size = 1
                        # vis.get_render_option().show_coordinate_frame = True
                        # pcd = o3d.geometry.PointCloud()
                        # pcd.points = o3d.utility.Vector3dVector(points)
                        # vis.add_geometry(pcd)
                        # #pcd_dd = o3d.geometry.PointCloud()
                        # #pcd_dd.points = o3d.utility.Vector3dVector(points_box + bbox.center)
                        # #pcd_dd.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0]]))
                        # vis.add_geometry(pcd)
                        # #vis.add_geometry(pcd_dd)
                        # bbox = o3d.geometry.OrientedBoundingBox(bbox.center, bbox.R, bbox.extent)
                        # bbox.color = np.array([1, 0, 0])
                        # vis.add_geometry(bbox)
                        # vis.get_view_control().set_zoom(0.05)
                        # vis.get_view_control().set_front([-0.940, 0.096, 0.327])
                        # vis.get_view_control().set_lookat([17.053, 0.544, -2.165])
                        # vis.get_view_control().set_up([0.327, -0.014, 0.945])
                        # vis.run()
                        # vis.destroy_window()
                        f.write(f"{classes[i]} " + ' '.join(map(str, label)) + '\n')

                f.close()
                points.tofile(os.path.join(".", output, "kitti", split, "velodyne", os.path.basename(velodyne)))
                new_points = np.fromfile(os.path.join(".", output, "kitti", split, "velodyne", os.path.basename(velodyne)), dtype=np.float32).reshape(-1, 3)
                assert points.shape == new_points.shape, f"Shape 1 {points.shape} does not match shape 2 {new_points.shape}"


if __name__ == '__main__':
    kitti = KITTIDataset("/mnt/wwn-0x5000c500ecc16c3d-part1/kitti")
    kitti.load("training")
    kitti.inject("../distributions/good_cars.json", "/home/tung/Desktop", "training")
