import glob
import json
import os.path
import numpy as np
import open3d as o3d
from dataloader.dataloader import DataLoader
from datasets.KITTIdataset import KITTIDataset
from datasets.A9dataset import A9Dataset
from tqdm import tqdm


def main():
    dataset = A9Dataset("/mnt/wwn-0x5000c500ecc1920b-part1/a9")
    dataset.load("")
    # dataloader = DataLoader(dataset, ['Car'])
    # dataloader.load("training", 6000)

    # read output folder

    # read json

    good_cars = dict()
    bad_cars = dict()
    good_counter = 0
    bad_counter = 0
    max = 50
    counter = 0
    for j in tqdm(range(8000)):
        try:
            if not os.path.exists(f"output/file_{j}.json"):
                bbox = dataset.extract_bounding_boxes(j)
                for i, box in enumerate(bbox):
                    # load bbox
                    center = box.center
                    dim = box.extent
                    pcd = dataset.load_frame(i)
                    points = box.get_point_indices_within_bounding_box(pcd.points)
                    points = np.asarray(pcd.points)[points]
                    rot = np.arctan2(box.R[1, 0], box.R[0, 0])

                    # normalize bbox

                    def rotation_z(heading):
                        rotation_matrix_z = np.asarray([[np.cos(heading), -np.sin(heading), 0],
                                                        [np.sin(heading), np.cos(heading), 0],
                                                        [0, 0, 1]])
                        return rotation_matrix_z
                    normalized_points = (points - center) @ rotation_z(rot)
                    good_cars[f"{i}"] = {
                        'points': normalized_points.tolist(),
                        'dim': dim.tolist()
                    }
                with open(f"output/file_{j}.json", "w") as f:
                    json.dump(good_cars, f)
        except IndexError:
            raise IndexError
            continue
            # visualize bbox
        #     vis = o3d.visualization.Visualizer()
        #     vis.create_window()
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(normalized_points)
        #     vis.add_geometry(pcd)
        #     vis.add_geometry(o3d.geometry.OrientedBoundingBox(np.array([0, 0, 0]), np.identity(3), dim))
        #     vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
        #     vis.run()
        #     vis.capture_screen_image(f"output_image/{counter}.png")
        #     vis.destroy_window()
        #     while (True):
        #         try:
        #             val = int(input("Enter 0 - for good and 1 - for bad"))
        #             break
        #         except ValueError:
        #             print("enter a number")
        #
        #     if val == 0:
        #         good_cars[f"{good_counter}"] = {'points': normalized_points.tolist()}
        #         vis.capture_screen_image(f"output_image/good_cars/{counter}.png")
        #         good_counter += 1
        #     else:
        #         bad_cars[f"{bad_counter}"] = {'points': normalized_points.tolist()}
        #         vis.capture_screen_image(f"output_image/bad_cars/{counter}.png")
        #         bad_counter += 1
        #     counter += 1
        # if counter > max:
        #     break
    # with open("output/goodcars_a9.json", "w") as f:
    #     json.dump(good_cars, f)
    # with open("output/badcars_a91.json", "w") as f:
    #     json.dump(bad_cars, f)
        # save good and bad cars by input


if __name__ == '__main__':
    main()
