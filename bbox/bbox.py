import numpy as np
import open3d as o3d


class BoundingBox3D:
    """
    w l h
    """

    def __init__(self, *box_dims):
        self.points = []
        self.num_points = 0
        self.corner_points = None
        if len(box_dims) == 3:
            self.__init_3(box_dims)
        elif len(box_dims) == 7:
            self.__init_7(box_dims)

    def __init_7(self, args):
        self.location = np.asarray([args[0], args[1], args[2]], dtype=np.float32)
        self.dimension = np.asarray([args[3], args[4], args[5]], dtype=np.float32)
        self.rotation = np.float(args[6])

    def __init_3(self, args):
        self.location = np.asarray(args[0], dtype=np.float32)
        self.dimension = np.asarray(args[1], dtype=np.float32)
        self.rotation = np.float(args[2])

    def __repr__(self):
        return f"Location: {self.location} | Dimension: {self.dimension} | Rotation: {self.rotation} | num_points: {len(self.points)}\n"

    def get_distance_to_center(self, frame):
        pass

    def get_num_points_in_box(self, frame):
        for point in frame:
            if self.check_point_in_bbox(point):
                self.points.append(point)

        self.num_points = len(self.points)

    def set_num_points_in_box(self, points, num):
        self.points = points
        self.num_points = num

    def get_center(self):
        return self.location

    def rotate(self, rotation_matrix):
        pass

    def translate(self, translation_vector):
        pass

    def bbox_center_to_corner(self):
        translation = np.tile(self.location, (8, 1))
        width, length, height = self.dimension
        rot = self.rotation
        # print("dim", width, length, height)
        bbox_corner = np.array([
            [-length / 2, -length / 2, length / 2, length / 2, -length / 2, -length / 2, length / 2, length / 2],
            [width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2],
            [-height / 2, -height / 2, -height / 2, -height / 2, height / 2, height / 2, height / 2, height / 2]
        ])
        # print("bbox", bbox_corner)
        rotation_matrix = np.array([
            [np.cos(rot), -np.sin(rot), 0.0],
            [np.sin(rot), np.cos(rot), 0.0],
            [0.0, 0.0, 1.0]
        ])

        bbox_corner = rotation_matrix @ bbox_corner + translation.T
        # print("bbox_translate", bbox_corner)
        self.corner_points = bbox_corner.T
        return bbox_corner

    def check_point_in_bbox(self, point):
        is_inside = False
        if self.corner_points is None:
            corner_points = self.bbox_center_to_corner().T
        else:
            corner_points = self.corner_points
        # print(corner_points)
        u = corner_points[0] - corner_points[1]
        v = corner_points[0] - corner_points[3]
        w = corner_points[0] - corner_points[4]
        if u @ corner_points[1].T <= u @ point.T <= u @ corner_points[0].T and v @ corner_points[
            3].T <= v @ point.T <= v @ corner_points[0].T and w @ corner_points[4].T <= w @ point.T <= w @ \
                corner_points[0].T:
            is_inside = True
        return is_inside

    def normalized(self):
        # assert len(self.points) != 0, f"self.points=={len(self.points)}, load_frame() before normalizing"
        normalized = []

        for point in self.points:
            normalized.append(point - self.location)

        return normalized

    def create_bbox(self, bbox_corners, color=None):
        """

        Args:
            bbox_corners: list of corner points
            color: line colors [R, G, B]

        Returns:

        """
        assert bbox_corners is not None
        if color is None:
            color = [1, 0, 0]

        lines = [[0, 1], [1, 2], [2, 3], [0, 3],
                 [4, 5], [5, 6], [6, 7], [4, 7],
                 [0, 4], [1, 5], [2, 6], [3, 7]]

        colors = [color for _ in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bbox_corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set

    def visualize(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        vis.add_geometry(pcd)
        vis.add_geometry(self.create_bbox(self.corner_points))
        vis.run()
        vis.destroy_window()
