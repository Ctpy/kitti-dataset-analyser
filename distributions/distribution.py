import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class Distribution:
    """
    A class to represent a distribution based on given data points

    '''
    Attributes
    ----------
    data : list [x: float, y: float, z: float]
        list of points in cartesian coordinates
    """
    def __init__(self, data):
        """
        Args:
            data: list[x: int, y: int, z: int]
        """
        self.data = np.asarray(data)
        self.hist = None
        self.bins = None

    def histogram(self, axis, bins):
        """
        Calculates the histogram of the data
        Args:
            axis: histogram over axis [x, y, z]
            bins: number of bins
        """
        assert isinstance(axis, int) and axis < 3
        self.hist, self.bins = np.histogram(self.data[:, axis], bins=bins, density=True)

    def KL_divergence(self, distribution):
        """
        Calculates the KL-Divergence between this distribution and the given distribution
        Args:
            distribution: Distribution with same bin size

        Returns:
            The total KL value
        """
        assert isinstance(distribution, Distribution)
        assert self.hist is not None
        assert distribution.hist is not None
        assert len(self.hist) == len(distribution.hist)

        kl_sum = 0
        for i in range(len(self.hist)):
            if distribution.hist[i] == 0:
                continue
            elif self.hist[i] == 0:
                continue
            kl_sum += self.hist[i] * np.log(self.hist[i]/distribution.hist[i])
        return kl_sum

    def plot(self, filename, axis=None, density=False):
        axis_mapping = ["X", "Y", "Z"]
        if axis is None:
            plt.hist(self.data, self.bins, density=False)
            plt.legend(axis_mapping)
            plt.savefig(filename)
            plt.clf()
        else:
            plt.hist(self.data.T[axis], self.bins, density=False)
            plt.legend(axis_mapping[axis])
            plt.savefig(filename)
            plt.clf()


if __name__ == '__main__':
    data = json.load(open("../output/goodcars_a9.json"))
    x_distribution = []
    y_distribution = []
    z_distribution = []
    counter = 0.0
    for k in data.keys():
        points = np.asarray(data[k]["points"]).T
        x, y, z = points[0], points[1], points[2]
        x_distribution.extend(x)
        y_distribution.extend(y)
        z_distribution.extend(z)
        counter += 1

    good_car = json.load(open("../output/goodcars_a9.json"))
    hist_good = Distribution(np.array([x_distribution, y_distribution, z_distribution]).T)
    hist_good.histogram(0, 10)
    hist_good.plot("test_good.pdf", axis=0, density=True)
    counter = 0
    dict_bbox = dict()
    for i in tqdm(range(5000)):
        try:
            data_processing = json.load(open(f"../output/file_{i}.json"))
            for k in data_processing.keys():
                if len(data_processing[k]["points"]) == 0:
                    continue
                hist_1 = Distribution(data_processing[k]["points"])
                hist_1.histogram(0, 10)
                # hist_1.plot("test.pdf", axis=0, density=True)
                val = hist_1.KL_divergence(hist_good)
                if val > 1.5:
                    dict_bbox[f"{counter}"] = data_processing[k]
                    counter += 1
        except:
            continue

        with open(f"good_cars.json", "w") as f:
            json.dump(dict_bbox, f)

    # counts, bins = np.histogram(x_distribution)
    # counts = counts / sum(counts * np.diff(bins))
    # plt.hist(bins[:-1], bins, weights=counts, density=True)
    # plt.title("Histogram of good cars x-axis")
    # plt.savefig("a9_hist_x_good_cars.png")
    # plt.clf()
    # counts, bins = np.histogram(y_distribution)
    # counts = counts / sum(counts * np.diff(bins))
    # plt.hist(bins[:-1], bins, weights=counts, density=True)
    # plt.title("Histogram of good cars y-axis")
    # plt.savefig("a9_hist_y_good_cars.png")
    # plt.clf()
    # counts, bins = np.histogram(z_distribution)
    # counts = counts / sum(counts * np.diff(bins))
    # plt.hist(bins[:-1], bins, weights=counts, density=True)
    # plt.title("Histogram of good cars z-axis")
    # plt.savefig("a9_hist_z_good_cars.png")
    # plt.clf()
    # data = json.load(open("../output/badcars_a9.json"))
    # x_distribution = []
    # y_distribution = []
    # z_distribution = []
    # counter = 0.0
    # for k in data.keys():
    #     points = np.asarray(data[k]["points"]).T
    #     x, y, z = points[0], points[1], points[2]
    #     x_distribution.extend(x)
    #     y_distribution.extend(y)
    #     z_distribution.extend(z)
    #     counter += 1
    # counts, bins = np.histogram(x_distribution)
    # print(bins)
    # counts = counts / sum(counts * np.diff(bins))
    # plt.hist(bins[:-1], bins, weights=counts, density=True)
    # plt.title("Histogram of bad cars x-axis")
    # plt.savefig("a9_hist_x_bad_cars.png")
    # plt.clf()
    # counts, bins = np.histogram(y_distribution)
    # counts = counts / sum(counts * np.diff(bins))
    # plt.hist(bins[:-1], bins, weights=counts, density=True)
    # plt.title("Histogram of bad cars y-axis")
    # plt.savefig("a9_hist_y_bad_cars.png")
    # plt.clf()
    # counts, bins = np.histogram(z_distribution)
    # counts = counts / sum(counts * np.diff(bins))
    # plt.hist(bins[:-1], bins, weights=counts, density=True)
    # plt.title("Histogram of bad cars z-axis")
    # plt.savefig("a9_hist_z_bad_cars.png")
    # plt.clf()