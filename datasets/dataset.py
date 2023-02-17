from abc import ABC, abstractmethod


class Dataset(ABC):

    def __init__(self, root_path):
        self.root_path = root_path

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def load(self, split):
        pass

    @abstractmethod
    def extract_bounding_boxes(self, idx):
        pass

    @abstractmethod
    def format(self, data):
        pass

    @abstractmethod
    def visualize(self, frame):
        pass
