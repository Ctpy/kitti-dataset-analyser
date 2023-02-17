from dataclasses import dataclass
from datasets.dataset import Dataset
from bbox.bbox import BoundingBox3D
from analysistool.tool import Tool


@dataclass
class Analyser:

    dataset: Dataset
    bounding_boxes: list[BoundingBox3D]
    tool_list: list[Tool]

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def add_tool(self, tool: Tool):
        self.tool_list.append(tool)
