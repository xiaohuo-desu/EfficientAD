"""Sample Model class for track 1."""

import torch
from torch import nn
from torchvision import transforms

from common import get_pdn_small

class Patchcore(nn.Module):
    """Examplary Model class for track 1.

    This class contains the torch model and the transformation pipeline.
    Forward-pass should first transform the input batch and then pass it through the model.

    Note:
        This is the example model class name. You can replace it with your model class name.

    Args:
        backbone (str): Name of the backbone model to use.
            Default: "wide_resnet50_2".
        layers (list[str]): List of layer names to use.
            Default: ["layer1", "layer2", "layer3"].
        pre_trained (bool): If True, use pre-trained weights.
            Default: True.
        num_neighbors (int): Number of neighbors to use.
            Default: 9.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: list[str] = ["layer1", "layer2", "layer3"],  # noqa: B006
        pre_trained: bool = True,
        num_neighbors: int = 9,
    ) -> None:
        super().__init__()

        # NOTE: Create your transformation pipeline here.
        # We use Resize, CenterCrop, and Normalize for an example.
        self.transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

        # NOTE: Create your model here. We use PatchcoreModel for an example.
        self.teacher = get_pdn_small(384)
        self.student = get_pdn_small(2 * 384)
        teacher_state_dict = torch.load(config.weights, map_location='cpu')
        self.teacher.load_state_dict(teacher_state_dict)
        student_state_dict = torch.load(config.weights, map_location='cpu')
        self.student.load_state_dict(student_state_dict)



    def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Transform the input batch and pass it through the model.

        This model returns a dictionary with the following keys
        - ``anomaly_map`` - Anomaly map.
        - ``pred_score`` - Predicted anomaly score.
        """
        """q_st_start = torch.quantile(maps_st, q=0.9)
        q_st_end = torch.quantile(maps_st, q=0.995)"""
        orig_width = image.width
        orig_height = image.height
        image = self.transform(image)
        image = image[None]
        map_st = predict(
            image=image, teacher=self.teacher, student=self.student,
            teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=None, q_st_end=None,
            )
        map_st = torch.nn.functional.pad(map_st, (4, 4, 4, 4))
        map_st = torch.nn.functional.interpolate(
            map_st, (orig_height, orig_width), mode='bilinear')
        map_st = map_st[0, 0].cpu().numpy()

        y_true_image = 0 if defect_class == 'good' else 1
        y_score_image = np.max(map_st)
        y_true.append(y_true_image)
        y_score.append(y_score_image)   
        







        
        batch = self.transform(batch)
        return self.model(batch)
