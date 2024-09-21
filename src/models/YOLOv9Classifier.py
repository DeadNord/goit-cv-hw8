import torch
from ultralytics import YOLO
import numpy as np


class YOLOv9Classifier:
    """
    YOLOv9 Classifier with customizable architecture and hyperparameters.
    """

    def __init__(
        self,
        model_name="yolov9e",  # YOLOv9 model file
        lr=0.001,
        epochs=100,
        batch_size=32,
        device="cpu",
        optimizer_type="auto",  # YOLO uses built-in optimizers
        random_state=None,
        epochs_logger=True,
    ):
        """
        Initialize the YOLOv9 classifier with the provided architecture and hyperparameters.
        """
        self.model_name = model_name
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.optimizer_type = optimizer_type
        self.random_state = random_state
        self.epochs_logger = epochs_logger

        # Load YOLO model
        self.model = YOLO(model_name)

        if random_state is not None:
            self._set_random_state(random_state)

    def _set_random_state(self, random_state):
        """
        Set the random seed for reproducibility.
        """
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(random_state)

    def set_params(self, **params):
        """
        Set parameters for the classifier and reinitialize the model.
        """
        for param, value in params.items():
            setattr(self, param, value)

    def fit(self, data_yaml=None, imgsz=(640, 640)):
        """
        Train the YOLOv9 model on the data. Handles both training and validation logic.
        """
        if data_yaml is None:
            raise ValueError("Data YAML file must be provided for YOLO training.")

        self.model.train(
            data=data_yaml,  # Path to dataset YAML file
            imgsz=imgsz,  # Image size
            epochs=self.epochs,
            batch=self.batch_size,
            optimizer=self.optimizer_type,
            lr0=self.lr,
            device=self.device,
        )

    def evaluate(self, data_yaml=None, imgsz=(640, 640)):
        """
        Evaluate the YOLOv9 model on the validation data.
        """
        if data_yaml is None:
            raise ValueError("Data YAML file must be provided for YOLO evaluation.")

        metrics = self.model.val(
            data=data_yaml,  # Path to dataset YAML file
            imgsz=imgsz,
            batch=self.batch_size,
            device=self.device,
        )
        return metrics

    def predict(self, img_paths):
        """
        Predict using the YOLOv9 model on the given images.
        """
        results = self.model.predict(img_paths)
        return results

    def save_model(self, path):
        """
        Save the YOLOv9 model to the specified path.
        """
        self.model.save(path)
