import torch
from ultralytics import YOLO
import numpy as np
import os


class YOLOv9Classifier:
    """
    YOLOv9 Classifier with customizable architecture and hyperparameters.
    """

    def __init__(
        self,
        model_name="yolov9e.pt",  # YOLOv9 model file
        lr=1e-3,
        lrf=0.01,
        weight_decay=5e-4,
        dropout=0.0,
        fraction=1.0,
        epochs=100,
        batch_size=4,  # Default taken from CFG
        device="cpu",
        optimizer_type="auto",  # YOLO uses built-in optimizers
        patience=20,
        profile=False,
        label_smoothing=0.0,
        random_state=None,
        seed=42,
        verbose=False,
        exp_name="experiment",
        data_yaml=None,
        imgsz=(640, 640),
        task="detect",
        val=False,
        amp=True,
        exist_ok=True,
        resume=False,
        output_dir=".",
    ):
        """
        Initialize the YOLOv9 classifier with the provided architecture and hyperparameters.
        """
        self.model_name = model_name
        self.task = task
        self.lr = lr
        self.lrf = lrf
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.fraction = fraction
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.optimizer_type = optimizer_type
        self.patience = patience
        self.profile = profile
        self.label_smoothing = label_smoothing
        self.random_state = random_state
        self.verbose = verbose
        self.seed = seed
        self.exp_name = exp_name
        self.data_yaml = data_yaml
        self.imgsz = imgsz
        self.val = val
        self.amp = amp
        self.exist_ok = exist_ok
        self.resume = resume
        self.output_dir = output_dir

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

    def fit(self):
        """
        Train the YOLOv9 model on the data. Handles both training and validation logic.
        """
        if self.data_yaml is None:
            raise ValueError("Data YAML file must be provided for YOLO training.")

        # Train the model using provided parameters
        self.model.train(
            data=self.data_yaml,  # Path to dataset YAML file
            task=self.task,
            imgsz=self.imgsz,  # Image size
            epochs=self.epochs,
            batch=self.batch_size,
            optimizer=self.optimizer_type,
            lr0=self.lr,
            lrf=self.lrf,
            weight_decay=self.weight_decay,
            dropout=self.dropout,
            patience=self.patience,
            fraction=self.fraction,
            profile=self.profile,
            label_smoothing=self.label_smoothing,
            name=self.exp_name,
            seed=self.seed,
            val=self.val,
            amp=self.amp,  # Using automatic mixed precision
            exist_ok=self.exist_ok,
            resume=self.resume,
            device=self.device,
            verbose=self.verbose,
            project=self.output_dir,
        )

    def evaluate(self):
        """
        Evaluate the YOLOv9 model on the validation data.
        """
        if self.data_yaml is None:
            raise ValueError("Data YAML file must be provided for YOLO evaluation.")

        metrics = self.model.val(
            data=self.data_yaml,  # Path to dataset YAML file
            imgsz=self.imgsz,
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
