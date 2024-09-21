import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import yaml


class YOLODataLoader:
    """
    Custom DataLoader class for loading YOLO formatted data (images and labels)
    and returning a DataLoader directly.

    Attributes
    ----------
    image_dir : str
        Path to the directory containing images.
    label_dir : str
        Path to the directory containing labels in YOLO format.
    batch_size : int
        The number of samples in each batch.
    shuffle : bool
        Whether to shuffle the dataset.
    """

    def __init__(
        self,
        image_dir,
        label_dir,
        yaml_path=None,
        batch_size=8,
        shuffle=True,
    ):
        """
        Parameters
        ----------
        image_dir : str
            Directory containing the images.
        label_dir : str
            Directory containing the labels.
        yaml_path : str, optional
            Path to the YAML file with class names and other metadata.
        batch_size : int, optional
            The number of samples in each batch (default is 8).
        shuffle : bool, optional
            Whether to shuffle the dataset (default is True).
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.image_filenames = [
            f for f in os.listdir(self.image_dir)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]
        self.class_names = self.load_yaml_classes(yaml_path) if yaml_path else []

        # Create the dataset
        self.dataset = YOLODataset(self.image_dir, self.label_dir)

        # Create the DataLoader
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
        )

    def get_loader(self):
        """
        Returns the DataLoader instance.
        """
        return self.loader

    def load_yaml_classes(self, yaml_path):
        """
        Loads class names from a YAML file.

        Parameters
        ----------
        yaml_path : str
            Path to the YAML file containing class information.

        Returns
        -------
        class_names : list
            List of class names from the YAML file.
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML file not found at {yaml_path}")

        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)

        # Extract the class names from the YAML file
        class_names = data.get("names", [])

        if not class_names:
            raise ValueError("No class names found in the YAML file.")

        print(f"Classes loaded: {class_names}")
        return class_names

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle variable size labels (for YOLO formatted data).
        """
        images, labels = zip(*batch)
        return list(images), list(labels)


class YOLODataset(Dataset):
    """
    Dataset class for loading YOLO formatted data (images and labels).
    """

    def __init__(self, image_dir, label_dir):
        """
        Parameters
        ----------
        image_dir : str
            Directory containing the images.
        label_dir : str
            Directory containing the labels.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_filenames = [
            f for f in os.listdir(self.image_dir)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        """Returns the total number of image files."""
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        Fetches an image and its corresponding label.

        Parameters
        ----------
        idx : int
            Index of the image file.

        Returns
        -------
        image : PIL.Image
            The image in PIL format.
        label : list
            List of labels (bounding boxes and classes) for the image.
        """
        # Load the image
        img_filename = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        image = Image.open(img_path).convert("RGB")

        # Load the corresponding label
        label_filename = (
            img_filename.replace(".png", ".txt")
            .replace(".jpg", ".txt")
            .replace(".jpeg", ".txt")
        )
        label_path = os.path.join(self.label_dir, label_filename)

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                label_data = f.readlines()
            # Convert the label data to a list of floats
            label = [list(map(float, line.strip().split())) for line in label_data]
        else:
            label = []  # If no label, return an empty list

        return image, label
