import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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
    transform : torchvision.transforms
        Transformation to apply to the images.
    """

    def __init__(
        self,
        image_dir,
        label_dir,
        yaml_path=None,
        batch_size=8,
        shuffle=True,
        transform=None,
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
        transform : callable, optional
            Optional transform to be applied on an image.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        # Используем приватный метод _get_transforms для ресайза с паддингом
        self.transform = transform if transform else self._get_transforms((640, 640))
        self.image_filenames = [
            f
            for f in os.listdir(self.image_dir)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]
        self.class_names = self.load_yaml_classes(yaml_path) if yaml_path else []

        # Create the dataset
        self.dataset = YOLODataset(self.image_dir, self.label_dir, self.transform)

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
        return torch.stack(images, 0), list(labels)

    def _get_transforms(self, image_size=(640, 640)):
        """
        Private method to resize image with padding to maintain aspect ratio.

        Parameters
        ----------
        image_size : tuple
            The target size of the image after resizing and padding (default is (640, 640)).

        Returns
        -------
        transforms.Compose
            A composition of transforms that resizes the image and applies padding.
        """

        def calculate_padding(img):
            """Calculate padding for each side to maintain aspect ratio."""
            width, height = img.size
            pad_left = (image_size[1] - width) // 2
            pad_top = (image_size[0] - height) // 2
            pad_right = image_size[1] - width - pad_left
            pad_bottom = image_size[0] - height - pad_top
            return (pad_left, pad_top, pad_right, pad_bottom)

        return transforms.Compose(
            [
                transforms.Resize(
                    image_size[0], interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.Lambda(
                    lambda img: transforms.functional.pad(
                        img, calculate_padding(img), fill=(0, 0, 0)
                    )
                ),
                transforms.ToTensor(),
            ]
        )


class YOLODataset(Dataset):
    """
    Dataset class for loading YOLO formatted data (images and labels).
    """

    def __init__(self, image_dir, label_dir, transform):
        """
        Parameters
        ----------
        image_dir : str
            Directory containing the images.
        label_dir : str
            Directory containing the labels.
        transform : callable
            Transform to be applied to the images.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_filenames = [
            f
            for f in os.listdir(self.image_dir)
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
        image : torch.Tensor
            Transformed image tensor.
        label : torch.Tensor
            Tensor containing the labels for the image.
        """
        # Load the image
        img_filename = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

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
            # Convert the label data to a tensor
            label_tensor = torch.tensor(
                [list(map(float, line.strip().split())) for line in label_data]
            )
        else:
            label_tensor = torch.tensor([])  # If no label, return an empty tensor

        return image, label_tensor
