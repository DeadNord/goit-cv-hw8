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
    train_dir : str
        Path to the directory containing train images and labels.
    val_dir : str
        Path to the directory containing validation images and labels.
    test_dir : str
        Path to the directory containing test images and labels.
    batch_size : int
        The number of samples in each batch.
    shuffle : bool
        Whether to shuffle the dataset.
    """

    def __init__(
        self,
        train_dir,
        val_dir,
        test_dir,
        yaml_path=None,
        batch_size=8,
        shuffle=True,
    ):
        """
        Parameters
        ----------
        train_dir : str
            Directory containing the train images and labels.
        val_dir : str
            Directory containing the validation images and labels.
        test_dir : str
            Directory containing the test images and labels.
        yaml_path : str, optional
            Path to the YAML file with class names and other metadata.
        batch_size : int, optional
            The number of samples in each batch (default is 8).
        shuffle : bool, optional
            Whether to shuffle the dataset (default is True).
        """
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_image_dir = os.path.join(train_dir, "images")
        self.train_label_dir = os.path.join(train_dir, "labels")

        self.val_image_dir = os.path.join(val_dir, "images")
        self.val_label_dir = os.path.join(val_dir, "labels")

        self.test_image_dir = os.path.join(test_dir, "images")
        self.test_label_dir = os.path.join(test_dir, "labels")

        self.class_names = self.load_yaml_classes(yaml_path) if yaml_path else []

        # Create datasets
        self.train_dataset = YOLODataset(self.train_image_dir, self.train_label_dir)
        self.val_dataset = YOLODataset(self.val_image_dir, self.val_label_dir)
        self.test_dataset = YOLODataset(self.test_image_dir, self.test_label_dir)

        # Create DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def get_loaders(self):
        """
        Returns train, val, and test DataLoader instances.
        """
        return self.train_loader, self.val_loader, self.test_loader

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

    def update_yaml_paths(self, yaml_path, output_yaml_path):
        """
        Update paths in the YAML file with the correct paths of the train, val, and test directories.

        Parameters
        ----------
        yaml_path : str
            Path to the original YAML file.
        output_yaml_path : str
            Path to save the updated YAML file with new paths for train, val, and test.
        """
        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)

        # Update the paths for train, val, and test with absolute paths
        data["train"] = os.path.abspath(os.path.join(self.train_dir, "images"))
        data["val"] = os.path.abspath(os.path.join(self.val_dir, "images"))
        data["test"] = os.path.abspath(os.path.join(self.test_dir, "images"))

        with open(output_yaml_path, "w") as file:
            yaml.safe_dump(data, file)

        print(f"YAML paths updated and saved to {output_yaml_path}")
        return output_yaml_path


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
        image : PIL.Image
            The image in PIL format.
        label : list
            List of labels (bounding boxes and classes) for the image.
        """
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
            label = [list(map(float, line.strip().split())) for line in label_data]
        else:
            label = []

        return image, label
