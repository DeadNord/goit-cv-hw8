import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.patches as patches


class EDA:
    """
    A class to perform Exploratory Data Analysis (EDA) on image datasets, specifically for YOLO-based tasks.

    Attributes
    ----------
    yolo_dataloader : YOLODataLoader
        The custom YOLODataLoader (train/val/test) to be analyzed.
    class_names : list
        List of class names in the YOLODataLoader.
    """

    def __init__(self, yolo_dataloader):
        """
        Constructs all the necessary attributes for the EDA object.

        Parameters
        ----------
        yolo_dataloader : YOLODataLoader
            The custom YOLODataLoader (train/val/test) to be analyzed.
        """
        self.dataloader = yolo_dataloader.get_loader()  # Access DataLoader from YOLODataLoader
        self.class_names = yolo_dataloader.class_names  # Access class names from YOLODataLoader

    def show_sample_images(self, num_images=6):
        """
        Displays a grid of sample images from the dataloader with bounding boxes.

        Parameters
        ----------
        num_images : int, optional
            Number of images to display (default is 6).
        """
        # Get a batch of images from the DataLoader
        images, labels = next(iter(self.dataloader))

        for i in range(min(num_images, len(images))):
            self.imshow(images[i], labels[i])

    def imshow(self, image, label):
        """
        Helper function to display an image with its bounding boxes.

        Parameters
        ----------
        image : torch.Tensor
            The tensor containing image data.
        label : torch.Tensor
            The tensor containing bounding box data.
        """
        # Convert image to numpy format
        img = image.numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)

        # Check if the label is empty
        if label.numel() == 0:
            print("Warning: No labels for this image.")
            return  # Skip images with no labels

        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Plot bounding boxes
        for obj in label:
            class_idx = int(obj[0])  # First value is the class index
            x_center, y_center, width, height = obj[1:]  # Rest are bbox coordinates

            # Convert YOLO format (center x, center y, width, height) to (x_min, y_min, width, height)
            x_min = (x_center - width / 2) * img.shape[1]
            y_min = (y_center - height / 2) * img.shape[0]
            width *= img.shape[1]
            height *= img.shape[0]

            # Create a rectangle patch
            rect = patches.Rectangle(
                (x_min, y_min),
                width,
                height,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
            plt.text(
                x_min,
                y_min,
                self.class_names[class_idx],
                color="white",
                verticalalignment="top",
                bbox={"facecolor": "red", "alpha": 0.5, "pad": 1},
            )

        plt.axis("off")
        plt.show()
        plt.close(fig)  # Закрываем текущую фигуру

    def show_images_from_each_class(self):
        """
        Displays one image from each class in the dataset with bounding boxes.
        """
        images_per_class = {}

        # Iterate through the DataLoader
        for imgs, labels in self.dataloader:
            for img, label in zip(imgs, labels):
                if label.numel() > 0:  # Skip images with no labels
                    for obj in label:
                        class_idx = int(obj[0])  # Get class index
                        class_name = self.class_names[class_idx]

                        if class_name not in images_per_class:
                            images_per_class[class_name] = (img, label)
                        if len(images_per_class) == len(self.class_names):
                            break
            if len(images_per_class) == len(self.class_names):
                break

        plt.figure(figsize=(10, 10))
        for i, (class_name, (img, label)) in enumerate(images_per_class.items()):
            ax = plt.subplot(1, len(self.class_names), i + 1)
            self.imshow(img, label)

    def plot_class_distribution(self):
        """
        Plots the distribution of objects across different classes in the dataloader.
        """
        class_counts = [0] * len(self.class_names)

        for _, labels in self.dataloader:
            for label in labels:
                if label.numel() > 0:  # Skip images with no labels
                    for obj in label:
                        class_idx = int(obj[0])
                        class_counts[class_idx] += 1

        plt.figure(figsize=(10, 6))
        plt.bar(self.class_names, class_counts, color="blue")
        plt.title("Class Distribution")
        plt.xlabel("Classes")
        plt.ylabel("Number of Objects")
        plt.show()

    def display_image_by_index(self, idx):
        """
        Displays a specific image by its index in the dataloader with bounding boxes.

        Parameters
        ----------
        idx : int
            Index of the image to display.
        """
        # Find the image and label by index
        img, label = None, None
        count = 0
        for imgs, labels in self.dataloader:
            for i in range(len(imgs)):
                if count == idx:
                    img, label = imgs[i], labels[i]
                    break
                count += 1
            if img is not None:
                break

        if img is not None:
            self.imshow(img, label)
        else:
            print(f"Index {idx} is out of range.")

    def show_image_shape(self):
        """
        Displays the shape of the first image in the dataloader to verify dimensions.
        """
        images, _ = next(iter(self.dataloader))
        for i, img in enumerate(images):
            print(f"Shape of image {i}: {img.shape}")
