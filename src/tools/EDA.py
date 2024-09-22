import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as patches
from IPython.display import display


class EDA:
    """
    A class to perform Exploratory Data Analysis (EDA) on image datasets, specifically for YOLO-based tasks.

    Attributes
    ----------
    yolo_loader : YOLODataLoader
        An instance of YOLODataLoader that contains 'train', 'valid', and 'test' DataLoaders.
    class_names : list
        List of class names in the YOLODataLoader.
    """

    def __init__(self, yolo_loader):
        """
        Constructs all the necessary attributes for the EDA object.

        Parameters
        ----------
        yolo_loader : YOLODataLoader
            An instance of YOLODataLoader that contains 'train', 'valid', and 'test' DataLoaders.
        """
        self.yolo_loader = yolo_loader
        self.class_names = yolo_loader.class_names

    def show_sample_images(self, num_images=6, loader_type="train"):
        """
        Displays a grid of sample images from the dataset.

        Parameters
        ----------
        num_images : int, optional
            Number of images to display (default is 6).
        loader_type : str, optional
            Specifies which DataLoader to use ('train', 'valid', 'test').
        """
        images_shown = 0
        num_cols = 3
        num_rows = (num_images + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

        dataloaders = self.yolo_loader.get_loaders()

        if loader_type == "train":
            dataloader = dataloaders[0]
        elif loader_type == "valid":
            dataloader = dataloaders[1]
        elif loader_type == "test":
            dataloader = dataloaders[2]
        else:
            raise ValueError("loader_type should be one of 'train', 'valid', or 'test'")

        for images, labels in dataloader:
            for i in range(len(images)):
                if images_shown >= num_images:
                    break

                row = images_shown // num_cols
                col = images_shown % num_cols
                ax = axes[row, col] if num_rows > 1 else axes[col]

                ax.imshow(images[i])
                ax.axis("off")
                self.display_bboxes_on_image(ax, labels[i], images[i])
                images_shown += 1

            if images_shown >= num_images:
                break

        for j in range(images_shown, num_rows * num_cols):
            fig.delaxes(
                axes[j // num_cols, j % num_cols]
                if num_rows > 1
                else axes[j % num_cols]
            )

        plt.tight_layout()
        plt.show()

    def display_bboxes_on_image(self, ax, label, image):
        """
        Helper function to display bounding boxes on an image.

        Parameters
        ----------
        ax : matplotlib axis
            Axis to plot on.
        label : list
            List of labels containing bounding boxes.
        image : PIL.Image
            The image to display.
        """
        if len(label) == 0:
            return

        for obj in label:
            class_idx = int(obj[0])
            x_center, y_center, width, height = obj[1:]

            x_min = (x_center - width / 2) * image.size[0]
            y_min = (y_center - height / 2) * image.size[1]
            width *= image.size[0]
            height *= image.size[1]

            rect = patches.Rectangle(
                (x_min, y_min),
                width,
                height,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                x_min,
                y_min,
                self.class_names[class_idx],
                color="white",
                verticalalignment="top",
                bbox={"facecolor": "red", "alpha": 0.5, "pad": 1},
            )

    def plot_class_distribution(self):
        """
        Plots the class distribution across different dataset modes (train/val/test).
        """
        class_info = []

        train_loader, val_loader, test_loader = self.yolo_loader.get_loaders()
        dataloaders = {"train": train_loader, "valid": val_loader, "test": test_loader}

        for mode, dataloader in dataloaders.items():
            mode_class_count = {
                self.class_names[i]: 0 for i in range(len(self.class_names))
            }

            for images, labels in dataloader:
                for label in labels:
                    for obj in label:
                        class_id = int(obj[0])
                        mode_class_count[self.class_names[class_id]] += 1

            class_info.append({**mode_class_count, "Mode": mode})

        dataset_stats_df = pd.DataFrame(class_info)

        fig, axes = plt.subplots(1, len(dataloaders), figsize=(15, 5))
        for i, (mode, dataloader) in enumerate(dataloaders.items()):
            sns.barplot(
                data=dataset_stats_df[dataset_stats_df["Mode"] == mode].drop(
                    columns="Mode"
                ),
                orient="v",
                ax=axes[i],
                palette="Set2",
            )
            axes[i].set_title(f"{mode.capitalize()} Class Statistics")
            axes[i].set_xlabel("Classes")
            axes[i].set_ylabel("Count")
            axes[i].tick_params(axis="x", rotation=90)

        plt.tight_layout()
        plt.show()

    def display_image_by_index(self, idx, loader_type="train"):
        """
        Displays a specific image by its index in the dataloader with bounding boxes.

        Parameters
        ----------
        idx : int
            Index of the image to display.
        loader_type : str
            Specifies which DataLoader to use ('train', 'valid', 'test').
        """
        img, label = None, None
        count = 0
        dataloaders = self.yolo_loader.get_loaders()
        dataloader = dataloaders[loader_type]

        for imgs, labels in dataloader:
            for i in range(len(imgs)):
                if count == idx:
                    img, label = imgs[i], labels[i]
                    break
                count += 1
            if img is not None:
                break

        if img is not None:
            fig, ax = plt.subplots(1)
            ax.imshow(img)
            ax.axis("off")
            self.display_bboxes_on_image(ax, label, img)
            plt.show()
        else:
            print(f"Index {idx} is out of range.")

    def show_dataset_statistics(self):
        """
        Displays class statistics for train/valid/test datasets.
        """
        class_info = []

        train_loader, val_loader, test_loader = self.yolo_loader.get_loaders()
        dataloaders = {"train": train_loader, "valid": val_loader, "test": test_loader}

        for mode, dataloader in dataloaders.items():
            mode_class_count = {
                self.class_names[i]: 0 for i in range(len(self.class_names))
            }

            for images, labels in dataloader:
                for label in labels:
                    for obj in label:
                        class_id = int(obj[0])
                        mode_class_count[self.class_names[class_id]] += 1

            class_info.append({**mode_class_count, "Mode": mode})

        dataset_stats_df = pd.DataFrame(class_info)
        display(dataset_stats_df)
