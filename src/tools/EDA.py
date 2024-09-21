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
    yolo_loaders : dict
        A dictionary containing 'train', 'valid', 'test' YOLODataLoader instances.
    class_names : list
        List of class names in the YOLODataLoader.
    """

    def __init__(self, yolo_loaders):
        """
        Constructs all the necessary attributes for the EDA object.

        Parameters
        ----------
        yolo_loaders : dict
            Dictionary containing 'train', 'valid', 'test' YOLODataLoader instances.
        """
        self.yolo_loaders = yolo_loaders
        self.class_names = list(yolo_loaders.values())[0].class_names

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
        num_cols = 3  # Set the number of columns
        num_rows = (
            num_images + num_cols - 1
        ) // num_cols  # Calculate the number of rows dynamically

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

        dataloader = self.yolo_loaders[loader_type].get_loader()

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

        # Remove empty subplots
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

            # Convert YOLO format (center x, center y, width, height) to (x_min, y_min, width, height)
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

        # Собираем информацию для каждого набора данных (train, valid, test)
        for mode, loader in self.yolo_loaders.items():
            mode_class_count = {
                self.class_names[i]: 0 for i in range(len(self.class_names))
            }

            # Пробегаем по каждой партии данных в текущем DataLoader
            for images, labels in loader.get_loader():
                for label in labels:
                    for obj in label:
                        class_id = int(obj[0])
                        mode_class_count[self.class_names[class_id]] += 1

            class_info.append({**mode_class_count, "Mode": mode})

        dataset_stats_df = pd.DataFrame(class_info)

        # Визуализируем распределение классов для каждого набора
        fig, axes = plt.subplots(1, len(self.yolo_loaders), figsize=(15, 5))
        for i, (mode, loader) in enumerate(self.yolo_loaders.items()):
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
        dataloader = self.yolo_loaders[loader_type].get_loader()

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

        # Собираем информацию для каждого набора данных (train, valid, test)
        for mode, loader in self.yolo_loaders.items():
            mode_class_count = {
                self.class_names[i]: 0 for i in range(len(self.class_names))
            }

            # Пробегаем по каждой партии данных в текущем DataLoader
            for images, labels in loader.get_loader():
                for label in labels:
                    for obj in label:
                        class_id = int(obj[0])
                        mode_class_count[self.class_names[class_id]] += 1

            class_info.append({**mode_class_count, "Mode": mode})

        # Преобразуем данные в DataFrame
        dataset_stats_df = pd.DataFrame(class_info)
        display(dataset_stats_df)
