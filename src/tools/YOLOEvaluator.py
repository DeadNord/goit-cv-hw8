import torch
import pandas as pd
import matplotlib.pyplot as plt
from torchsummary import summary
import glob
from PIL import Image
from IPython.display import display


class YOLOEvaluator:
    """
    A class to evaluate and display YOLO model performance results for object detection tasks.
    """

    def __init__(self, output_dir, exp_name):
        """
        Initialize the evaluator with the paths where model outputs are stored.

        Parameters
        ----------
        output_dir : str
            Directory where the model outputs are saved.
        exp_name : str
            Experiment name or folder where model results are stored.
        """
        self.output_dir = output_dir
        self.exp_name = exp_name
        self.log_path = f"{self.output_dir}/{self.exp_name}/results.csv"

    def load_training_logs(self):
        """
        Load the training logs from the CSV file.

        Returns
        -------
        df : pandas.DataFrame
            The DataFrame containing the training logs.
        """
        df = pd.read_csv(self.log_path)
        df = df.rename(columns=lambda x: x.replace(" ", ""))
        return df

    def display_results(
        self,
        best_models,
        best_params,
        best_scores,
        best_model_name,
        help_text=False,
    ):
        """
        Displays the evaluation metrics for the best models and their parameters using the logs.
        """
        # Load the training and validation logs
        df = self.load_training_logs()

        # Извлечение финальных метрик
        final_results = df.iloc[-1]

        # Извлечение метрик из логов
        results = [
            {
                "Model": best_model_name,
                "Precision": final_results.get("metrics/precision(B)", "N/A"),
                "Recall": final_results.get("metrics/recall(B)", "N/A"),
                "mAP 50": final_results.get("metrics/mAP50(B)", "N/A"),
                "mAP 50-95": final_results.get("metrics/mAP50-95(B)", "N/A"),
                "Validation Box Loss": final_results.get("val/box_loss", "N/A"),
                "Validation Cls Loss": final_results.get("val/cls_loss", "N/A"),
                "Validation DFL Loss": final_results.get("val/dfl_loss", "N/A"),
            }
        ]

        results_df = pd.DataFrame(results)

        param_df = (
            pd.DataFrame(best_params).T.reset_index().rename(columns={"index": "Model"})
        )
        best_model_df = pd.DataFrame(
            {
                "Overall Best Model": [best_model_name],
                "Score (mAP 50-95)": [best_scores.get(best_model_name, "N/A")],
            }
        )

        # Display metrics
        print("Evaluation Metrics for Test Set:")
        display(results_df)

        print("\nBest Parameters for Each Model (found during hyperparameter tuning):")
        display(param_df)

        print("\nOverall Best Model and Score (based on validation mAP):")
        display(best_model_df)

        if help_text:
            print("\nMetric Explanations for Object Detection:")
            print("Box Loss: Localization loss (bounding box).")
            print("Cls Loss: Classification loss (object type).")
            print("DFL Loss: Distribution Focal Loss.")
            print("Precision: Ratio of correctly predicted positives.")
            print("Recall: Ratio of correctly predicted actual positives.")
            print("mAP 50: Mean Average Precision at IoU threshold 0.5.")
            print(
                "mAP 50-95: Mean Average Precision across IoU thresholds 0.5 to 0.95."
            )

    def display_predictions(self, num_columns=3):
        """
        Display predicted images from the model's output folder in a grid.

        Parameters
        ----------
        num_columns : int
            Number of columns for displaying images in a grid.
        """
        results_paths = [
            i
            for i in glob.glob(f"{self.output_dir}/{self.exp_name}/*.png")
            + glob.glob(f"{self.output_dir}/{self.exp_name}/*.jpg")
            if "batch" not in i
        ]
        results_paths = sorted(results_paths)

        if len(results_paths) == 0:
            print(
                f"No results found in the directory: {self.output_dir}/{self.exp_name}"
            )
            return

        print(
            f"Displaying {len(results_paths)} prediction images from {self.exp_name}:"
        )
        num_rows = len(results_paths) // num_columns + int(len(results_paths) % num_columns > 0)

        fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 5))
        axes = axes.flatten()

        for i, file in enumerate(results_paths):
            img = Image.open(file)
            axes[i].imshow(img)
            axes[i].axis("off")

        # Turn off any remaining empty axes
        for j in range(i + 1, num_rows * num_columns):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    def plot_training_metrics(self):
        """
        Plot the training and validation metrics for the model.
        """
        df = self.load_training_logs()

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

        # Training and Validation box_loss
        ax1.set_title("Box Loss")
        ax1.plot(
            df["epoch"],
            df["train/box_loss"],
            label="Training box_loss",
            marker="o",
            linestyle="-",
        )
        ax1.plot(
            df["epoch"],
            df["val/box_loss"],
            label="Validation box_loss",
            marker="o",
            linestyle="-",
        )
        ax1.set_ylabel("Box Loss")
        ax1.legend()
        ax1.grid(True)

        # Training and Validation cls_loss
        ax2.set_title("Cls Loss")
        ax2.plot(
            df["epoch"],
            df["train/cls_loss"],
            label="Training cls_loss",
            marker="o",
            linestyle="-",
        )
        ax2.plot(
            df["epoch"],
            df["val/cls_loss"],
            label="Validation cls_loss",
            marker="o",
            linestyle="-",
        )
        ax2.set_ylabel("Cls Loss")
        ax2.legend()
        ax2.grid(True)

        # Training and Validation dfl_loss
        ax3.set_title("DFL Loss")
        ax3.plot(
            df["epoch"],
            df["train/dfl_loss"],
            label="Training dfl_loss",
            marker="o",
            linestyle="-",
        )
        ax3.plot(
            df["epoch"],
            df["val/dfl_loss"],
            label="Validation dfl_loss",
            marker="o",
            linestyle="-",
        )
        ax3.set_xlabel("Epochs")
        ax3.set_ylabel("DFL Loss")
        ax3.legend()
        ax3.grid(True)

        plt.suptitle("Training Metrics vs. Epochs")
        plt.show()

    def plot_map_history(self):
        """
        Plot the mAP 50 and mAP 50-95 history of the YOLO model from logs.
        """
        df = self.load_training_logs()

        plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@50")
        plt.plot(
            df["epoch"],
            df["metrics/mAP50-95(B)"],
            label="mAP@50:95",
            color="orange",
        )
        plt.title("mAP@50 and mAP@50:95 over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("mAP")
        plt.legend()
        plt.grid(True)
        plt.show()