import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
from torchsummary import summary
import glob
from PIL import Image


class YOLOEvaluator:
    """
    A class to evaluate and display PyTorch CNN model performance results for classification tasks.
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

    def display_results(
        self,
        test_dataset,
        best_models,
        best_params,
        best_scores,
        best_model_name,
        help_text=False,
    ):
        """
        Displays the evaluation metrics for the best models and their parameters using the test dataset.
        """
        results = []

        for model_name, cnn_model in best_models.items():
            all_preds, all_targets, val_loss, val_accuracy = cnn_model._evaluate(
                test_dataset
            )

            accuracy = accuracy_score(all_targets, all_preds)
            balanced_acc = balanced_accuracy_score(all_targets, all_preds)
            f1 = f1_score(all_targets, all_preds, average="weighted")
            precision = precision_score(all_targets, all_preds, average="weighted")
            recall = recall_score(all_targets, all_preds, average="weighted")

            results.append(
                {
                    "Model": model_name,
                    "Accuracy": accuracy,
                    "Balanced Accuracy": balanced_acc,
                    "F1 Score": f1,
                    "Precision": precision,
                    "Recall": recall,
                }
            )

        results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
        param_df = (
            pd.DataFrame(best_params).T.reset_index().rename(columns={"index": "Model"})
        )

        best_model_df = pd.DataFrame(
            {
                "Overall Best Model": [best_model_name],
                "Score (based on cross-validation score)": [
                    best_scores[best_model_name]
                ],
            }
        )

        # Display metrics
        print("Evaluation Metrics for Test Set:")
        display(results_df)

        print("\nBest Parameters for Each Model (found during hyperparameter tuning):")
        display(param_df)

        print("\nOverall Best Model and Score (based on cross-validation score):")
        display(best_model_df)

        if help_text:
            print("\nMetric Explanations for Classification:")
            print(
                "Accuracy: The ratio of correctly predicted instances to the total instances."
            )
            print("Balanced Accuracy: The average of recall obtained on each class.")
            print("F1 Score: Harmonic mean of precision and recall.")
            print(
                "Precision: Ratio of correctly predicted positive observations to all positive predictions."
            )
            print(
                "Recall: Ratio of correctly predicted positive observations to all actual positives."
            )

    def display_predictions(self):
        """
        Display predicted images from the model's output folder.
        """
        results_paths = [
            i for i in
            glob.glob(f'{self.output_dir}/{self.exp_name}/*.png') +
            glob.glob(f'{self.output_dir}/{self.exp_name}/*.jpg')
            if 'batch' not in i
        ]
        results_paths = sorted(results_paths)

        if len(results_paths) == 0:
            print(f"No results found in the directory: {self.output_dir}/{self.exp_name}")
            return

        print(f"Displaying {len(results_paths)} prediction images from {self.exp_name}:")
        for file in results_paths:
            img = Image.open(file)
            plt.imshow(img)
            plt.axis('off')
            plt.show()

    def load_training_logs(self):
        """
        Load the training logs from the CSV file.

        Returns
        -------
        df : pandas.DataFrame
            The DataFrame containing the training logs.
        """
        log_path = f'{self.output_dir}/{self.exp_name}/results.csv'
        df = pd.read_csv(log_path)
        df = df.rename(columns=lambda x: x.replace(" ", ""))
        df.to_csv(f'{self.output_dir}/training_log_df.csv', index=False)
        return df

    def plot_training_metrics(self):
        """
        Plot the training and validation metrics for the model.
        """
        df = self.load_training_logs()

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

        # Training and Validation box_loss
        ax1.set_title('Box Loss')
        ax1.plot(df['epoch'], df['train/box_loss'], label='Training box_loss', marker='o', linestyle='-')
        ax1.plot(df['epoch'], df['val/box_loss'], label='Validation box_loss', marker='o', linestyle='-')
        ax1.set_ylabel('Box Loss')
        ax1.legend()
        ax1.grid(True)

        # Training and Validation cls_loss
        ax2.set_title('Cls Loss')
        ax2.plot(df['epoch'], df['train/cls_loss'], label='Training cls_loss', marker='o', linestyle='-')
        ax2.plot(df['epoch'], df['val/cls_loss'], label='Validation cls_loss', marker='o', linestyle='-')
        ax2.set_ylabel('Cls Loss')
        ax2.legend()
        ax2.grid(True)

        # Training and Validation dfl_loss
        ax3.set_title('DFL Loss')
        ax3.plot(df['epoch'], df['train/dfl_loss'], label='Training dfl_loss', marker='o', linestyle='-')
        ax3.plot(df['epoch'], df['val/dfl_loss'], label='Validation dfl_loss', marker='o', linestyle='-')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('DFL Loss')
        ax3.legend()
        ax3.grid(True)

        plt.suptitle('Training Metrics vs. Epochs')
        plt.show()

    def plot_loss_history(self, best_models, best_model_name):
        """
        Plots the training and validation loss history of the provided PyTorch model.

        Parameters
        ----------
        best_models : dict
            Dictionary of best models from GridSearchCV.
        best_model_name : str
            Name of the best model to plot the loss history.
        """
        best_model = best_models[best_model_name]

        if hasattr(best_model, "train_loss_history") and hasattr(
            best_model, "val_loss_history"
        ):
            plt.plot(best_model.train_loss_history, label="Training Loss")
            plt.plot(
                best_model.val_loss_history,
                label="Validation Loss",
                color="orange",
            )
            plt.title("Training vs Validation Loss per Epoch")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
        else:
            print("The provided model does not have a loss history.")

    def plot_accuracy_history(self, best_models, best_model_name):
        """
        Plots the training and validation accuracy history of the provided PyTorch model.

        Parameters
        ----------
        best_models : dict
            Dictionary of best models from GridSearchCV.
        best_model_name : str
            Name of the best model to plot the accuracy history.
        """
        best_model = best_models[best_model_name]

        if hasattr(best_model, "train_accuracy_history") and hasattr(
            best_model, "val_accuracy_history"
        ):
            plt.plot(best_model.train_accuracy_history, label="Training Accuracy")
            plt.plot(
                best_model.val_accuracy_history,
                label="Validation Accuracy",
                color="orange",
            )
            plt.title("Training vs Validation Accuracy per Epoch")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.show()
        else:
            print("The provided model does not have an accuracy history.")

    def visualize_pipeline(self, model_name, best_models):
        """
        Visualizes the structure of a PyTorch model within the best models.

        Parameters
        ----------
        model_name : str
            The name of the model to visualize.
        best_models : dict
            A dictionary containing the best models.
        """
        best_model = best_models.get(model_name)
        if best_model is None:
            raise ValueError(f"Model with name {model_name} not found in best_models.")

        model = best_model.model
        if isinstance(model, torch.nn.Module):
            print(f"Visualizing the architecture of the model: {model_name}")
            summary(model, input_size=(3, 224, 224))
        else:
            raise ValueError(
                f"Model {model_name} is not a PyTorch nn.Module, but {type(model)}"
            )
