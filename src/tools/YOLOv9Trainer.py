from sklearn.model_selection import ParameterGrid
import os


class YOLOv9Trainer:
    """
    A class to train YOLOv9 models with manual hyperparameter tuning.
    Supports object detection tasks.
    """

    def __init__(self, device="cpu"):
        """
        Initialize the trainer with a device.
        """
        self.device = device
        self.best_estimators = {}
        self.best_params = {}
        self.best_scores = {}
        self.best_model_name = None
        self.best_model_score = float("-inf")
        self.best_model_path = None

    def train(self, models, param_grids, output_base_dir="./results"):
        """
        Train the YOLOv9 models using manual hyperparameter tuning.

        Parameters
        ----------
        models : dict
            Dictionary of models to be trained.
        param_grids : dict
            Dictionary containing parameter grids for each model.
        output_base_dir : str
            Base directory to store output results.
        """
        print(f"Training on device: {self.device}")

        total_iterations = 0
        for model_name, model in models.items():
            param_combinations = list(ParameterGrid(param_grids[model_name]))
            total_iterations += sum([params["epochs"] for params in param_combinations])

        experiment_count = 1

        for model_name, model in models.items():
            param_grid = param_grids[model_name]
            param_combinations = list(ParameterGrid(param_grid))

            for params in param_combinations:
                exp_name = f"{model_name}_experiment_{experiment_count}"

                exp_output_dir = os.path.join(output_base_dir, exp_name)
                if not os.path.exists(exp_output_dir):
                    os.makedirs(exp_output_dir)

                print(
                    f"\nTraining {model_name} with parameters: {params}, saving to {exp_output_dir}"
                )

                model.set_params(
                    **params, output_dir=output_base_dir, exp_name=exp_name
                )

                model.fit()

                metrics = model.evaluate()

                score = metrics.box.map

                self.best_scores[model_name] = score

                if score > self.best_model_score:
                    self.best_model_name = model_name
                    self.best_model_score = score
                    self.best_estimators[model_name] = model
                    self.best_params[model_name] = params
                    self.best_model_path = exp_output_dir

                experiment_count += 1

        print(
            f"\nBest Model: {self.best_model_name} with score: {self.best_model_score}"
        )
        print(f"Best model saved at: {self.best_model_path}")
