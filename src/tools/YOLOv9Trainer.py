from sklearn.model_selection import ParameterGrid


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

    def train(self, models, param_grids):
        """
        Train the YOLOv9 models using manual hyperparameter tuning.
        """
        print(f"Training on device: {self.device}")

        total_iterations = 0
        for model_name, model in models.items():
            param_combinations = list(ParameterGrid(param_grids[model_name]))
            total_iterations += sum([params["epochs"] for params in param_combinations])

        for model_name, model in models.items():
            param_grid = param_grids[model_name]
            param_combinations = list(ParameterGrid(param_grid))

            for params in param_combinations:
                print(f"\nTraining {model_name} with parameters: {params}")
                model.set_params(**params)

                model.fit()

                metrics = model.evaluate()

                score = metrics.box.map

                self.best_scores[model_name] = score

                if score > self.best_model_score:
                    self.best_model_name = model_name
                    self.best_model_score = score
                    self.best_estimators[model_name] = model
                    self.best_params[model_name] = params

        print(
            f"\nBest Model: {self.best_model_name} with score: {self.best_model_score}"
        )
