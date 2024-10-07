import sys
import importlib.util
from lightning import Trainer
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint, EarlyStopping
from torch import Tensor

if importlib.util.find_spec("ipywidgets") is not None:
    from tqdm.auto import tqdm
else:
    from tqdm import tqdm


class GlobalProgressBar(TQDMProgressBar):

    def __init__(self):
        super().__init__(process_position=1)
        self.main_progress_bar = None

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.main_progress_bar = tqdm(
            desc="Global",
            initial=trainer.current_epoch,
            total=trainer.max_epochs,
            position=1,
            disable=False,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        self.main_progress_bar.close()

    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        self.main_progress_bar.update(1)


class LazyModelCheckPoint(ModelCheckpoint):
    """
    Enables choosing whether to update the model checkpoint
    when the metric is equal to the best value.
    """

    def __init__(
        self,
        *args,
        lazy: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lazy = lazy

    def check_monitor_top_k(
        self, trainer: Trainer, current: Tensor | None = None
    ) -> bool:
        """
        Check if the current model should be saved based on the metric being equal or better than the top k models.
        """
        if self.lazy:
            return super().check_monitor_top_k(trainer=trainer, current=current)

        if current is None:
            return False

        if self.save_top_k == -1:
            return True

        less_than_k_models = len(self.best_k_models) < self.save_top_k
        if less_than_k_models:
            return True

        monitor_op = {"min": lambda x, y: x <= y, "max": lambda x, y: x >= y}[self.mode]
        should_update_best_and_save = monitor_op(
            current, self.best_k_models[self.kth_best_model_path]
        )

        # If using multiple devices, make sure all processes are unanimous on the decision.
        should_update_best_and_save = trainer.strategy.reduce_boolean_decision(
            bool(should_update_best_and_save)
        )

        return should_update_best_and_save


class LazyEarlyStopping(EarlyStopping):

    def __init__(
        self,
        *args,
        lazy: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if not lazy:
            self.mode_dict = {"min": lambda x, y: x <= y, "max": lambda x, y: x >= y}
            self.order_dict = {"min": "<=", "max": ">="}
