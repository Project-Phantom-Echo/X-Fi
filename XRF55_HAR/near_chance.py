"""Training-only collapse detection; accuracy is a fraction, not percent."""
from dataclasses import dataclass, field


class TrainingAtChance(RuntimeError):
    pass


@dataclass
class NearChanceStop:
    patience: int = 30
    threshold: float = 0.022
    history: list = field(default_factory=list)
    best_accuracy: float = 0.0

    def observe(self, loss, accuracy, samples):
        self.best_accuracy = max(self.best_accuracy, accuracy)
        self.history.append({'epoch': len(self.history) + 1, 'loss': loss,
                             'accuracy': accuracy, 'samples': samples})
        if len(self.history) >= self.patience and self.best_accuracy <= self.threshold:
            raise TrainingAtChance(
                f'best training accuracy {self.best_accuracy:.6f} <= '
                f'{self.threshold:.6f} through epoch {len(self.history)}')

    def record(self):
        return {'patience_epochs': self.patience, 'threshold_accuracy': self.threshold,
                'best_train_accuracy': self.best_accuracy,
                'completed_epochs': len(self.history), 'training_history': self.history}
