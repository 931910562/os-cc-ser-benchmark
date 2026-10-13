# STA optimization, logging, loss and evaluation utilities.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import os
from collections import Counter
import matplotlib.pyplot as plt
import io  # In-memory stream utilities.
import csv  # CSV logging utilities.
from datetime import datetime


class Accumulator(dict):
    def __init__(self, name_or_names, accumulate_fn=np.concatenate):
        super(Accumulator, self).__init__()
        self.names = (
            [name_or_names] if isinstance(name_or_names, str) else name_or_names
        )
        self.accumulate_fn = accumulate_fn
        for name in self.names:
            self.__setitem__(name, [])

    def updateData(self, scope):
        for name in self.names:
            self.__getitem__(name).append(scope[name])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb:
            print(exc_tb)
            return False

        for name in self.names:
            self.__setitem__(name, self.accumulate_fn(self.__getitem__(name)))

        return True


class TrainingModeManager:
    def __init__(self, nets, train=False):
        self.nets = nets
        self.modes = [net.training for net in nets]
        self.train = train

    def __enter__(self):
        for net in self.nets:
            net.train(self.train)

    def __exit__(self, exceptionType, exception, exceptionTraceback):
        for mode, net in zip(self.modes, self.nets):
            net.train(mode)
        self.nets = None  # release reference, to avoid imexplicit reference
        if exceptionTraceback:
            print(exceptionTraceback)
            return False
        return True


def clear_output():
    def clear():
        return

    try:
        from IPython.display import clear_output as clear
    except ImportError as e:
        pass

    def cls():
        os.system('cls' if os.name == 'nt' else 'clear')

    clear()
    cls()


def addkey(diction, key, global_vars):
    diction[key] = global_vars[key]


def track_scalars(logger, names, global_vars):
    """Read named values from the scope and log them as scalars."""
    values = {}
    for name in names:
        addkey(values, name, global_vars)
    for k in values:
        values[k] = variable_to_numpy(values[k])
    # Write each scalar through the logger.
    for k, v in values.items():
        logger.log_scalar(k, v)
    print(values)


def variable_to_numpy(x):
    """Convert a PyTorch tensor or array-like value to NumPy."""
    if isinstance(x, torch.Tensor):
        ans = x.cpu().detach().numpy()
        if torch.numel(x) == 1:
            return float(np.sum(ans))
        return ans
    return np.array(x)  # Convert non-tensor inputs to NumPy arrays.


def inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    """Compute the inverse-decay learning rate."""
    return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter))) ** (-power))


def aToBSheduler(step, A, B, gamma=10, max_iter=10000):
    """Interpolate from A to B with a logistic schedule."""
    ans = A + (2.0 / (1 + np.exp(-gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)
    return float(ans)


def one_hot(n_class, index):
    """Create a NumPy one-hot vector."""
    tmp = np.zeros((n_class,), dtype=np.float32)
    tmp[index] = 1.0
    return tmp


class OptimWithSheduler:
    """Update an optimizer's learning rate using a schedule."""

    def __init__(self, optimizer, scheduler_func):
        self.optimizer = optimizer
        self.scheduler_func = scheduler_func
        self.global_step = 0.0
        for g in self.optimizer.param_groups:
            g['initial_lr'] = g['lr']

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.scheduler_func(
                step=self.global_step, initial_lr=g['initial_lr']
            )
        self.optimizer.step()
        self.global_step += 1


class OptimizerManager:
    """Manage zero_grad and step calls for multiple optimizers."""

    def __init__(self, optims):
        self.optims = optims  # Optimizers managed by this context.

    def __enter__(self):
        for op in self.optims:
            op.zero_grad()

    def __exit__(self, exceptionType, exception, exceptionTraceback):
        for op in self.optims:
            op.step()
        self.optims = None
        if exceptionTraceback:
            print(exceptionTraceback)
            return False
        return True


def setGPU(i):
    """Configure the visible CUDA devices."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % (i)
    gpus = [x.strip() for x in (str(i)).split(',')]
    NGPU = len(gpus)
    print('gpu(s) to be used: %s' % str(gpus))
    return NGPU


class Logger(object):
    """Log scalars to CSV and save visualizations with Python utilities."""

    def __init__(self, log_dir, clear=False):
        # Clear the existing log directory if requested.
        if clear and os.path.exists(log_dir):
            import shutil

            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        self.log_dir = log_dir
        self.step = 0
        self.scalar_log = os.path.join(log_dir, "scalars.csv")

        # Initialize the scalar CSV log.
        with open(self.scalar_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'step', 'tag', 'value'])

    def log_scalar(self, tag, value, step=None):
        """Append a scalar measurement to the CSV log."""
        if step is None:
            step = self.step
        # Advance the default logging step.
        self.step = step + 1

        # Append the scalar value to the CSV log.
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.scalar_log, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, step, tag, value])

    def log_images(self, tag, images, step=None):
        """Save a batch of images to the logging directory."""
        if step is None:
            step = self.step
        self.step = step + 1

        # Create the image output directory.
        img_dir = os.path.join(self.log_dir, "images", tag)
        os.makedirs(img_dir, exist_ok=True)

        # Save each image in the batch.
        for nr, img in enumerate(images):
            # Add a channel dimension for grayscale images.
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
            if img.shape[-1] == 1:
                img = np.tile(img, [1, 1, 3])

            # Save the image for the current step.
            img_path = os.path.join(img_dir, f"step_{step}_img_{nr}.png")
            plt.imsave(img_path, img)

    def log_histogram(self, tag, values, step=None, bins=1000):
        """Save a histogram of the supplied values."""
        if step is None:
            step = self.step
        self.step = step + 1

        # Create the histogram output directory.
        hist_dir = os.path.join(self.log_dir, "histograms")
        os.makedirs(hist_dir, exist_ok=True)

        # Plot and save the histogram.
        plt.figure()
        plt.hist(values, bins=bins)
        plt.title(f"{tag} (step {step})")
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.savefig(os.path.join(hist_dir, f"{tag}_step_{step}.png"))
        plt.close()

    def log_bar(self, tag, values, xs=None, step=None):
        """Save a bar chart of the supplied values."""
        if step is None:
            step = self.step
        self.step = step + 1

        # Create the bar-chart output directory.
        bar_dir = os.path.join(self.log_dir, "bars")
        os.makedirs(bar_dir, exist_ok=True)

        # Plot and save the bar chart.
        plt.figure()
        xs = xs if xs is not None else list(range(len(values)))
        plt.bar(xs, values)
        plt.title(f"{tag} (step {step})")
        plt.xlabel("X")
        plt.ylabel("Value")
        plt.savefig(os.path.join(bar_dir, f"{tag}_step_{step}.png"))
        plt.close()


class AccuracyCounter:
    """Accumulate prediction accuracy across batches."""

    def __init__(self):
        self.Ncorrect = 0.0
        self.Ntotal = 0.0

    def addOntBatch(self, predict, label):
        """Add the prediction and label counts for one batch."""
        predict = variable_to_numpy(predict)
        label = variable_to_numpy(label)

        assert predict.shape == label.shape
        correct_prediction = np.equal(np.argmax(predict, 1), np.argmax(label, 1))
        Ncorrect = np.sum(correct_prediction.astype(np.float32))
        Ntotal = len(label)
        self.Ncorrect += Ncorrect
        self.Ntotal += Ntotal
        return Ncorrect / Ntotal

    def reportAccuracy(self):
        """Return the accumulated accuracy."""
        if self.Ntotal == 0:
            return 0.0
        return np.asarray(self.Ncorrect, dtype=float) / np.asarray(
            self.Ntotal, dtype=float
        )


def CrossEntropyLoss(
    label,
    predict_prob,
    class_level_weight=None,
    instance_level_weight=None,
    epsilon=1e-12,
):
    """Compute weighted multiclass cross-entropy in PyTorch."""
    if predict_prob.numel() == 0 or label.numel() == 0:
        return torch.tensor(0.0, device=predict_prob.device, requires_grad=True)

    N, C = label.size()
    N_, C_ = predict_prob.size()
    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(
                instance_level_weight.size(0), 1
            )
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    ce = -label * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * ce * class_level_weight) / float(N)


def BCELossForMultiClassification(
    label,
    predict_prob,
    class_level_weight=None,
    instance_level_weight=None,
    epsilon=1e-12,
):
    """Compute weighted binary cross-entropy for class probabilities."""
    N, C = label.size()
    N_, C_ = predict_prob.size()
    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(
                instance_level_weight.size(0), 1
            )
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    bce = -label * torch.log(predict_prob + epsilon) - (1.0 - label) * torch.log(
        1.0 - predict_prob + epsilon
    )
    return torch.sum(instance_level_weight * bce * class_level_weight) / float(N)


def EntropyLoss(
    predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-20
):
    """Compute weighted prediction entropy in PyTorch."""
    N, C = predict_prob.size()

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(
                instance_level_weight.size(0), 1
            )
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    entropy = -predict_prob * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * entropy * class_level_weight) / float(N)


def plot_confusion_matrix(
    cm,
    true_classes,
    pred_classes=None,
    normalize=False,
    title='Confusion matrix',
    cmap=plt.cm.Blues,
):
    """Plot a confusion matrix with optional row normalization."""
    import itertools

    pred_classes = pred_classes or true_classes
    if normalize:
        cm = cm.astype(np.float) / np.sum(cm, axis=1, keepdims=True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    true_tick_marks = np.arange(len(true_classes))
    plt.yticks(true_tick_marks, true_classes)
    pred_tick_marks = np.arange(len(pred_classes))
    plt.xticks(pred_tick_marks, pred_classes, rotation=45)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def extended_confusion_matrix(y_true, y_pred, true_labels=None, pred_labels=None):
    """Build a confusion matrix with separate true and predicted label sets."""
    if not true_labels:
        true_labels = sorted(list(set(list(y_true))))
    true_label_to_id = {x: i for (i, x) in enumerate(true_labels)}
    if not pred_labels:
        pred_labels = true_labels
    pred_label_to_id = {x: i for (i, x) in enumerate(pred_labels)}

    confusion_matrix = np.zeros([len(true_labels), len(pred_labels)])
    for true, pred in zip(y_true, y_pred):
        confusion_matrix[true_label_to_id[true]][pred_label_to_id[pred]] += 1.0
    return confusion_matrix
