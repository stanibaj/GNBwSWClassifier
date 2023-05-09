from collections import deque
import numpy as np

class GaussianNaiveBayesWithSlidingWindow:
    """
    Initializes the GaussianNaiveBayesWithSlidingWindow class.

    Parameters
    ----------
    window_size : int, optional
        The size of the sliding window, by default 100.
    var_smoothing : float, optional
        The value added to the variances for numerical stability, by default 1e-9.

    Returns
    -------
    None

    """
    def __init__(self, window_size=100, var_smoothing=1e-9):
        self.n_classes = 2
        self.window = deque(maxlen=window_size)
        self.var_smoothing = var_smoothing

    def learn_one(self, x, y):
        """
        Updates the model using a single sample.

        Parameters
        ----------
        x : array-like of shape (n_features,)
            The input data.
        y : int
            The target value.

        Returns
        -------
        None

        """
        self.window.append((x, y))
        if len(self.window) == self.window.maxlen:
            X = np.array([i[0] for i in self.window])
            y = np.array([i[1] for i in self.window])
            self.counts = np.zeros(self.n_classes)
            self.mean = np.zeros((self.n_classes, X.shape[1]))
            self.variance = np.zeros((self.n_classes, X.shape[1]))
            for i in range(len(X)):
                self.counts[y[i]] += 1
                self.mean[y[i]] += (X[i] - self.mean[y[i]]) / self.counts[y[i]]
                self.variance[y[i]] += ((X[i] - self.mean[y[i]]) ** 2 - self.variance[y[i]]) / self.counts[y[i]]
            self.variance += self.var_smoothing

    def predict_one(self, x):
        """
        Predicts the class label of a single sample.
        Before calling this method, the model have to be trained with number of data points greater or equal to sliding window size. Otherwise exception is raised.

        Parameters
        ----------
        x : array-like of shape (n_features,)
            The input data.

        Returns
        -------
        int
            The predicted class label.

        """
        if len(self.window) < self.window.maxlen:
            raise Exception("Too few samples used to train this model.")
        class_probs = []
        for c in range(self.n_classes):
            prior = self.counts[c] / sum(self.counts)
            likelihood = np.exp(-0.5 * ((x - self.mean[c]) ** 2 / self.variance[c] + np.log(2 * np.pi * self.variance[c])))
            posterior = prior * np.prod(likelihood)
            class_probs.append(posterior)
        return np.argmax(class_probs)

