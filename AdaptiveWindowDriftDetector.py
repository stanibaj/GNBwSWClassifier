# import numpy as np

# class AdaptiveWindowDriftDetector:
#     def __init__(self, min_width=30, max_width=100, warning_level=2.0, drift_level=3.0):
#         self.min_width = min_width
#         self.max_width = max_width
#         self.warning_level = warning_level
#         self.drift_level = drift_level
#         self.width = self.min_width
#         self.buffer = []
#         self.buffer_sum = 0.0
#         self.buffer_mean = 0.0
#         self.buffer_var = 0.0
#         self.last_change = 0
#         self.detected_warning = False
#         self.detected_drift = False

#     def add_element(self, value):
#         self.buffer.append(value)
#         self.buffer_sum += value
#         n = len(self.buffer)
#         if n > self.width:
#             self.buffer_sum -= self.buffer[0]
#             del self.buffer[0]
#             n -= 1
#         if n == 0:
#             self.buffer_mean = 0.0
#             self.buffer_var = 0.0
#         else:
#             self.buffer_mean = self.buffer_sum / n
#             self.buffer_var = sum([(x - self.buffer_mean) ** 2 for x in self.buffer]) / n

#         # Update window size
#         if n < self.min_width:
#             self.width = self.min_width
#         elif n > self.max_width:
#             self.width = self.max_width
#         else:
#             self.width = int(2.0 * (self.warning_level ** 2) * self.buffer_var / (self.drift_level ** 2))

#         # Check for warning level
#         if not self.detected_warning and n >= self.width and abs(value - self.buffer_mean) > self.warning_level * (self.buffer_var ** 0.5):
#             self.detected_warning = True
#             self.last_change = n
#         # Check for drift level
#         if not self.detected_drift and n >= self.width and abs(value - self.buffer_mean) > self.drift_level * (self.buffer_var ** 0.5):
#             self.detected_drift = True
#             self.last_change = n

#     def detected_warning_zone(self):
#         return self.detected_warning

#     def detected_change(self):
#         return self.detected_drift

#     def reset(self):
#         self.width = self.min_width
#         self.buffer = []
#         self.buffer_sum = 0.0
#         self.buffer_mean = 0.0
#         self.buffer_var = 0.0
#         self.last_change = 0
#         self.detected_warning = False
#         self.detected_drift = False


import numpy as np

class AdaptiveWindowDriftDetector:
    """
        Initialize the AdaptiveWindowDriftDetector class.

        Parameters:
            - min_width (int): Minimum window width.
            - max_width (int): Maximum window width.
            - warning_level (float): Warning level for detecting drift.
            - drift_level (float): Drift level for detecting significant drift.
        """
    
    def __init__(self, min_width=30, max_width=100, warning_level=2.0, drift_level=3.0):
        
        self.min_width = min_width
        self.max_width = max_width
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.width = self.min_width
        self.buffer = []
        self.buffer_sum = 0.0
        self.buffer_mean = 0.0
        self.buffer_var = 0.0
        self.last_change = 0
        self.detected_warning = False
        self.detected_drift = False

    def add_element(self, value):
        """
        Add an element to the drift detector.

        Parameters:
            value: Value to be added.

        Notes:
            - Updates the buffer with the new value.
            - Calculates the buffer's mean and variance.
            - Adjusts the window size based on the buffer's statistics.
            - Checks for warning and drift levels.
        """
        self.buffer.append(value)
        self.buffer_sum += value
        n = len(self.buffer)
        if n > self.width:
            self.buffer_sum -= self.buffer[0]
            del self.buffer[0]
            n -= 1
        if n == 0:
            self.buffer_mean = 0.0
            self.buffer_var = 0.0
        else:
            self.buffer_mean = self.buffer_sum / n
            self.buffer_var = sum([(x - self.buffer_mean) ** 2 for x in self.buffer]) / n

        # Update window size
        if n < self.min_width:
            self.width = self.min_width
        elif n > self.max_width:
            self.width = self.max_width
        else:
            self.width = int(2.0 * (self.warning_level ** 2) * self.buffer_var / (self.drift_level ** 2))

        # Check for warning level
        if not self.detected_warning and n >= self.width and abs(value - self.buffer_mean) > self.warning_level * (self.buffer_var ** 0.5):
            self.detected_warning = True
            self.last_change = n
        # Check for drift level
        if not self.detected_drift and n >= self.width and abs(value - self.buffer_mean) > self.drift_level * (self.buffer_var ** 0.5):
            self.detected_drift = True
            self.last_change = n

    def detected_warning_zone(self):
        """
        Check if a warning zone has been detected.

        Returns:
            bool: True if a warning zone has been detected, False otherwise.
        """
        return self.detected_warning

    def detected_change(self):
        """
        Check if a significant drift has been detected.

        Returns:
            bool: True if a significant drift has been detected, False otherwise.
        """
        return self.detected_drift

    def reset(self):
        """
        Reset the drift detector to its initial state.
        """
        self.width = self.min_width
        self.buffer = []
        self.buffer_sum = 0.0
        self.buffer_mean = 0.0
        self.buffer_var = 0.0
        self.last_change = 0
        self.detected_warning = False
        self.detected_drift = False
