"""
Haar Wavelet Transform Class

This module implements the Haar Wavelet Transform for both 1D signals and 2D images.
It supports multi-level decomposition and reconstruction, making it useful for applications
such as image compression, denoising, and feature extraction. The class provides methods
for both forward and inverse wavelet transforms, along with a visualization tool for
viewing wavelet decomposition results.


Note: The input dimensions for the 2D image must be powers of 2 for the Haar wavelet transform to work correctly.

"""



import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt


class HaarWavelet:
    """
    A class implementing the Haar Wavelet Transform for both 1D signals and 2D images.
    Supports multi-level decomposition and reconstruction.
    """

    def __init__(self, signal_or_image: np.ndarray):
        """
        Initialize the HaarWavelet transformer with input data.

        Args:
            signal_or_image: Input signal (1D array) or image (2D array)
        """
        self.data = np.array(signal_or_image, dtype=np.float32)
        self.original_shape = self.data.shape

        # Verify input dimensions
        if len(self.original_shape) not in [1, 2]:
            raise ValueError("Input must be 1D signal or 2D image")

        # Verify input size is power of 2
        if not all(s & (s - 1) == 0 for s in self.original_shape):
            raise ValueError("Input dimensions must be powers of 2")

    def _transform_1d(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one level of Haar wavelet transform on a 1D signal.

        Args:
            signal: 1D input array

        Returns:
            Tuple of (approximation coefficients, detail coefficients)
        """
        N = len(signal)
        avg = (signal[0::2] + signal[1::2]) / np.sqrt(2)
        diff = (signal[0::2] - signal[1::2]) / np.sqrt(2)
        return avg, diff

    def _inverse_transform_1d(self, avg: np.ndarray, diff: np.ndarray) -> np.ndarray:
        """
        Perform one level of inverse Haar wavelet transform on 1D coefficients.

        Args:
            avg: Approximation coefficients
            diff: Detail coefficients

        Returns:
            Reconstructed signal
        """
        reconstructed = np.zeros(len(avg) * 2, dtype=np.float32)
        reconstructed[0::2] = (avg + diff) / np.sqrt(2)
        reconstructed[1::2] = (avg - diff) / np.sqrt(2)
        return reconstructed

    def decompose_1d(self, levels: int = 1) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Perform multi-level 1D Haar wavelet decomposition.

        Args:
            levels: Number of decomposition levels

        Returns:
            List of (approximation, detail) coefficient pairs for each level
        """
        if len(self.data.shape) != 1:
            raise ValueError("Input must be 1D for 1D decomposition")

        coefficients = []
        current = self.data.copy()

        for _ in range(levels):
            if len(current) < 2:
                break
            avg, diff = self._transform_1d(current)
            coefficients.append((avg, diff))
            current = avg

        return coefficients

    def decompose_2d(self, levels: int = 1) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Perform multi-level 2D Haar wavelet decomposition.

        Args:
            levels: Number of decomposition levels

        Returns:
            List of (LL, LH, HL, HH) coefficient arrays for each level
        """
        if len(self.data.shape) != 2:
            raise ValueError("Input must be 2D for 2D decomposition")

        coefficients = []
        current = self.data.copy()

        for _ in range(levels):
            if min(current.shape) < 2:
                break

            # Transform rows
            rows_L = np.zeros((current.shape[0], current.shape[1] // 2))
            rows_H = np.zeros((current.shape[0], current.shape[1] // 2))

            for i in range(current.shape[0]):
                rows_L[i], rows_H[i] = self._transform_1d(current[i])

            # Transform columns
            LL = np.zeros((current.shape[0] // 2, current.shape[1] // 2))
            LH = np.zeros((current.shape[0] // 2, current.shape[1] // 2))
            HL = np.zeros((current.shape[0] // 2, current.shape[1] // 2))
            HH = np.zeros((current.shape[0] // 2, current.shape[1] // 2))

            for j in range(rows_L.shape[1]):
                LL[:, j], LH[:, j] = self._transform_1d(rows_L[:, j])
                HL[:, j], HH[:, j] = self._transform_1d(rows_H[:, j])

            coefficients.append((LL, LH, HL, HH))
            current = LL

        return coefficients

    def reconstruct_2d(self, coefficients: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Reconstruct 2D image from wavelet coefficients.

        Args:
            coefficients: List of coefficient tuples from decompose_2d()

        Returns:
            Reconstructed image
        """
        current = coefficients[0][0].copy()  # Start with the lowest frequency band

        for LL, LH, HL, HH in reversed(coefficients):
            # Reconstruct columns
            rows_L = np.zeros((LL.shape[0] * 2, LL.shape[1]))
            rows_H = np.zeros((LL.shape[0] * 2, LL.shape[1]))

            for j in range(LL.shape[1]):
                rows_L[:, j] = self._inverse_transform_1d(LL[:, j], LH[:, j])
                rows_H[:, j] = self._inverse_transform_1d(HL[:, j], HH[:, j])

            # Reconstruct rows
            current = np.zeros((rows_L.shape[0], rows_L.shape[1] * 2))
            for i in range(rows_L.shape[0]):
                current[i] = self._inverse_transform_1d(rows_L[i], rows_H[i])

        return current

    def visualize_decomposition(self, coefficients: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
        """
        Visualize 2D wavelet decomposition results.

        Args:
            coefficients: List of coefficient tuples from decompose_2d()
        """
        # Create composite image
        composite = np.zeros_like(self.data)
        current_size = (self.data.shape[0] // 2, self.data.shape[1] // 2)

        for level, (LL, LH, HL, HH) in enumerate(coefficients):
            # Calculate position for this level
            pos_y = 0 if level == 0 else self.data.shape[0] // 2 ** (level)
            pos_x = 0 if level == 0 else self.data.shape[1] // 2 ** (level)
            size = current_size[0] // (2 ** level), current_size[1] // (2 ** level)

            if level == 0:
                composite[:size[0], :size[1]] = LL
                composite[:size[0], size[1]:size[1] * 2] = LH
                composite[size[0]:size[0] * 2, :size[1]] = HL
                composite[size[0]:size[0] * 2, size[1]:size[1] * 2] = HH

        plt.figure(figsize=(10, 10))
        plt.imshow(composite, cmap='gray')
        plt.title('Wavelet Decomposition')
        plt.colorbar()
        plt.show()


# Example usage:
if __name__ == "__main__":
    # Create sample 2D image (8x8)
    sample_image = np.array([
        [100, 100, 100, 100, 50, 50, 50, 50],
        [100, 100, 100, 100, 50, 50, 50, 50],
        [100, 100, 100, 100, 50, 50, 50, 50],
        [100, 100, 100, 100, 50, 50, 50, 50],
        [25, 25, 25, 25, 75, 75, 75, 75],
        [25, 25, 25, 25, 75, 75, 75, 75],
        [25, 25, 25, 25, 75, 75, 75, 75],
        [25, 25, 25, 25, 75, 75, 75, 75]
    ], dtype=np.float32)

    # Create transformer
    transformer = HaarWavelet(sample_image)

    # Perform 2-level decomposition
    coefficients = transformer.decompose_2d(levels=2)

    # Visualize results
    transformer.visualize_decomposition(coefficients)

    # Reconstruct image
    reconstructed = transformer.reconstruct_2d(coefficients)

    # Verify reconstruction
    print("Max reconstruction error:", np.max(np.abs(sample_image - reconstructed)))