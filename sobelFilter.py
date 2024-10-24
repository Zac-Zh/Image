"""
This code implements a Sobel filter for edge detection in images. It supports both grayscale
and RGB images, automatically converting RGB images to grayscale before applying the filter.
The Sobel filter computes horizontal and vertical gradients, as well as the gradient magnitude,
to detect edges in the image. Additional functionality includes:

- Convolution is handled using scipy's optimized 2D convolution.
- Edge detection with optional thresholding.
- Results can be visualized and saved to files.

Usage Example:
    image = np.random.rand(100, 100, 3)  # Example random image
    sobel = SobelFilter(image)
    grad_x, grad_y, magnitude = sobel.compute_gradients()
    sobel.visualize_results(grad_x, grad_y, magnitude, threshold=0.5)
    sobel.save_results(grad_x, grad_y, magnitude, "output")"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class SobelFilter:
    def __init__(self, image: np.ndarray):
        """
        Initialize the SobelFilter class with an image.

        Args:
            image: Input image (grayscale or RGB), 2D or 3D array

        Raises:
            ValueError: If image format is invalid
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array")

        if len(image.shape) not in [2, 3]:
            raise ValueError("Image must be 2D or 3D array")

        # Convert to float32 for better memory efficiency
        image = image.astype(np.float32)

        # Check if the image is RGB (3D array)
        if len(image.shape) == 3 and image.shape[2] == 3:
            self.image = self.rgb_to_grayscale(image)
        else:
            self.image = image

        self.sobel_x = np.array([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]], dtype=np.float32)

        self.sobel_y = np.array([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]], dtype=np.float32)

    def rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB image to grayscale using standard weights."""
        return np.dot(image[..., :3], [0.299, 0.587, 0.114])

    def apply_convolution(self, kernel: np.ndarray) -> np.ndarray:
        """Apply convolution using scipy's optimized implementation."""
        return signal.convolve2d(self.image, kernel, mode='same', boundary='symm')

    def compute_gradients(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute gradients and magnitude."""
        grad_x = self.apply_convolution(self.sobel_x)
        grad_y = self.apply_convolution(self.sobel_y)
        grad_magnitude = np.hypot(grad_x, grad_y)  # More numerically stable than np.sqrt(x**2 + y**2)
        return grad_x, grad_y, grad_magnitude

    def get_edges(self, threshold: float = 0.5) -> np.ndarray:
        """
        Get binary edge image using threshold.

        Args:
            threshold: Value between 0 and 1 for edge detection sensitivity
        """
        _, _, magnitude = self.compute_gradients()
        # Normalize and threshold
        normalized = magnitude / magnitude.max()
        return (normalized > threshold).astype(np.uint8) * 255

    def save_results(self, grad_x: np.ndarray, grad_y: np.ndarray,
                     grad_magnitude: np.ndarray, file_prefix: str):
        """Save results with error handling."""
        try:
            for name, data in [("x", grad_x), ("y", grad_y), ("magnitude", grad_magnitude)]:
                filename = f"{file_prefix}_sobel_{name}.txt"
                np.savetxt(filename, data, fmt='%.2f')
        except IOError as e:
            print(f"Error saving files: {e}")

    def visualize_results(self, grad_x: np.ndarray, grad_y: np.ndarray,
                          grad_magnitude: np.ndarray, threshold: Optional[float] = None):
        """Visualize results with optional thresholded edges."""
        fig, axes = plt.subplots(1, 4 if threshold else 3, figsize=(16 if threshold else 12, 4))

        images = [
            (grad_x, 'Sobel X (Horizontal)'),
            (grad_y, 'Sobel Y (Vertical)'),
            (grad_magnitude, 'Gradient Magnitude')
        ]

        if threshold:
            edges = self.get_edges(threshold)
            images.append((edges, f'Edges (threshold={threshold})'))

        for ax, (img, title) in zip(axes, images):
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example usage with error handling
    try:
        image = np.random.rand(100, 100, 3)  # Example random image
        sobel = SobelFilter(image)
        grad_x, grad_y, magnitude = sobel.compute_gradients()

        # Visualize with thresholded edges
        sobel.visualize_results(grad_x, grad_y, magnitude, threshold=0.5)

        # Save results
        sobel.save_results(grad_x, grad_y, magnitude, "output")

    except Exception as e:
        print(f"Error processing image: {e}")
