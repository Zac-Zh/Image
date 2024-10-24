import numpy as np


class SobelFilter:
    def __init__(self, image):
        """
        Initialize the SobelFilter class with an image.
        :param image: Input grayscale image, 2D array
        """
        self.image = image
        self.sobel_x = np.array([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]])

        self.sobel_y = np.array([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]])

    def apply_convolution(self, kernel):
        """
        Apply convolution operation to the image with the given kernel.
        :param kernel: Convolution kernel, 2D array
        :return: Convolved image
        """
        img_height, img_width = self.image.shape
        kernel_size = kernel.shape[0]
        pad = kernel_size // 2  # Padding size for the borders

        # Pad the image with zeros to handle borders
        padded_image = np.pad(self.image, ((pad, pad), (pad, pad)), mode='constant')

        # Initialize the result of the convolution
        convolved_image = np.zeros_like(self.image)

        # Perform the convolution operation
        for i in range(img_height):
            for j in range(img_width):
                # Extract the current 3x3 region
                region = padded_image[i:i + kernel_size, j:j + kernel_size]
                # Convolution operation (element-wise multiplication and sum)
                convolved_image[i, j] = np.sum(region * kernel)

        return convolved_image

    def compute_gradients(self):
        """
        Compute the gradients in the horizontal and vertical directions, as well as the gradient magnitude.
        :return: Horizontal gradient image, vertical gradient image, gradient magnitude image
        """
        grad_x = self.apply_convolution(self.sobel_x)
        grad_y = self.apply_convolution(self.sobel_y)
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return grad_x, grad_y, grad_magnitude

    def save_results(self, grad_x, grad_y, grad_magnitude, file_prefix):
        """
        Save the results of the Sobel filter to files.
        :param grad_x: Horizontal gradient image
        :param grad_y: Vertical gradient image
        :param grad_magnitude: Gradient magnitude image
        :param file_prefix: Prefix for the filenames
        """
        np.savetxt(f"{file_prefix}_sobel_x.txt", grad_x, fmt='%d')
        np.savetxt(f"{file_prefix}_sobel_y.txt", grad_y, fmt='%d')
        np.savetxt(f"{file_prefix}_sobel_magnitude.txt", grad_magnitude, fmt='%d')


# Example usage:
if __name__ == "__main__":
    # Assume image is a 2D array representing a grayscale image
    image = np.array([[...]])  # Replace with actual image data

    # Create a SobelFilter object
    sobel_filter = SobelFilter(image)

    # Compute the gradients
    grad_x, grad_y, grad_magnitude = sobel_filter.compute_gradients()

    # Save the results to files
    sobel_filter.save_results(grad_x, grad_y, grad_magnitude, "output")

    print("Sobel filter results have been saved.")
