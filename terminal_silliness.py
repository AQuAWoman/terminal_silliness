import numpy as np
import torch
from torch import nn
import os
import time
import subprocess  # For running FFmpeg
import shutil
import math
import sys
import requests  # For downloading images
from PIL import Image  # For image loading
from kernels import Kernels

class ImageConvolution:
    def __init__(self):
        self.kernels = Kernels().kernels
        self.kernel_height, self.kernel_width = self.kernels[0][0].shape
        self.char_map = {i: x[1] for i, x in enumerate(self.kernels)}

        # Create character and kernel map tensors
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.char_tensor = torch.tensor([ord(c) for c in self.char_map.values()], dtype=torch.int32).to(self.device)
        self.kernel_map_tensor = torch.from_numpy(np.array([x[0] for x in self.kernels])).to(self.device)

    def convolve_image(self, image_path, target_width=None, color_codes=None):
        print(f"Starting image convolution from: {image_path}")
        start_time = time.time()

        # Download image if it's a URL
        if image_path.startswith("http"):
            image_path = self.download_image(image_path)
            if not image_path:
                return  # Exit if download failed

        # Load the image using PIL
        try:
            img = Image.open(image_path).convert("L")  # Convert to grayscale
        except FileNotFoundError:
            print(f"\033[1;91mError: Image not found at {image_path}\033[0m")
            return

        # Convert the image to a NumPy array
        frame = np.array(img)

        # Get terminal width
        if target_width is None or target_width == 0:
            width, height = os.get_terminal_size()
            target_width = width

        # Calculate the target height
        kernel_aspect_ratio = self.kernel_height / self.kernel_width
        original_aspect_ratio = frame.shape[0] / frame.shape[1]
        target_height = int(target_width * self.kernel_height * original_aspect_ratio / kernel_aspect_ratio)

        # Ensure the target height is a multiple of kernel height
        target_height = ((target_height // self.kernel_height) + 1) * self.kernel_height

        # Rescale the image using Pillow with NEAREST neighbor resampling
        img = Image.fromarray(frame)
        img = img.resize((target_width * self.kernel_width, target_height), Image.Resampling.NEAREST)
        frame = np.array(img)

        # Split the image into chunks
        reconstructed_string = ""
        char_count = 0  # Track character count within each line

        # Remove this line: char_count = 0  # Reset char_count for the next line
        for y in range(0, frame.shape[0], self.kernel_height):
            # Extract the chunk
            chunk = frame[y:y + self.kernel_height, :]

            # Convolve the chunk
            chunk_string, char_count = self.convolve_chunk(chunk, color_codes, char_count)

            # Append the chunk string to the reconstructed string
            reconstructed_string += chunk_string + "\n"

        print(reconstructed_string)

        end_time = time.time()
        print(f"Image convolution completed in {end_time - start_time:.4f} seconds.")

    def convolve_chunk(self, chunk, color_codes, char_count):
        # Optimized block processing (GPU)
        input_img_tensor = torch.from_numpy(chunk).to(self.device)
        kernel_map_tensor = self.kernel_map_tensor.to(self.device)
        kernel_map_tensor = kernel_map_tensor.transpose(1,2)

        block_height, block_width = self.kernel_height, self.kernel_width
        input_img_tensor = input_img_tensor.unfold(0, block_height, block_height)
        input_img_tensor = input_img_tensor.reshape(-1, block_width, block_height)

        sad_values = torch.sum(torch.abs(kernel_map_tensor.unsqueeze(0) - input_img_tensor.unsqueeze(1)), dim=(2, 3))
        best_kernel_indices = torch.argmin(sad_values, dim=1)
        kernel_map = best_kernel_indices.reshape(
            chunk.shape[1] // self.kernel_width
        )

        # Reconstruct string with GPU
        chunk_string = self.reconstruct_string_gpu(kernel_map, self.char_tensor)

        colored_chunk = ""
        # Apply colors based on code combinations
        if color_codes is not None:
            for i, char in enumerate(chunk_string):
                if char == "\n":
                    continue  # Skip newlines for color counting
                color_code_index = char_count % len(color_codes)
                color_code_parts = color_codes[color_code_index].split("-")

                # Build combined ANSI escape sequence
                ansi_code = "\033["

                # Initialize variables for foreground and background colors
                foreground_color = None
                background_color = None

                # Process color code parts
                for j, part in enumerate(color_code_parts):
                    if part.isdigit() and foreground_color is None:
                        foreground_color = int(part)
                        if foreground_color < 0 or foreground_color > 15:
                            raise ValueError(f"Invalid foreground color: {foreground_color}")
                    elif part.isdigit() and background_color is None:
                        background_color = int(part)
                        if background_color < 0 or background_color > 15:
                            raise ValueError(f"Invalid background color: {background_color}")
                    elif part.isdigit() and (foreground_color is not None or background_color is not None):
                        raise ValueError(f"Invalid color code: {color_codes[color_code_index]} - Extra number found after color codes")
                    elif part in ["I", "B", "N"]:
                        if foreground_color is None:
                            foreground_color = 39 # Default foreground (white)
                        if background_color is None:
                            background_color = 49 # Default background (black)
                        if part == "I":
                            ansi_code += ";3"  # Italic
                        elif part == "B":
                            ansi_code += ";1"  # Bold
                        elif part == "N":
                            ansi_code += ";7"  # Invert
                    elif part.isdigit():
                        raise ValueError(f"Invalid color code: {color_codes[color_code_index]} - Numbers after style codes are not allowed.")

                # Apply the colors to the ANSI code if they were set
                if foreground_color is not None:
                    if foreground_color >= 8 and foreground_color <= 15:
                        ansi_code += f"9{foreground_color - 8}"  # High-intensity foreground
                    else:
                        ansi_code += f"3{foreground_color}"  # Foreground (0-7)
                if background_color is not None:
                    if background_color >= 8 and background_color <= 15:
                        ansi_code += f";10{background_color - 8}"  # High-intensity background
                    else:
                        ansi_code += f";4{background_color}"  # Background (0-7)

                ansi_code += "m"  # Add 'm' at the end of the combined sequence

                colored_chunk += ansi_code + char + "\033[0m"  # Add the character and reset color
                char_count += 1

        # No need to reset color here as it's already done after each character
        # if color_codes is not None:
        #    colored_chunk += "\033[0m"  # Reset color at the end of the line

        # Return both chunk_string and char_count
        return colored_chunk, char_count

    def download_image(self, url):
        """Downloads an image from a URL and saves it to a temporary file."""
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}  # Identify as a browser
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes
            temp_filename = 'temp_image.png'  # Temporary filename
            with open(temp_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return temp_filename  # Return the temporary filename
        except requests.exceptions.RequestException as e:
            print(f"\033[1;91mError downloading image: {e}\033[0m")
            return None

    def reconstruct_string_gpu(self, kernel_map, char_tensor):
        device = kernel_map.device
        reconstructed_string_tensor = char_tensor[kernel_map]
        reconstructed_string_tensor = reconstructed_string_tensor.flatten()
        reconstructed_string = ''.join(chr(i.item()) for i in reconstructed_string_tensor.cpu().numpy())
        return reconstructed_string

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python image_convolution.py <image_path> [target_width] [color_codes...]")
        sys.exit(1)

    image_path = sys.argv[1]
    target_width = int(sys.argv[2]) if len(sys.argv) > 2 else None
    color_codes = sys.argv[3:] if len(sys.argv) > 3 else None

    app = ImageConvolution()
    app.convolve_image(image_path, target_width, color_codes)

    # Clean up the temporary image file if downloaded
    if image_path.startswith("http"):
        os.remove("temp_image.png")