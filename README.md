# 3D Primitive Autoencoders for Shape Representation

This project explores autoencoder architectures for learning representations of 3D shapes, specifically focusing on decomposing shapes into basic geometric primitives (cubes, spheres, cylinders).

## Project Overview

The project consists of two main parts:

1.  **Voxel-based 3D Convolutional Autoencoder:**
    *   A standard 3D CNN autoencoder that learns to reconstruct 3D shapes represented as voxel grids.
    *   This serves as a baseline and helps in understanding direct voxel manipulation.

2.  **Primitive-based 3D Autoencoder:**
    *   An advanced autoencoder that, instead of directly outputting voxels, predicts parameters for a set of geometric primitives (type, position, size).
    *   A differentiable renderer then uses these parameters to generate a voxel representation of the combined primitives.
    *   The model is trained by comparing the rendered voxels with the original input voxels.
    *   This approach aims for a more structured and interpretable latent representation.

## Core Components

1.  **Shape Generation (`create_cube`, `create_sphere`, `create_cylinder`):**
    *   Python functions using NumPy to generate 32x32x32 voxel grids of basic 3D primitives.
    *   These can generate primitives with varying sizes and dimensions.

2.  **Synthetic Primitives Dataset (`SyntheticPrimitivesDataset`):**
    *   A PyTorch `Dataset` class that generates samples of cubes, spheres, and cylinders.
    *   It can produce primitives with randomized parameters (size, radius, height) for diverse training.
    *   Each sample consists of a voxel grid and its corresponding class label.

3.  **Voxel Autoencoder (`Conv3DAutoencoder`):**
    *   A 3D convolutional neural network designed for compressing and reconstructing voxel grids.
    *   **Encoder:** Reduces the 3D input (1x32x32x32) through convolutional layers to a flat latent vector.
    *   **Decoder:** Reconstructs the voxel grid from the latent vector using 3D transposed convolutional layers.
    *   Uses a Sigmoid activation in the final layer for outputting voxel occupancy probabilities [0,1].
    *   Trained with Binary Cross-Entropy (BCE) loss between the input and reconstructed voxels.

4.  **Primitive Autoencoder (`PrimitiveAutoencoder`):**
    *   **Encoder:** Similar 3D CNN architecture to the Voxel AE, compressing the input voxel grid into a latent vector.
    *   **Parameter Head:** A linear layer that maps the latent vector to a set of parameters for `N` primitives. Each primitive has 9 parameters:
        *   3 for type (logits for cube, sphere, cylinder - passed through Softmax).
        *   3 for position (normalized coordinates [0,1] - passed through Sigmoid).
        *   3 for size (normalized dimensions [0,1] - passed through Sigmoid).
    *   The number of primitives `N` to predict per shape is configurable (e.g., `num_primitives_to_predict=1` for learning to represent single input shapes).

5.  **Differentiable Primitive Renderer (`generate_primitive_voxel_torch`, `primitives_to_voxel_grid_torch`):**
    *   **`generate_primitive_voxel_torch`:** Takes the parameters for a single primitive (type probabilities, normalized position, normalized size) and generates a "soft" voxel grid for that primitive. It uses sigmoid functions to create smooth boundaries, making the process differentiable.
    *   **`primitives_to_voxel_grid_torch`:** Takes the parameters for all `N` predicted primitives for a shape. It calls `generate_primitive_voxel_torch` for each and combines their voxel grids (e.g., using a maximum operation for union) to form the final reconstructed shape.
    *   This renderer allows gradients to flow from the final voxel reconstruction loss back to the predicted primitive parameters.

6.  **Training Loops:**
    *   Standard PyTorch training loops for both autoencoders.
    *   Optimizers (e.g., Adam) and loss functions (e.g., BCELoss for voxel reconstruction) are used.
    *   Includes basic logging of epoch loss.

7.  **Visualization:**
    *   **2D Slices:** `matplotlib` is used to show 2D cross-sections of the 3D voxel grids (original, and reconstructed by the Voxel AE).
    *   **3D Voxel Plots:** `matplotlib.pyplot.voxels` is used for basic 3D rendering of original and reconstructed shapes from the Primitive AE (by binarizing the soft output).
    *   *(Advanced visualization using libraries like `vedo` for mesh rendering and individual primitive display is a potential extension but not covered in the core version here).*


## How to Run

1.  **Prerequisites:**
    *   Python 3.7+
    *   PyTorch (version compatible with your system, >=1.7 recommended, >=1.10 for `torch.meshgrid` with `indexing` argument)
    *   NumPy
    *   Matplotlib
    *   `packaging` (for PyTorch version checking if using the meshgrid compatibility fix)
    ```bash
    pip install torch torchvision torchaudio numpy matplotlib packaging
    ```

2.  **Prepare the Code:**
    *   Organize the provided code blocks into Python files or a Jupyter Notebook .

3.  **Train the Voxel Autoencoder (Optional but Recommended Baseline):**
    *   Run the script/cells corresponding to `train_voxel_ae.py`.
    *   Observe the training loss and visualize some reconstructions.

4.  **Train the Primitive Autoencoder:**
    *   Run the script/cells corresponding to `train_primitive_ae.py`.
    *   This is the core part of the project. Monitor the loss.
    *   Visualize the original and reconstructed shapes. Pay attention to how well the predicted parameters (when rendered) match the input shape.

5.  **Experiment:**
    *   Modify hyperparameters (learning rate, batch size, latent dimension, number of primitives to predict).
    *   Change the `num_samples_per_class` or `num_epochs`.
    *   Analyze the quality of reconstructions and the learned primitive parameters.

## Future Work & Potential Improvements

*   Implement a validation set and early stopping to prevent overfitting.
*   Add more sophisticated evaluation metrics like Intersection over Union (IoU).
*   Experiment with Batch Normalization in the autoencoder architectures.
*   Advanced 3D visualization using libraries like `vedo` or `ipyvolume` to render meshes and individual predicted primitives.
*   Train the `PrimitiveAutoencoder` to decompose more complex shapes made of multiple primitives (requires an appropriate dataset).
*   Supervise primitive parameters directly if ground truth parameters are available.
*   Explore generative modeling using the learned latent spaces.
