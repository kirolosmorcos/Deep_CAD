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


## Design Choices and Hyperparameter Rationale

The specific architectures and hyperparameters used in this project were chosen based on common practices in 3D deep learning, empirical experimentation typical for autoencoder tasks, and the goal of achieving a balance between model capacity and computational feasibility.

### 1. General Autoencoder Structure

*   **Encoder-Decoder Architecture:** This is the standard paradigm for autoencoders, aiming to learn a compressed representation (latent space) and then reconstruct the original input from it.
*   **Convolutional Layers (3D):** For volumetric data like voxel grids, 3D convolutions are essential as they can capture spatial relationships in all three dimensions. They respect the grid structure of the input.
*   **Downsampling in Encoder / Upsampling in Decoder:**
    *   **Encoder:** Strided convolutions (`stride=2`) are used for downsampling. This progressively reduces the spatial dimensions while increasing the number of feature channels, allowing the network to learn hierarchical features.
    *   **Decoder:** Transposed convolutions (`ConvTranspose3d` with `stride=2`) are used for upsampling, gradually increasing the spatial dimensions to reconstruct the original shape. `output_padding=1` is often used with `kernel_size=3, stride=2, padding=1` in transposed convolutions to achieve a clean doubling of the spatial dimension.
*   **ReLU Activation:** A common, effective, and computationally efficient non-linear activation function used after most convolutional layers to introduce non-linearity into the model.
*   **Sigmoid Activation (Final Layer):** For both autoencoders, the final layer of the decoder uses a Sigmoid activation. This is crucial because:
    *   The input voxel grids represent occupancy (0 for empty, 1 for filled).
    *   Sigmoid squashes the output to the range [0,1], which can be interpreted as voxel occupancy probabilities.
    *   This output range is compatible with Binary Cross-Entropy (BCE) loss.

### 2. Voxel Autoencoder (`Conv3DAutoencoder`)

*   **Channel Progression (e.g., 1 -> 32 -> 64 -> 128):** A common strategy to increase the number of channels (features) as spatial dimensions decrease. This allows the network to learn more complex features in the deeper layers. The specific numbers (32, 64, 128) are typical starting points for 3D tasks and offer a reasonable capacity for a 32x32x32 input.
*   **Latent Dimension (`latent_dim=128`):** This is an empirical choice.
    *   A smaller latent dimension forces more aggressive compression, potentially leading to lossier reconstruction but a more disentangled or meaningful representation if successful.
    *   A larger latent dimension makes reconstruction easier but might lead to a less compressed or "trivial" latent space.
    *   128 is a common starting point for relatively simple shapes on a 32^3 grid. This would be a key hyperparameter to tune.
*   **Kernel Size (`kernel_size=3`):** A common choice for 3D convolutions, balancing receptive field size with computational cost and number of parameters. `padding=1` is used with `kernel_size=3` to maintain spatial dimensions when `stride=1`, or to ensure consistent downsampling with `stride=2`.

### 3. Primitive Autoencoder (`PrimitiveAutoencoder`)

*   **Encoder Architecture:** Similar to the Voxel AE's encoder (e.g., 1 -> 16 -> 32 -> 64 channels). The capacity might be slightly lower because the decoder's task (predicting a few parameters) is arguably simpler than reconstructing an entire voxel grid, though the renderer adds complexity.
*   **Parameter Head (`nn.Linear`):** A fully connected layer is a standard way to map a flat latent vector to a fixed number of output parameters.
*   **Parameter Interpretation (Type, Position, Size):**
    *   **Type (Softmax):** Softmax is the natural choice for multi-class classification (cube, sphere, cylinder), outputting a probability distribution over the types.
    *   **Position & Size (Sigmoid):** Sigmoid outputs values in (0,1). This is useful for representing *normalized* parameters. The renderer then denormalizes these based on the `voxel_dim`. This helps keep the network outputs bounded and can make learning more stable.
*   **`num_primitives_to_predict=1` (Initial Choice):**
    *   For the `SyntheticPrimitivesDataset` which generates single, non-composite shapes, starting with `num_primitives_to_predict=1` is a logical first step. The model learns to represent one input shape with one set of primitive parameters.
    *   Increasing this would be for future work on decomposing more complex shapes.
*   **Differentiable Renderer:**
    *   **Smoothness (`smoothness` parameter in `generate_primitive_voxel_torch`):** Controls the "softness" of the rendered primitive edges. A higher value makes edges sharper. This is important for differentiability, as hard boundaries (like a perfect step function) have zero gradients almost everywhere. Sigmoid functions are used to approximate these hard boundaries smoothly.
    *   **Combination of Primitives (`torch.maximum`):** Using `torch.maximum` to combine voxel grids from multiple primitives implements a union operation, which is a common way to merge shapes.

### 4. Training Hyperparameters

*   **Optimizer (`Adam`):** A widely used and generally effective adaptive learning rate optimization algorithm. It often requires less manual tuning of the learning rate compared to SGD.
*   **Learning Rate (`lr=1e-3`):** A common default starting learning rate for Adam. It's often a good balance between fast convergence and stability. This would be a key parameter to tune (e.g., trying `1e-4`, `5e-4`).
*   **Loss Function (`BCELoss` - Binary Cross-Entropy Loss):**
    *   Appropriate for the voxel reconstruction task where target voxels are binary (0 or 1) and the model's output (after Sigmoid) is a probability [0,1].
    *   It penalizes deviations between the predicted probability and the true binary label.
*   **Batch Size (e.g., 4, 8, 16):**
    *   Limited by GPU memory for 3D data. Smaller batch sizes are common.
    *   Larger batch sizes can provide more stable gradient estimates but require more memory and can sometimes lead to sharper minima (less generalization). Smaller batch sizes introduce more noise, which can sometimes help escape local minima.
    *   The chosen values (e.g., 8 for Voxel AE, 4 for Primitive AE) are practical starting points for 32^3 voxel grids.
*   **Number of Epochs (e.g., 5, 10, 20 for initial tests):**
    *   Highly dependent on the dataset size, model complexity, and learning rate.
    *   For initial development and testing, fewer epochs are used to get quick feedback. For final training, more epochs would be run, ideally with early stopping based on a validation set.

### Summary of Rationale

The design choices prioritize standard and proven techniques for 3D autoencoders while introducing the novel aspect of a differentiable primitive-based decoder. Hyperparameters are set to common initial values suitable for experimentation on the given task and input size, with the understanding that further tuning would be necessary for optimal performance. The aim is to create a system that is learnable, interpretable (especially the Primitive AE), and provides a solid foundation for future extensions.

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


## Comparison to "DeepCAD: A Deep Generative Network for Computer-Aided Design Models"

This project, while sharing an interest in programmatic and structured 3D shape representation, differs significantly in scope and methodology from "DeepCAD: A Deep Generative Network for Computer-Aided Design Models" by Wu, Xiao, and Zheng (Columbia University). DeepCAD focuses on learning a generative model for CAD by predicting sequences of sketch-and-extrude operations, akin to how a human might design a part in CAD software.

Here's a comparison:

### 1. Objectives and Scope:

*   **DeepCAD:**
    *   **Main Goal:** To develop a deep generative model that can automatically synthesize realistic and complex 3D CAD models by predicting a sequence of parameterized 2D sketching and 3D extrusion operations.
    *   **Dataset:** Utilizes a dataset of CAD models (e.g., from Onshape via ABC dataset) represented as sequences of these operations.
    *   **Complexity of Shapes:** Aims to generate fairly complex mechanical parts and designs that have a procedural/programmatic structure.
*   **This Project:**
    *   **Main Goal:** To explore autoencoder architectures for learning representations of simple 3D shapes, with a specific focus on a `PrimitiveAutoencoder` that predicts parameters for a small, fixed set of basic geometric primitives (cubes, spheres, cylinders) and reconstructs the shape via a differentiable voxel renderer.
    *   **Dataset:** `SyntheticPrimitivesDataset` generating isolated, simple 3D primitives with randomized basic parameters.
    *   **Complexity of Shapes:** Limited to individual, non-composite cubes, spheres, and cylinders.

### 2. Methodology Differences:

*   **DeepCAD:**
    *   **Input Representation (for training):** Sequences of CAD operations (e.g., sketch profiles, extrusion types, parameters).
    *   **Network Architecture:** Employs a recurrent neural network (RNN, likely an LSTM or GRU) to model the sequence of operations. It might use PointNet-like or CNN-based encoders for processing sketch information and other auto-regressive components to predict parameters for each step.
    *   **Output Representation:** A sequence of CAD program steps. The final 3D model is implicitly defined by executing this program.
    *   **Loss Functions:** Likely involves a combination of losses, such as cross-entropy for operation types, and regression losses (e.g., L1/L2) for sketch coordinates and extrusion parameters, potentially on the final rendered shape if an end-to-end approach is taken for some parts.
*   **This Project:**
    *   **Input Representation (for training):** 32x32x32 binary voxel grids of simple primitives.
    *   **Network Architecture:**
        *   `Conv3DAutoencoder`: Standard 3D CNN for voxel-to-voxel reconstruction.
        *   `PrimitiveAutoencoder`: 3D CNN encoder followed by a linear head to predict parameters for a fixed number (`num_primitives_to_predict`, e.g., 1) of primitives. No sequential modeling.
    *   **Output Representation (`PrimitiveAutoencoder`):** A fixed set of parameters (type, position, size) for `num_primitives_to_predict` primitives. The final 3D model is explicitly rendered as a voxel grid using a differentiable renderer.
    *   **Loss Functions:** Binary Cross-Entropy (BCELoss) directly on the reconstructed voxel grid compared to the input voxel grid.

### 3. Performance and Evaluation:

*   **DeepCAD:**
    *   **Key Metrics:**
        *   Visual quality and realism of generated CAD models.
        *   Validity of the generated CAD operation sequences (can they be executed by a CAD kernel?).
        *   Diversity of generated shapes.
        *   Potentially, reconstruction accuracy if an autoencoder variant is used, or likelihood scores for generative models. Comparison of geometric properties (e.g., IoU, Chamfer Distance) between generated/reconstructed and ground truth CAD models.
    *   **Reported Results:** Demonstrates the ability to generate novel and complex CAD models that resemble human designs.
*   **This Project:**
    *   **Key Metrics:**
        *   Training loss (BCELoss) to monitor learning progress.
        *   Qualitative visual inspection of the reconstructed voxel grids.
        *   Potential for Intersection over Union (IoU) on the synthetic dataset.
    *   **Expected Results:** Successful reconstruction of the input simple primitives. The `PrimitiveAutoencoder` should learn to output parameters that, when rendered, closely match the input voxel shape. Direct numerical comparison to DeepCAD's metrics is not feasible due to the vast differences in task and output complexity.

### 4. Key Takeaways from Comparison:

*   **DeepCAD's Strengths & Focus:**
    *   **Programmatic Generation:** Captures the procedural nature of CAD design, leading to interpretable and editable outputs (if the sequence is valid).
    *   **Complexity & Realism:** Aims for generating models with a level of complexity and realism seen in actual CAD workflows.
    *   **Sequential Decision Making:** Leverages RNNs to model the step-by-step process of design.
*   **This Project's Strengths & Focus:**
    *   **Differentiable Primitive Fitting:** Explores the idea of directly predicting parameters for 3D primitives and using a differentiable renderer, which is a different paradigm from sequential program synthesis.
    *   **Foundational Exploration:** Serves as an educational tool to understand how an autoencoder can learn to map from raw voxels to a structured, parametric (though simple) representation.
    *   **Fixed-Structure Output:** The `PrimitiveAutoencoder` outputs a fixed number of primitives, unlike DeepCAD's variable-length program.

### 5. How This Project Relates and Differs Fundamentally:

*   **Analogy:**
    *   **DeepCAD** is like learning to write a *recipe* (a sequence of steps) to bake a variety of cakes.
    *   **This Project's `PrimitiveAutoencoder`** is like learning to describe a single, simple object (e.g., "a red ball") by identifying its core type ("sphere") and its properties ("red," "specific size," "specific location") without a sequence of construction steps. If `num_primitives_to_predict > 1`, it's like describing a simple scene with a few known objects.
*   **Abstraction Level:** DeepCAD operates at a higher level of abstraction (CAD operations), while this project (especially the `PrimitiveAutoencoder`) works with lower-level geometric primitives and their direct parameterization as a target representation from voxels.

### Conclusion of Comparison:

This project and DeepCAD tackle 3D shape representation from different angles. DeepCAD pioneers complex, sequential, programmatic generation mirroring human CAD processes. Our work, particularly the `PrimitiveAutoencoder`, explores a more direct mapping from voxel data to a small, fixed set of parametric primitives via a differentiable rendering process. While DeepCAD's ambitions and capabilities are far broader, our project provides a foundational understanding of how neural networks can learn to "see" and parameterize basic geometric components within voxelized shapes, which could be a building block or complementary idea for more complex systems. Future work could explore bridging these ideas, for instance, by having a system predict a sequence of operations where some operations involve placing/parameterizing primitives learned by an autoencoder similar to the one developed here.
