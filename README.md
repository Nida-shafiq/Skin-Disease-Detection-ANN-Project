**Skin Disease Detection Model with Enhanced CNN**
	The skin disease detection model leverages a deep convolutional neural network (CNN) architecture designed to accurately classify skin disease images. The model was trained using an enhanced version of CNN, referred to as EnhancedCNN, which incorporates multiple convolutional blocks, batch normalization, and max-pooling layers to improve the feature extraction process. Here's an overview of the key components:

Model Architecture
Block 1:

	Convolutional Layer: The first layer uses 3 input channels (RGB images) and applies 32 filters with a kernel size of 3x3, ensuring the output feature map has a 		spatial dimension of 32x32x32.
	Batch Normalization & Pooling: Batch normalization is applied for stable learning, followed by max-pooling with a 2x2 kernel, reducing the spatial dimension to 		16x16.
Block 2:

	Convolutional Layer: This block applies 64 filters to the previous output, maintaining the spatial size at 16x16x64 after convolution and batch normalization.
	Pooling: Max-pooling reduces the output size to 8x8.
Block 3:

	Convolutional Layer: The third block expands the depth further to 128 filters, resulting in an output size of 8x8x128.
	Pooling: Max-pooling reduces it to 4x4.
Block 4:

	Convolutional Layer: The fourth block increases the depth to 256 filters, resulting in an output of size 4x4x256.
	Pooling: Max-pooling reduces it to 2x2.
**Fully Connected Layers:**

	Flattened: The feature map is flattened before being passed into fully connected layers.
	FC1: A dense layer with 512 neurons follows, utilizing Leaky ReLU activation to introduce non-linearity.
	Dropout: A dropout layer with a rate of 0.5 helps prevent overfitting.
	FC2: The final fully connected layer outputs predictions for the different classes (skin diseases).

**Training Process**
	The model is trained using cross-entropy loss and optimized with an adaptive optimizer like Adam. The training process involves the following steps:

**Epoch Loop:** The model trains for a specified number of epochs (e.g., 15), iterating through both training and testing phases.
Optimization: During the training phase, gradients are computed and parameters are updated. In the testing phase, only forward propagation occurs to evaluate the model's performance.
**Scheduler:** A learning rate scheduler adjusts the learning rate based on the training progress to ensure more stable convergence.
Best Model Selection: The model's weights are updated when a better performance (accuracy) is observed on the test set.
The model is designed to predict the class label for skin diseases based on the input images, and the objective is to achieve high accuracy in detecting various skin conditions.

**Code Explanation**
The code provided implements the EnhancedCNN architecture and trains the model using PyTorch. It first initializes the CNN architecture and then uses the train_model function to train it over multiple epochs. Here’s a brief breakdown of the code:

**EnhancedCNN Class:**

It defines the convolutional layers, batch normalization, pooling layers, and fully connected layers.
The forward method specifies the forward pass, using Leaky ReLU activation after each convolutional block, followed by max-pooling and a dropout layer in the fully connected part.

**Training Function (train_model):**

This function manages the training and testing phases for each epoch.
It tracks the loss and accuracy for both phases and adjusts the learning rate using the scheduler.
The model's best weights are saved if a higher accuracy is achieved in the testing phase.

**Training Execution**:The model is trained with 15 epochs, using a specified loss function, optimizer, and scheduler. The final model with the best accuracy is returned after training.
This approach helps the model generalize well to unseen skin disease images, improving its ability to classify images accurately.

The model is trained with 15 epochs, using a specified loss function, optimizer, and scheduler. The final model with the best accuracy is returned after training.
This approach helps the model generalize well to unseen skin disease images, improving its ability to classify images accurately.
