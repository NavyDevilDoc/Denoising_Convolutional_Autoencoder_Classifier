import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist, cifar10
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (Add, Dropout, BatchNormalization,
                                     ReLU, Conv2DTranspose, Input,
                                     Conv2D, MaxPooling2D, UpSampling2D,
                                     Flatten, Dense, Reshape, Concatenate,
                                     Cropping2D)
import warnings
import os


warnings.filterwarnings('ignore')

def print_layer_shapes(model):
    for layer in model.layers:
        print(f"{layer.name}: {layer.output_shape}")

# Plot predictions with true and predicted labels
def plot_denoised_predictions_with_probs(x_test_denoised, predictions, class_names):
    indices = np.random.choice(len(x_test_denoised), 10, replace=False)

    num_channels = x_test_denoised.shape[-1]

    for index in indices:
        plt.figure(figsize=(12, 4))

        # Displaying the reconstructed (denoised) image
        plt.subplot(1, 2, 1)

        if num_channels == 1:
            plt.imshow(x_test_denoised[index].reshape(input_img.shape[1], input_img.shape[2]), cmap='gray')
        else:
            plt.imshow(x_test_denoised[index].reshape(input_img.shape[1], input_img.shape[2], num_channels))

        plt.title(class_names[np.argmax(predictions[index])])
        plt.axis('off')

        # Displaying the class probabilities
        plt.subplot(1, 2, 2)
        plt.bar(class_names, predictions[index], alpha=0.7)
        plt.ylabel('Probability')
        plt.title('Class Probabilities')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()


def visualize_images(original, noisy, denoised, dataset_name, indices_to_display):
    '''Visualizes original, noisy, and denoised images side by side.'''

    fig, axes = plt.subplots(len(indices_to_display), 3, figsize=(10, 4 * len(indices_to_display)))

    for i, idx in enumerate(indices_to_display):
        # Original Image
        if dataset_name == "fashion_mnist":
            axes[i, 0].imshow(original[idx].reshape(28, 28), cmap='gray')
        elif dataset_name == "cifar10":
            axes[i, 0].imshow(original[idx])
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')

        # Noisy Image
        if dataset_name == "fashion_mnist":
            axes[i, 1].imshow(noisy[idx].reshape(28, 28), cmap='gray')
        elif dataset_name == "cifar10":
            axes[i, 1].imshow(noisy[idx])
        axes[i, 1].set_title('Noisy')
        axes[i, 1].axis('off')

        # Denoised Image
        if dataset_name == "fashion_mnist":
            axes[i, 2].imshow(denoised[idx].reshape(28, 28), cmap='gray')
        elif dataset_name == "cifar10":
            axes[i, 2].imshow(denoised[idx])
        axes[i, 2].set_title('Denoised')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()


# Calculate the MSE and SSIM for a set of images
def compare_images(original, reconstructed):
    mse_values = []
    ssim_values = []

    for orig, recon in zip(original, reconstructed):
        mse_values.append(np.mean((orig - recon) ** 2))

        # For CIFAR-10 or any other color dataset
        if orig.shape[-1] == 3:
            s = ssim(orig, recon, multichannel=True)
        # For grayscale datasets like Fashion MNIST
        else:
            s = ssim(orig, recon, win_size=5)  # or you can omit the win_size if not needed

        ssim_values.append(s)

    return mse_values, ssim_values


# Load and preprocess the data
def load_data(dataset_name):
    if dataset_name == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    elif dataset_name == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.reshape(-1, 32, 32, 3)
        x_test = x_test.reshape(-1, 32, 32, 3)
        class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
                       'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    return x_train, y_train, x_test, y_test, class_names


# Select the dataset; either "fashion_mnist" or "cifar10"
dataset_name = "fashion_mnist"

# Split the data
x_train, y_train, x_test, y_test, class_names = load_data(dataset_name)

# Add noise
noise_factor = 0.2
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0,
                                                          scale=1.0,
                                                          size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0,
                                                        scale=1.0,
                                                        size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


# Constants for model paths, early stopping, and model checkpoints
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
if dataset_name == "cifar10":
    input_img = Input(shape=(32, 32, 3))
    channels = 3
    UNET_PATH = 'best_unet_cifar.h5'
    CLASSIFIER_PATH = 'best_classifier_cifar.h5'
    checkpoint_autoencoder = ModelCheckpoint('best_unet_cifar.h5',
                                             monitor='val_loss',
                                             verbose=1,
                                             save_best_only=True)
    checkpoint_classifier = ModelCheckpoint('best_classifier_cifar.h5',
                                            monitor='val_loss',
                                            verbose=1,
                                            save_best_only=True)
elif dataset_name == "fashion_mnist":
    input_img = Input(shape=(28, 28, 1))
    channels = 1
    UNET_PATH = 'best_unet_fashion_mnist.h5'
    CLASSIFIER_PATH = 'best_classifier_fashion_mnist.h5'
    checkpoint_autoencoder = ModelCheckpoint('best_unet_fashion_mnist.h5',
                                             monitor='val_loss',
                                             verbose=1,
                                             save_best_only=True)
    checkpoint_classifier = ModelCheckpoint('best_classifier_fashion_mnist.h5',
                                            monitor='val_loss',
                                            verbose=1,
                                            save_best_only=True)


# Contracting Path (Encoder)
x1 = Conv2D(32 if dataset_name == 'cifar10' else 28, 
            (3, 3), padding='same')(input_img)
x1 = Conv2D(32 if dataset_name == 'cifar10' else 28, 
            (3, 3), padding='same')(x1)
x1 = Conv2D(32 if dataset_name == 'cifar10' else 28, 
            (3, 3), padding='same')(x1)
x1 = BatchNormalization()(x1)
x1 = ReLU()(x1)
x2 = MaxPooling2D((2, 2), padding='same')(x1)

x2 = Conv2D(64, (3, 3), padding='same')(x2)
x2 = Conv2D(64, (3, 3), padding='same')(x2)
x2 = Conv2D(64, (3, 3), padding='same')(x2)
x2 = BatchNormalization()(x2)
x2 = ReLU()(x2)
x3 = MaxPooling2D((2, 2), padding='same')(x2)

x3 = Conv2D(128, (3, 3), padding='same')(x3)
x3 = Conv2D(128, (3, 3), padding='same')(x3)
x3 = Conv2D(128, (3, 3), padding='same')(x3)
x3 = BatchNormalization()(x3)
x3 = ReLU()(x3)
x4 = MaxPooling2D((2, 2), padding='same')(x3)

x4 = Conv2D(256, (3, 3), padding='same')(x4)
x4 = Conv2D(256, (3, 3), padding='same')(x4)
x4 = Conv2D(256, (3, 3), padding='same')(x4)
x4 = BatchNormalization()(x4)
x4 = ReLU()(x4)
x5 = MaxPooling2D((2, 2), padding='same')(x4)

x5 = Conv2D(512, (3, 3), padding='same')(x5)
x5 = Conv2D(512, (3, 3), padding='same')(x5)
x5 = Conv2D(512, (3, 3), padding='same')(x5)
x5 = BatchNormalization()(x5)
x5 = ReLU()(x5)
encoded = MaxPooling2D((2, 2), padding='same')(x5)

# Expanding Path (Decoder)
x6 = Conv2DTranspose(512, (3, 3), strides=2, padding='same')(encoded)
x6 = Concatenate()([x6, x5]) # Skip connection
x6 = BatchNormalization()(x6)
x6 = ReLU()(x6)

x7 = Conv2DTranspose(256, (3, 3), strides=2, padding='same')(x6)
x7 = Concatenate()([x7, x4]) # Skip connection
x7 = BatchNormalization()(x7)
x7 = ReLU()(x7)

x8 = Conv2DTranspose(128, (3, 3), strides=2, padding='same')(x7)
if dataset_name == 'fashion_mnist':
    # Add a cropping layer
    x8_cropped = Cropping2D(cropping=((1, 0), (1, 0)))(x8)
    x8 = Concatenate()([x8_cropped, x3]) # Skip connection
else:
    x8 = Concatenate()([x8, x3]) # Skip connection
x8 = BatchNormalization()(x8)
x8 = ReLU()(x8)

x9 = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x8)
x9 = Concatenate()([x9, x2]) # Skip connection
x9 = BatchNormalization()(x9)
x9 = ReLU()(x9)

x10 = Conv2DTranspose(32, (3, 3), strides=2, padding='same')(x9)
x10 = BatchNormalization()(x10)
x10 = ReLU()(x10)
channels = 1 if dataset_name == "fashion_mnist" else 3
decoded = Conv2D(channels, (3, 3), activation='sigmoid', padding='same')(x10)

# Create U-Net Model
unet = tf.keras.Model(input_img, decoded)
unet.summary()
unet.compile(optimizer='adam', loss='binary_crossentropy')

'''
print("U-Net Shapes:")
print_layer_shapes(unet)
print("--------------")
'''

# Create a CNN classifier using encoded features
x = Conv2D(32, (3, 3), padding='same')(decoded)
x = Conv2D(32, (3, 3), padding='same')(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(64, (3, 3), padding='same')(decoded)
x = Conv2D(64, (3, 3), padding='same')(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(128, (3, 3), padding='same')(x)
x = Conv2D(128, (3, 3), padding='same')(x)
x = Conv2D(128, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(256, (3, 3), padding='same')(x)
x = Conv2D(256, (3, 3), padding='same')(x)
x = Conv2D(256, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(128, activation='relu')(x)
classify = Dense(10, activation='softmax')(x)

classifier = tf.keras.Model(input_img, classify)
classifier.summary()
print("Classifier Shapes:")
print_layer_shapes(classifier)
classifier.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])


# Load or Train the U-Net
unet_hist = unet.fit(
    x_train_noisy, x_train,
    epochs=100, batch_size=64,
    shuffle=True, validation_data=(x_test_noisy, x_test),
    callbacks=[early_stopping, checkpoint_autoencoder]
)


# Use the encoder to get the denoised image
x_train_denoised = unet.predict(x_train_noisy)
x_test_denoised = unet.predict(x_test_noisy)


# Load or Train the CNN classifier
classifier_hist = classifier.fit(
    x_train_denoised, y_train,
    epochs=100, batch_size=64,
    shuffle=True, validation_data=(x_test_denoised, y_test),
    callbacks=[early_stopping, checkpoint_classifier]
)


# Only plot if the training history exists
if unet_hist is not None:
    plt.plot(unet_hist.history['loss'], label='Train')
    plt.plot(unet_hist.history['val_loss'], label='Test')
    plt.title('Autoencoder Loss')
    plt.legend()
    plt.show()

if classifier_hist is not None:
    plt.plot(classifier_hist.history['accuracy'], label='Train')
    plt.plot(classifier_hist.history['val_accuracy'], label='Test')
    plt.title('Classifier Accuracy')
    plt.legend()
    plt.show()


# Display original, noisy, and denoised images
n = 10
plt.figure(figsize=(20, 6))

num_channels = channels

for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    if num_channels == 1:
        plt.imshow(x_test[i].reshape(input_img.shape[1], input_img.shape[2]), cmap='gray')
    else:
        plt.imshow(x_test[i].reshape(input_img.shape[1], input_img.shape[2], num_channels))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + 1 + n)
    if num_channels == 1:
        plt.imshow(x_test_noisy[i].reshape(input_img.shape[1], input_img.shape[2]), cmap='gray')
    else:
        plt.imshow(x_test_noisy[i].reshape(input_img.shape[1], input_img.shape[2], num_channels))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + 1 + 2*n)
    if num_channels == 1:
        plt.imshow(x_test_denoised[i].reshape(input_img.shape[1], input_img.shape[2]), cmap='gray')
    else:
        plt.imshow(x_test_denoised[i].reshape(input_img.shape[1], input_img.shape[2], num_channels))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()


# Predictions
predictions = classifier.predict(x_test_denoised)
y_pred = classifier.predict(x_test_denoised)

# Call the function to plot
plot_denoised_predictions_with_probs(x_test_denoised, predictions, class_names)

# Calculate the metrics
mse_values, ssim_values = compare_images(x_test, x_test_denoised)

# MSE and SSIM Visualization
n = 10
indices = np.random.choice(len(x_test), n, replace=False)

plt.figure(figsize=(25, 8))
for i, idx in enumerate(indices):
    ax = plt.subplot(2, n, i + 1)

    if channels == 1:
        plt.imshow(x_test[idx].reshape(input_img.shape[1], input_img.shape[2]), cmap='gray')
    else:
        plt.imshow(x_test[idx].reshape(input_img.shape[1], input_img.shape[2], channels))

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(f"Original")

    ax = plt.subplot(2, n, i + 1 + n)

    if channels == 1:
        plt.imshow(x_test_denoised[idx].reshape(input_img.shape[1], input_img.shape[2]), cmap='gray')
    else:
        plt.imshow(x_test_denoised[idx].reshape(input_img.shape[1], input_img.shape[2], channels))

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(f"Recon. MSE: {mse_values[idx]:.2f}\nSSIM: {ssim_values[idx]:.2f}")

plt.tight_layout()
plt.subplots_adjust(hspace=0.5) # Adjust the horizontal space
plt.show()


# Later in your script, after training your model and getting the denoised images:
indices_to_display = [0, 5, 10, 15]  # or any indices of interest
visualize_images(x_test, x_test_noisy, x_test_denoised, dataset_name, indices_to_display)


# Ensure the images are in the range [0, 1]
original = x_test
reconstructed = x_test_denoised

# Flatten the datasets to compute the metrics
original_flattened = original.reshape(original.shape[0], -1)
reconstructed_flattened = reconstructed.reshape(reconstructed.shape[0], -1)

# Compute mean squared error for all images
mse_total = mean_squared_error(original_flattened, reconstructed_flattened)

# Compute SSIM for all images (requires iterating through images)
ssim_values = []
for i in range(original.shape[0]):
    if dataset_name == "fashion_mnist":
        s, _ = ssim(original[i, :, :, 0], reconstructed[i, :, :, 0], 
                    full=True, 
                    data_range=1.0,
                    win_size=5)
    elif dataset_name == "cifar10":
        s_r, _ = ssim(original[i, :, :, 0], reconstructed[i, :, :, 0], full=True, data_range=1.0)
        s_g, _ = ssim(original[i, :, :, 1], reconstructed[i, :, :, 1], full=True, data_range=1.0)
        s_b, _ = ssim(original[i, :, :, 2], reconstructed[i, :, :, 2], full=True, data_range=1.0)
        s = (s_r + s_g + s_b) / 3  # Take the average SSIM of three channels
    ssim_values.append(s)

ssim_total = np.mean(ssim_values)

print(f"Mean Squared Error (Total Dataset): {mse_total:.3f}")
print(f"Structural Similarity Index (Total Dataset): {ssim_total:.3f}")

