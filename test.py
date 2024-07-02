from keras.models import load_model
from numpy import load, vstack
import numpy as np
from matplotlib import pyplot
from numpy.random import randint
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from skimage.io import imsave

# Load and prepare training images
def load_real_samples(filename):
    # Load compressed arrays
    data = load(filename)
    # Unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # Scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]

# Plot source, generated and target images
def plot_images(src_img, gen_img, tar_img):
    images = vstack((src_img, gen_img, tar_img))
    # Scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    titles = ['Source', 'Generated', 'Expected']
    # Plot images row by row
    for i in range(len(images)):
        # Define subplot
        pyplot.subplot(1, 3, 1 + i)
        # Turn off axis
        pyplot.axis('off')
        # Plot raw pixel data
        pyplot.imshow(images[i])
        # Show title
        pyplot.title(titles[i])
    pyplot.show()

# Load dataset
[X1, X2] = load_real_samples('maps_256.npz')
print('Loaded', X1.shape, X2.shape)

# Load model
model = load_model('model_054800.h5')

# Select random example
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]

# Generate image from source
gen_image = model.predict(src_image)

# Plot all three images
plot_images(src_image, gen_image, tar_image)

# Scale from [-1,1] to [0,255]
gen_image = ((gen_image + 1) / 2.0) * 255
tar_image = ((tar_image + 1) / 2.0) * 255

# Convert images to uint8 format
gen_image = gen_image.astype(np.uint8)
tar_image = tar_image.astype(np.uint8)

# Save images to folder path
folder_path = "./Output/Trained/"
gen_filename = "gen_image.png"
tar_filename = "tar_image.png"
imsave(folder_path + "/" + gen_filename, gen_image[0])
imsave(folder_path + "/" + tar_filename, tar_image[0])

# Calculate PSNR
psnr = peak_signal_noise_ratio(tar_image, gen_image)
print("PSNR:", psnr)