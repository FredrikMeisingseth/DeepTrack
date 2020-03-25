def get_image_parameters(
        particle_center_x_list=lambda: [0, ],
        particle_center_y_list=lambda: [0, ],
        particle_radius_list=lambda: [3, ],
        particle_bessel_orders_list=lambda: [[1, ], ],
        particle_intensities_list=lambda: [[.5, ], ],
        image_size=lambda: 128,
        image_background_level=lambda: .5,
        signal_to_noise_ratio=lambda: 30,
        gradient_intensity=lambda: .2,
        gradient_direction=lambda: 0,
        ellipsoidal_orientation=lambda: [0, ],
        ellipticity=lambda: 1):
    """Get image parameters.
    
    Inputs:
    particle_center_x_list: x-centers of the particles [px, list of real numbers]
    particle_center_y_list: y-centers of the particles [px, list of real numbers]
    particle_radius_list: radii of the particles [px, list of real numbers]
    particle_bessel_orders_list: Bessel orders of the particles
    particle_intensities_list: intensities of the particles [list of real numbers, normalized to 1]
    image_size: size of the image in pixels [px, positive integer]
    image_background_level: background level [real number normalized to 1]
    signal_to_noise_ratio: signal to noise ratio [positive real number]
    gradient_intensity: gradient intensity [real number normalized to 1]
    gradient_direction: gradient angle [rad, real number]
    ellipsoidal_orientation: Orientation of elliptical particles [rad, real number] 
    ellipticity: shape of the particles, from spherical to elliptical [real number]
    
    Note: particle_center_x, particle_center_x, particle_radius, 
    particle_bessel_order, particle_intensity, ellipsoidal_orientation must have the same length.
    
    Output:
    image_parameters: list with the values of the image parameters in a dictionary:
        image_parameters['Particle Center X List']
        image_parameters['Particle Center Y List']
        image_parameters['Particle Radius List']
        image_parameters['Particle Bessel Orders List']
        image_parameters['Particle Intensities List']
        image_parameters['Image Size']
        image_parameters['Image Background Level']
        image_parameters['Signal to Noise Ratio']
        image_parameters['Gradient Intensity']
        image_parameters['Gradient Direction']
        image_parameters['Ellipsoid Orientation']
        image_parameters['Ellipticity']
    """

    image_parameters = {}
    image_parameters['Particle Center X List'] = particle_center_x_list()
    image_parameters['Particle Center Y List'] = particle_center_y_list()
    image_parameters['Particle Radius List'] = particle_radius_list()
    image_parameters['Particle Bessel Orders List'] = particle_bessel_orders_list()
    image_parameters['Particle Intensities List'] = particle_intensities_list()
    image_parameters['Image Size'] = image_size()
    image_parameters['Image Background Level'] = image_background_level()
    image_parameters['Signal to Noise Ratio'] = signal_to_noise_ratio()
    image_parameters['Gradient Intensity'] = gradient_intensity()
    image_parameters['Gradient Direction'] = gradient_direction()
    image_parameters['Ellipsoid Orientation'] = ellipsoidal_orientation()
    image_parameters['Ellipticity'] = ellipticity()

    return image_parameters


def get_image_parameters_preconfig(image_size=256):
    from numpy.random import uniform, randint
    from numpy import ones
    from math import pi

    particle_number = randint(10, 30)
    particle_radius_list = uniform(0.5, 2, particle_number)
    (particle_center_x_list, particle_center_y_list) = get_particle_positions(particle_radius_list, image_size)

    particle_bessel_orders_list = []
    particle_intensities_list = []

    for i in range(particle_number):
        particle_bessel_orders_list.append([1, ])
        particle_intensities_list.append([uniform(0.05, 0.2, 1), ])

    image_parameters = get_image_parameters(
        particle_center_x_list=lambda: particle_center_x_list,
        particle_center_y_list=lambda: particle_center_y_list,
        particle_radius_list=lambda: particle_radius_list,
        particle_bessel_orders_list=lambda: particle_bessel_orders_list,
        particle_intensities_list=lambda: particle_intensities_list,
        image_size=lambda: image_size,
        image_background_level=lambda: uniform(.3, .5),
        signal_to_noise_ratio=lambda: 50,
        gradient_intensity=lambda: uniform(0, 0.1),
        gradient_direction=lambda: uniform(-pi, pi),
        ellipsoidal_orientation=lambda: uniform(-pi, pi, particle_number),
        ellipticity=lambda: 1)

    return image_parameters


def get_aug_parameters():
    return dict(rotation_range=0.2,
                width_shift_range=0.05,
                height_shift_range=0.05,
                shear_range=0.05,
                zoom_range=0.05,
                horizontal_flip=True,
                fill_mode='nearest')


def get_particle_positions(particle_radius_list=[], image_size=128):
    """Generates multiple particle x- and y-coordinates with respect to each other.

    Inputs:  
    particle_number: number of particles to generate coordinates for
    first_particle_range: allowed x- and y-range of the centermost particle
    other_particle_range: allowed x- and y-range for all other particles
    particle_distance: particle interdistance
    
    Output:
    particles_center_x: list of x-coordinates for the particles
    particles_center_y: list of y-coordinates for the particles
    """

    from numpy.random import uniform
    from numpy import reshape

    particle_centers = []
    for radius in particle_radius_list:
        # print('X is: ' + str(x) + ". Y is: " + str(y) + ". Radius is: " + str(radius) + ". Image size is: " + str(image_size) + '.')
        for i in range(100):
            (x, y) = (uniform(radius, image_size - radius), uniform(radius, image_size - radius))
            if all(((x - coord[0]) ** 2 + (y - coord[1]) ** 2) ** 0.5 > radius for coord in particle_centers):
                particle_centers.append([x, y])
                break
            elif i == 99:
                raise Exception("Couldn't place out another particle after 100 tries")
    particle_centers_x = []
    particle_centers_y = []
    for coordinates in particle_centers:
        particle_centers_x.append(coordinates[0])
        particle_centers_y.append(coordinates[1])

    return (particle_centers_x, particle_centers_y)


def get_image(image_parameters):
    """Generate image with particles.
    Input:
    image_parameters: list with the values of the image parameters in a dictionary:
        image_parameters['Particle Center X List']
        image_parameters['Particle Center Y List']
        image_parameters['Particle Radius List']
        image_parameters['Particle Bessel Orders List']
        image_parameters['Particle Intensities List']
        image_parameters['Image Size']
        image_parameters['Image Background Level']
        image_parameters['Signal to Noise Ratio']
        image_parameters['Gradient Intensity']
        image_parameters['Gradient Direction']
        image_parameters['Ellipsoid Orientation']
        image_parameters['Ellipticity']
        
    Note: image_parameters is typically obained from the function get_image_parameters()
        
    Output:
    image: image of the particle [2D numpy array of real numbers betwen 0 and 1]
    """

    from numpy import meshgrid, arange, ones, zeros, sin, cos, sqrt, clip, array, ceil, mean, amax, asarray, amin
    from numpy.random import normal, poisson
    from math import e
    from scipy.special import jv as bessel
    import warnings

    particle_center_x_list = image_parameters['Particle Center X List']
    particle_center_y_list = image_parameters['Particle Center Y List']
    particle_radius_list = image_parameters['Particle Radius List']
    particle_bessel_orders_list = image_parameters['Particle Bessel Orders List']
    particle_intensities_list = image_parameters['Particle Intensities List']
    image_size = image_parameters['Image Size']
    image_background_level = image_parameters['Image Background Level']
    signal_to_noise_ratio = image_parameters['Signal to Noise Ratio']
    gradient_intensity = image_parameters['Gradient Intensity']
    gradient_direction = image_parameters['Gradient Direction']
    ellipsoidal_orientation_list = image_parameters['Ellipsoid Orientation']
    ellipticity = image_parameters['Ellipticity']

    ### CALCULATE BACKGROUND
    # initialize the image at the background level
    image_background = ones((image_size, image_size)) * image_background_level

    # calculate matrix coordinates from the center of the image
    image_coordinate_x, image_coordinate_y = meshgrid(arange(0, image_size),
                                                      arange(0, image_size),
                                                      sparse=False,
                                                      indexing='ij')

    # add gradient to image background
    image_background = image_background + gradient_intensity * (image_coordinate_x * sin(gradient_direction) +
                                                                image_coordinate_y * cos(gradient_direction)) / (
                               sqrt(2) * image_size)

    ### CALCULATE IMAGE PARTICLES
    image_particles = zeros((image_size, image_size))
    particle_intensities_for_SNR = []

    # calculate the particle profiles of all particles and add them to image_particles

    for particle_center_x, particle_center_y, particle_radius, particle_bessel_orders, particle_intensities, ellipsoidal_orientation in zip(
            particle_center_x_list, particle_center_y_list, particle_radius_list, particle_bessel_orders_list,
            particle_intensities_list, ellipsoidal_orientation_list):
        # calculate coordinates of cutoff window
        start_x = int(max(ceil(particle_center_x - particle_radius * 3), 0))
        stop_x = int(min(ceil(particle_center_x + particle_radius * 3), image_size))
        start_y = int(max(ceil(particle_center_y - particle_radius * 3), 0))
        stop_y = int(min(ceil(particle_center_y + particle_radius * 3), image_size))

        # calculate matrix coordinates from the center of the image
        image_coordinate_x, image_coordinate_y = meshgrid(arange(start_x, stop_x),
                                                          arange(start_y, stop_y),
                                                          sparse=False,
                                                          indexing='ij')

        # calculate the elliptical distance from the center of the particle normalized by the particle radius
        rotated_distance_x = (image_coordinate_x - particle_center_x) * cos(ellipsoidal_orientation) + (
                image_coordinate_y - particle_center_y) * sin(ellipsoidal_orientation)
        rotated_distance_y = -(image_coordinate_x - particle_center_x) * sin(ellipsoidal_orientation) + (
                image_coordinate_y - particle_center_y) * cos(ellipsoidal_orientation)

        # The factor 2 is because the particle radius is defined as the point where the intensity reaches 1/3 of
        # the intensity in the middle of the particle when Bessel order = 0. When Bessel order = 1, the middle of
        # the particle is black, and at the radius the intensity is approximately at its maximum. For higher
        # Bessel orders, there is no clear definition of the radius.
        elliptical_distance_from_particle = 2 * sqrt((rotated_distance_x) ** 2
                                                     + (rotated_distance_y / ellipticity) ** 2
                                                     + .001 ** 2) / particle_radius

        # calculate particle profile.
        for particle_bessel_order, particle_intensity in zip(particle_bessel_orders, particle_intensities):
            image_particle = 4 * particle_bessel_order ** 2.5 * (bessel(particle_bessel_order,
                                                                        elliptical_distance_from_particle) / elliptical_distance_from_particle) ** 2
            image_particles[start_x:stop_x, start_y:stop_y] = image_particles[start_x:stop_x,
                                                              start_y:stop_y] + particle_intensity * image_particle

    # calculate image without noise as background image plus particle image
    image_particles_without_noise = clip(image_background + image_particles, 0, 1)

    ### ADD NOISE
    image_particles_with_noise = poisson(
        image_particles_without_noise * signal_to_noise_ratio ** 2) / signal_to_noise_ratio ** 2

    cut_off_pixels = tuple([image_particles_with_noise > 1])

    percentage_of_pixels_that_were_cut_off = image_particles_with_noise[cut_off_pixels].size / (image_size ** 2) * 100

    # warn if there is a pixel brighter than 1
    def custom_formatwarning(msg, *args, **kwargs):
        # ignore everything except the message
        return str(msg) + '\n'

    if percentage_of_pixels_that_were_cut_off > 0:
        warnings.formatwarning = custom_formatwarning
        warn_message = ("Warning: %.5f%% of the pixels in the generated image are brighter than the 1 (%d pixels)! "
                        "These were cut-off to the max value 1. Consider adjusting your gradient intensity, particle "
                        "intensity, background level, or signal to noise ratio." % (
                            percentage_of_pixels_that_were_cut_off, image_particles_with_noise[cut_off_pixels].size))
        warnings.warn(warn_message)

    # print("After poisson: Min is %.4f, Max is %.4f" % (amin(image_particles_with_noise),
    #                                                    amax(image_particles_with_noise)))

    return clip(image_particles_with_noise, 0, 1)


def draw_image(img):
    from matplotlib import pyplot as plt
    plt.imshow(img, cmap='gray')
    plt.show()


def get_label(image_parameters=get_image_parameters_preconfig()):
    """Create and return binary target image given image parameters
    Input: Image parameters
    Output: Array of size (image_x, image_y, number_of_features = 5), where the features at index i are:
        i = 0 - binary image (is there a particle here?)
        i = 1 - delta_x (to the particle center)
        i = 2 - delta_y
        i = 3 - radius
        i = 4 - intensity
    """

    import numpy as np

    particle_center_x_list = image_parameters['Particle Center X List']
    particle_center_y_list = image_parameters['Particle Center Y List']
    particle_radius_list = image_parameters['Particle Radius List']
    image_size = image_parameters['Image Size']
    particle_intensities_list = image_parameters['Particle Intensities List']

    target_binary_image = np.zeros((image_size, image_size, 5))

    for particle_index in range(0, len(particle_center_x_list)):
        center_x = particle_center_x_list[particle_index]
        center_y = particle_center_y_list[particle_index]
        radius = particle_radius_list[particle_index]
        intensity = particle_intensities_list[particle_index]

        # loops over all pixels with center in coordinates = [ceil(center - radius): floor(center + radius)]. Adds the ones with
        # center within radius.
        for pixel_x in range(int(np.floor(center_x - radius)), int(np.ceil(center_x + radius))):
            for pixel_y in range(int(np.floor(center_y - radius)), int(np.ceil(center_y + radius))):
                if ((pixel_x - center_x) ** 2 + (pixel_y - center_y) ** 2 <= radius ** 2):
                    # print('Pixel_x is: ' + str(pixel_x) + ". Pixel_y is: " + str(pixel_y) + ".")
                    target_binary_image[pixel_x, pixel_y, 0] = 1
                    target_binary_image[pixel_x, pixel_y, 1] = center_x - pixel_x
                    target_binary_image[pixel_x, pixel_y, 2] = center_y - pixel_y
                    target_binary_image[pixel_x, pixel_y, 3] = radius
                    target_binary_image[pixel_x, pixel_y, 4] = intensity[0]

    return target_binary_image


def get_batch(get_image_parameters=lambda: get_image_parameters_preconfig(),
              batch_size=32):
    from numpy import zeros
    import time

    example_image_parameters = get_image_parameters()
    image_size = example_image_parameters['Image Size']
    image_batch = zeros((batch_size, image_size, image_size,
                         1))  # possibly save in smaller format? + Preallocating assumes equal image-sizes!
    label_batch = zeros((batch_size, image_size, image_size,
                         5))  # possibly save in smaller format? + Preallocating assumes equal image-sizes!

    t = time.time()
    for i in range(batch_size):
        image_parameters = get_image_parameters()
        image_batch[i, :, :, 0] = get_image(image_parameters)
        label_batch[i, :, :, 0:5] = get_label(image_parameters)

    time_taken = time.time() - t

    print("Time taken for batch generation of size " + str(batch_size) + ": " + str(time_taken) + " s.")

    return image_batch, label_batch


def save_batch(batch, image_path='data', label_path='data', image_filename='image', label_filename='label'):
    import cv2
    import numpy as np
    import os

    (image_batch, label_batch) = batch
    (batch_size) = image_batch.shape[0]
    if not os.path.isdir(image_path):
        os.mkdir(image_path)
        print("Created path " + image_path)
    if not os.path.isdir(label_path):
        os.mkdir(label_path)
        print("Created path " + label_path)

    for i in range(batch_size):
        image = (image_batch[i] * 255).astype(np.uint8)
        cv2.imwrite("%s/%s%d.png" % (image_path, image_filename, i), image)
        np.save("%s/%s%d" % (label_path, label_filename, i), label_batch[i])

    return


def load_batch(batch_size, image_path='data', label_path='data', image_filename='image', label_filename='label'):
    from skimage.io import imread
    import numpy as np

    image_shape = imread("%s/%s%d.png" % (image_path, image_filename, 0)).shape
    label_shape = np.load("%s/%s%d.npy" % (label_path, label_filename, 0)).shape
    image_batch = np.zeros((batch_size,) + image_shape + (1,))
    label_batch = np.zeros((batch_size,) + label_shape)

    for j in range(batch_size):
        image_batch[j, :, :, 0] = imread("%s/%s%d.png" % (image_path, image_filename, j))/255
        label_batch[j] = np.load("%s/%s%d.npy" % (label_path, label_filename, j))

    return image_batch, label_batch


def generator_for_training(get_batch=lambda: get_batch(), aug_parameters=get_aug_parameters()):
    from keras.preprocessing.image import ImageDataGenerator
    import numpy as np
    (image_batch, label_batch) = get_batch()
    image_shape = image_batch.shape
    image_batch = np.reshape(image_batch, (image_shape[0], image_shape[1], image_shape[2], 1))
    label_batch = np.reshape(label_batch,
                             (image_shape[0], image_shape[1], image_shape[2], 1))  # Expects a color channel
    print(image_batch.shape)

    data_generator = ImageDataGenerator(**aug_parameters)
    return data_generator.flow(image_batch, label_batch, batch_size=32)
    # Som jag fattar det, batch size här är hur många augmenterade bilder den genererar från grunddatan


def generator_for_training_load(image_path, label_path, aug_parameters=get_aug_parameters()):
    from keras.preprocessing.image import ImageDataGenerator
    image_datagenerator = ImageDataGenerator(**aug_parameters)
    label_datagenerator = ImageDataGenerator(**aug_parameters)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    image_generator = image_datagenerator.flow_from_directory(
        image_path,
        class_mode=None,
        seed=seed)

    label_generator = label_datagenerator.flow_from_directory(
        label_path,
        class_mode=None,
        seed=seed)

    # combine generators into one which yields image and masks
    return zip(image_generator, label_generator)


def get_batch_load(filename):
    from skimage import io
    from numpy import reshape
    image_batch = io.imread(filename)
    image_batch = reshape(image_batch, (image_batch.shape[0], image_batch.shape[1], image_batch.shape[2], 1))
    return image_batch


def get_padding(input_size, n):
    """Adds padding to the input image
    Inputs:
    input: the input image
    input_size: the size of the input image
    n: the input image dimensions are changed to be divisible by 2**n

    Outputs:
    padding: the padding that was used
    """
    C0 = 2 ** (n - 1)
    C1 = 2 ** (n - 1)
    if (input_size[0] % 8 != 0):
        top_pad = (input_size[0] % (2 * n) // 2)
        bottom_pad = (input_size[0] % (2 * n) - top_pad)
    else:
        top_pad = 0
        bottom_pad = 0
        C0 = 0
    if input_size[1] % 8 != 0:
        left_pad = (input_size[1] % (2 * n) // 2)
        right_pad = (input_size[1] % (2 * n) - left_pad)
    else:
        left_pad = 0
        right_pad = 0
        C1 = 0
        padding = ((C0 - top_pad, C0 - bottom_pad), (C1 - left_pad, C1 - right_pad))

    return (padding)


def create_data_generator(get_image_parameters=lambda: get_image_parameters_preconfig(),
                          epoch_batch_size=1000,
                          batch_size=32,
                          len=100):
    from keras.utils import Sequence

    class DataGenerator(Sequence):
        """
        At the beginning of each epoch, generates a batch of size epoch_batch_size using get_image_parameters. Then,
        for each step, outputs a batch of size batch_size. This is done at most len times.
        """

        def __init__(self,
                     get_image_parameters=lambda: get_image_parameters_preconfig(),
                     epoch_batch_size=1000,
                     batch_size=32,
                     len=100):
            'Initialization'
            self.get_image_parameters = get_image_parameters
            self.epoch_batch_size = epoch_batch_size
            self.on_epoch_end()
            self.len = len
            self.batch_size = batch_size

        def on_epoch_end(self):
            self.batch = get_batch(self.get_image_parameters, self.epoch_batch_size)
            image_batch, label_batch = self.batch
            from matplotlib import pyplot as plt
            plt.imshow(image_batch[0, :, :, 0], cmap='gray')
            plt.show()
            plt.imshow(label_batch[0, :, :, 0], cmap='gray')
            plt.show()

        def __len__(self):
            return (self.len)

        def __getitem__(self, index):
            from random import randint
            image_indices = [randint(0, self.epoch_batch_size - 1) for i in range(self.batch_size)]
            image_batch, label_batch = self.batch
            return image_batch[image_indices], label_batch[image_indices]

    return DataGenerator(get_image_parameters, epoch_batch_size, batch_size, len)


def get_particle_centers(label):
    from skimage import measure
    from statistics import mean
    from numpy import argwhere
    (label_id, number_of_particles) = measure.label(label[:, :, 0], return_num=True)
    # Bra namn
    x_mean_list = []
    y_mean_list = []
    r_mean_list = []
    i_mean_list = []
    for particle_id in range(1, number_of_particles + 1):
        x_list = []
        y_list = []
        r_list = []
        i_list = []
        coords = argwhere(label_id == particle_id)
        for coord in coords:
            x_list.append(coord[0] + label[coord[0], coord[1], 1])
            y_list.append(coord[1] + label[coord[0], coord[1], 2])
            r_list.append(label[coord[0], coord[1], 3])
            i_list.append(label[coord[0], coord[1], 4])
        x_mean_list.append(mean(x_list))
        y_mean_list.append(mean(y_list))
        r_mean_list.append(mean(r_list))
        i_mean_list.append(mean(i_list))
    return (x_mean_list, y_mean_list, r_mean_list, i_mean_list)
