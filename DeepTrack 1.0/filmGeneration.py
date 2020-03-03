def get_image_parameters(
    particle_center_x_list=lambda : [0, ], 
    particle_center_y_list=lambda : [0, ], 
    particle_radius_list=lambda : [3, ], 
    particle_bessel_orders_list=lambda : [[1, ], ], 
    particle_intensities_list=lambda : [[.5, ], ],
    image_size=lambda : 128, 
    image_background_level=lambda : .5,
    signal_to_noise_ratio=lambda : 30,
    gradient_intensity=lambda : .2, 
    gradient_direction=lambda : 0,
    ellipsoidal_orientation=lambda : [0, ], 
    ellipticity=lambda : 1):
    """Get image parameters.
    
    Inputs:
    particle_center_x_list: x-centers of the particles [px, list of real numbers]
    particle_center_y_list: y-centers of the particles [px, list of real numbers]
    particle_radius_list: radii of the particles [px, list of real numbers]
    particle_bessel_orders_list: Bessel orders of the particles [list (of lists) of positive integers]
    particle_intensities_list: intensities of the particles [list (of lists) of real numbers, normalized to 1]
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

def get_image_parameters_preconfig(image_size = 256):

    from numpy.random import uniform, randint
    from math import pi

    particle_number= randint(5, 6)
    particle_radius_list = uniform(0.25, 3, particle_number)
    (particle_center_x_list, particle_center_y_list) = get_particle_positions(particle_radius_list,image_size)

    particle_bessel_orders_list= []
    particle_intensities_list= []
    for i in range(particle_number):
        particle_bessel_orders_list.append([1,])
        particle_intensities_list.append([uniform(0.1,0.5,1),])

    image_parameters = get_image_parameters(
        particle_center_x_list= lambda : particle_center_x_list, 
        particle_center_y_list= lambda : particle_center_y_list,
        particle_radius_list=lambda : particle_radius_list,
        particle_bessel_orders_list= lambda:  particle_bessel_orders_list,
        particle_intensities_list= lambda : particle_intensities_list,
        image_size=lambda : image_size,
        image_background_level=lambda : uniform(.2, .5),
        signal_to_noise_ratio=lambda : 100,
        gradient_intensity=lambda : uniform(0.25, 0.75), 
        gradient_direction=lambda : uniform(-pi, pi),
        ellipsoidal_orientation=lambda : uniform(-pi, pi, particle_number), 
        ellipticity= lambda: 1)

    return image_parameters


def get_image_parameters_film(image_parameters_prev, image_size = 256):
    from numpy import reshape, delete
    from numpy.random import uniform, randint
    from math import pi, cos, sin

    max_move = 10
    particle_centers = []
    for (radius, x, y, index) in zip(image_parameters_prev['Particle Radius List'],
                             image_parameters_prev['Particle Center X List'],
                             image_parameters_prev['Particle Center Y List'],
                             range(len(image_parameters_prev['Particle Radius List']))):

        for i in range(100):
            r = uniform(0, max_move)
            theta = uniform(0, 2*pi)
            (x_new, y_new) = (x+r*cos(theta), y+r*sin(theta))
            if all(((x_new - coord[0]) ** 2 + (y_new - coord[1]) ** 2) ** 0.5 > radius for coord in particle_centers):
                if (not (radius < x_new < image_size-radius)) or (not (radius < y_new < image_size-radius)):
                    delete(image_parameters_prev['Particle Radius List'], index)
                    delete(image_parameters_prev['Particle Bessel Orders List'], index)
                    delete(image_parameters_prev['Particle Intensities List'], index)          
                else:
                    particle_centers.append([x_new, y_new])
                break
            elif i==99:
                raise Exception("Couldn't place out another particle after 100 tries")
    particle_centers_x=[]
    particle_centers_y=[]
    for coordinates in particle_centers:
        particle_centers_x.append(coordinates[0])
        particle_centers_y.append(coordinates[1])

    image_parameters_prev['Particle Center X List'] = particle_centers_x
    image_parameters_prev['Particle Center Y List'] = particle_centers_y

    return image_parameters_prev

def get_aug_parameters():
    return dict(rotation_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest')

def get_particle_positions(particle_radius_list = [], image_size = 128):
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

    
    particle_centers=[]
    for radius in particle_radius_list:
        #print('X is: ' + str(x) + ". Y is: " + str(y) + ". Radius is: " + str(radius) + ". Image size is: " + str(image_size) + '.')
        for i in range(100):
            (x, y) = (uniform(radius, image_size - radius), uniform(radius, image_size - radius))
            if all(((x - coord[0]) ** 2 + (y - coord[1]) ** 2) ** 0.5 > radius for coord in particle_centers):
                particle_centers.append([x, y])
                break
            elif i==99:
                raise Exception("Couldn't place out another particle after 100 tries")
    particle_centers_x=[]
    particle_centers_y=[]
    for coordinates in particle_centers:
        particle_centers_x.append(coordinates[0])
        particle_centers_y.append(coordinates[1])
    
    return (particle_centers_x,particle_centers_y)

def get_image(image_parameters, use_gpu=False):
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


    from numpy import meshgrid, arange, ones, zeros, sin, cos, sqrt, clip, array
    from numpy.random import poisson as poisson
    
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
    if gradient_intensity!=0:
        image_background = image_background + gradient_intensity * (image_coordinate_x * sin(gradient_direction) + 
                                                                    image_coordinate_y * cos(gradient_direction) ) / (sqrt(2) * image_size)

    ### CALCULATE IMAGE PARTICLES
    image_particles = zeros((image_size, image_size))

    # calculate the particle profiles of all particles and add them to image_particles
    if(use_gpu):
        print('No GPU!')
        #calc_particle_profile_gpu(particle_center_x_list, particle_center_y_list,particle_radius_list, image_particles,particle_intensities_list)
    else:
        from scipy.special import jv as bessel
        
        for particle_center_x, particle_center_y, particle_radius, particle_bessel_orders, particle_intensities, ellipsoidal_orientation in zip(particle_center_x_list, particle_center_y_list, particle_radius_list, particle_bessel_orders_list, particle_intensities_list, ellipsoidal_orientation_list):
        
        
            # calculate the radial distance from the center of the particle 
            # normalized by the particle radius
            radial_distance_from_particle = sqrt((image_coordinate_x - particle_center_x)**2 
                                            + (image_coordinate_y - particle_center_y)**2 
                                            + .001**2) / particle_radius
            

            # for elliptical particles
            rotated_distance_x = (image_coordinate_x - particle_center_x)*cos(ellipsoidal_orientation) + (image_coordinate_y - particle_center_y)*sin(ellipsoidal_orientation)
            rotated_distance_y = -(image_coordinate_x - particle_center_x)*sin(ellipsoidal_orientation) + (image_coordinate_y - particle_center_y)*cos(ellipsoidal_orientation)
            
            
            elliptical_distance_from_particle = sqrt((rotated_distance_x)**2 
                                            + (rotated_distance_y / ellipticity)**2 
                                            + .001**2) / particle_radius


            # calculate particle profile
            for particle_bessel_order, particle_intensity in zip(particle_bessel_orders, particle_intensities):
                image_particle = 4 * particle_bessel_order**2.5 * (bessel(particle_bessel_order, elliptical_distance_from_particle) / elliptical_distance_from_particle)**2
                image_particles = image_particles + particle_intensity * image_particle

    # calculate image without noise as background image plus particle image
    image_particles_without_noise = clip(image_background + image_particles, 0, 1)

    ### ADD NOISE
    image_particles_with_noise = poisson(image_particles_without_noise * signal_to_noise_ratio**2) / signal_to_noise_ratio**2

    return image_particles_with_noise

# def calc_particle_profile_gpu(particle_center_x_list, particle_center_y_list,particle_radius_list, image_particles,particle_intensities_list):
#     from numba import cuda
#     from math import ceil,exp


#     # the cuda kernel calculating the value of the Gauss function for each pixel in out image
#     @cuda.jit
#     def part_prof(d_dist_x,d_dist_y,d_radiuses,d_img_part,d_particle_intensities):

#         x, y = cuda.grid(2)

#         if x >= d_img_part.shape[0] and y >= d_img_part.shape[1]:
#             # Quit if (x, y) is outside of valid C boundary
#             return

#         for i in range(d_dist_x.shape[0]):

#             tmp = d_particle_intensities[i][0][0]*exp(-((x-d_dist_x[i])**2/(2*d_radiuses[i]**2) + (y-d_dist_y[i])**2/(2*d_radiuses[i]**2)))

#             d_img_part[x, y] = d_img_part[x,y] + tmp

#     # define threads per block and blocks per grid. This dictates how our cuda kernel devides tasks.
#     TPB = 32
#     threadsperblock = (TPB, TPB)
#     blockspergrid_x = int(ceil(image_particles.shape[0] / threadsperblock[0]))
#     blockspergrid_y = int(ceil(image_particles.shape[1] / threadsperblock[1]))
#     blockspergrid = (blockspergrid_x, blockspergrid_y)
    
#     # introduce stream dictating the order of data being sent to GPU
#     # create an cuda object of each object we wish to have handled by the GPU. This is because data transfer to and from GPU is costly.
#     stream = cuda.stream()
#     d_pos_x = cuda.to_device(particle_center_x_list,stream = stream)
#     d_pos_y = cuda.to_device(particle_center_y_list,stream = stream)
#     d_radiuses = cuda.to_device(particle_radius_list,stream = stream)
#     d_img_part = cuda.to_device(image_particles,stream = stream)
#     d_particle_intensities = cuda.to_device(particle_intensities_list,stream = stream)

#     # call the cuda kernel
#     part_prof[blockspergrid, threadsperblock](d_pos_x,d_pos_y,d_radiuses, d_img_part,d_particle_intensities)

#     # retrieve our image particle matrix from GPU
#     d_img_part.copy_to_host(image_particles, stream = stream)

def draw_image(img):
    from matplotlib import pyplot as plt
    plt.imshow(img, cmap='gray')
    plt.show()

def plot_sample_image(image, image_parameters, figsize=(15, 5)):
    """
    Plot a sample image.

    Inputs:
    image: image of the particles
    image_parameters: list with the values of the image parameters
    figsize: figure size [list of two positive numbers]
    """
    

    ### CALCULATE BACKGROUND
    # initialize the image at the background level
    image_background = ones((image_size, image_size)) * image_background_level
    
    # add gradient to image background
    if gradient_intensity!=0:
        image_background = image_background + gradient_intensity * (image_coordinate_x * sin(gradient_direction) + 
                                                                    image_coordinate_y * cos(gradient_direction) ) / (sqrt(2) * image_size)

        

    ### CALCULATE IMAGE PARTICLES
    image_particles = zeros((image_size, image_size))
    for particle_center_x, particle_center_y, particle_radius, particle_bessel_orders, particle_intensities, ellipsoidal_orientation in zip(particle_center_x_list, particle_center_y_list, particle_radius_list, particle_bessel_orders_list, particle_intensities_list, ellipsoidal_orientation_list):
    
    
        # calculate the radial distance from the center of the particle 
        # normalized by the particle radius
        radial_distance_from_particle = sqrt((image_coordinate_x - particle_center_x)**2 
                                        + (image_coordinate_y - particle_center_y)**2 
                                        + .001**2) / particle_radius
        

        # for elliptical particles
        rotated_distance_x = (image_coordinate_x - particle_center_x)*cos(ellipsoidal_orientation) + (image_coordinate_y - particle_center_y)*sin(ellipsoidal_orientation)
        rotated_distance_y = -(image_coordinate_x - particle_center_x)*sin(ellipsoidal_orientation) + (image_coordinate_y - particle_center_y)*cos(ellipsoidal_orientation)
        
        
        elliptical_distance_from_particle = sqrt((rotated_distance_x)**2 
                                        + (rotated_distance_y / ellipticity)**2 
                                        + .001**2) / particle_radius


        # calculate particle profile
        for particle_bessel_order, particle_intensity in zip(particle_bessel_orders, particle_intensities):
            image_particle = 4 * particle_bessel_order**2.5 * (bessel(particle_bessel_order, elliptical_distance_from_particle) / elliptical_distance_from_particle)**2
            image_particles = image_particles + particle_intensity * image_particle

        

    # calculate image without noise as background image plus particle image
    image_particles_without_noise = clip(image_background + image_particles, 0, 1)

    ### ADD NOISE
    image_particles_with_noise = poisson(image_particles_without_noise * signal_to_noise_ratio**2) / signal_to_noise_ratio**2
    

    return image_particles_with_noise

def get_label(image_parameters=get_image_parameters_preconfig(), use_gpu=False):
    """Create and return binary target image given image parameters
    Input: Image parameters
    Output: Array of size (image_x, image_y, number_of_features = 5), where the features at index i are:
        i = 0 - binary image (is there a particle here?)
        i = 1 - delta_x (to the particle center)
        i = 2 - delta_y
        i = 3 - radius
        i = 4 - intensity
    """
    if (use_gpu):
        pass
        # print('GPU not yet implemented')
        # return targetBinaryImage

    import numpy as np

    particle_center_x_list = image_parameters['Particle Center X List']
    particle_center_y_list = image_parameters['Particle Center Y List']
    particle_radius_list = image_parameters['Particle Radius List']
    image_size = image_parameters['Image Size']
    particle_intensities_list = image_parameters['Particle Intensities List']

    targetBinaryImage = np.zeros((image_size, image_size, 5))

    for particle_index in range(0, len(particle_center_x_list)):
        center_x = particle_center_x_list[particle_index]
        center_y = particle_center_y_list[particle_index]
        radius = particle_radius_list[particle_index]
        intensity = particle_intensities_list[particle_index]
        #print('Center_x is: ' + str(center_x) + ". Center_y is: " + str(center_y) + ". Radius is: " + str(radius) + ".")

        """Loops over all pixels with center in coordinates = [ceil(center - radius): floor(center + radius)]. Adds the ones with
        center within radius.
        """
        for pixel_x in range(int(np.floor(center_x - radius)), int(np.ceil(center_x + radius))):
            for pixel_y in range(int(np.floor(center_y - radius)), int(np.ceil(center_y + radius))):
                if ((pixel_x - center_x) ** 2 + (pixel_y - center_y) ** 2 <= radius ** 2):
                    # print('Pixel_x is: ' + str(pixel_x) + ". Pixel_y is: " + str(pixel_y) + ".")
                    targetBinaryImage[pixel_x, pixel_y, 0] = 1
                    targetBinaryImage[pixel_x, pixel_y, 1] = center_x - pixel_x
                    targetBinaryImage[pixel_x, pixel_y, 2] = center_y - pixel_y
                    # print('X vector is: ' + str(targetBinaryImage[pixel_x, pixel_y, 1]) + '. Y vector is: ' + str(targetBinaryImage[pixel_x, pixel_y, 2] ))
                    targetBinaryImage[pixel_x, pixel_y, 3] = radius
                    targetBinaryImage[pixel_x, pixel_y, 4] = intensity[0]

    return targetBinaryImage

def get_label_old(image_parameters=get_image_parameters_preconfig(), use_gpu=False):
    """Create and return binary target image given image parameters
    Input: Image parameters
    Output: Binary image of the input image size where pixels containing particles are marked as ones, while rest are zeros
    """
    if (use_gpu):
        print('GPU not yet implemented')
    else:
        import numpy as np


        particle_center_x_list = image_parameters['Particle Center X List']
        particle_center_y_list = image_parameters['Particle Center Y List']
        particle_radius_list = image_parameters['Particle Radius List']
        image_size = image_parameters['Image Size']

        targetBinaryImage = np.zeros((image_size, image_size))

        for particle_index in range(0, len(particle_center_x_list)):
            center_x = particle_center_x_list[particle_index]
            center_y = particle_center_y_list[particle_index]
            radius = particle_radius_list[particle_index]

            """Loops over all pixels with center in coordinates = [ceil(center - radius): floor(center + radius)]. Adds the ones with
            center within radius.
            """
            for pixel_x in range(int(np.floor(center_x - radius)), int(np.ceil(center_x + radius) + 1)):
                for pixel_y in range(int(np.floor(center_y - radius)), int(np.ceil(center_y + radius) + 1)):
                    if ((pixel_x - center_x) ** 2 + (pixel_y - center_y) ** 2 <= radius ** 2):
                        targetBinaryImage[pixel_x, pixel_y] = 1

        return targetBinaryImage

def get_batch(get_image_parameters = lambda prev: get_image_parameters_film(prev),
              batch_size=32, 
              use_gpu=False, 
              return_image_parameters=False):
    
    from numpy import zeros
    import time

    image_parameters_prev = get_image_parameters_preconfig()
    image_size = image_parameters_prev['Image Size']
    image_batch = zeros((batch_size, image_size, image_size,1)) #possibly save in smaller format? + Preallocating assumes equal image-sizes!
    label_batch = zeros((batch_size, image_size, image_size,5)) #possibly save in smaller format? + Preallocating assumes equal image-sizes!
    image_parameters_list = []

    t = time.time()
    for i in range(batch_size):
        image_parameters = get_image_parameters(image_parameters_prev)
        image_batch[i,:,:,0] = get_image(image_parameters, use_gpu)
        label_batch[i,:,:,:] = get_label(image_parameters, use_gpu)
        if return_image_parameters:
            image_parameters_list.append(image_parameters)
        image_parameters_prev = image_parameters

    time_taken=time.time()-t

    print("Time taken for batch generation of size " + str(batch_size) + ": " + str(time_taken) + " s.")
    
    if return_image_parameters:
        return (image_batch, label_batch, image_parameters_list)
    else:
        return (image_batch, label_batch)

def save_batch(batch, label_path='data/', image_path='data/', image_filename='image', label_filename='label'):
    
    from skimage.io import imsave
    from os.path import join
    
    (image_batch, label_batch) = batch
    (number_of_images, image_size_x, image_size_y, channels) = image_batch.shape

    for i in range(number_of_images):
        imsave(join(image_path, image_filename +"%d.png"%i),image_batch[i,:,:])
        imsave(join(label_path, label_filename +"%d.png"%i),label_batch[i,:,:,0])

    return

def generator_for_training(get_batch = lambda: get_batch(), aug_parameters = get_aug_parameters()):

    from keras.preprocessing.image import ImageDataGenerator
    import numpy as np
    (image_batch, label_batch) = get_batch()
    image_shape = image_batch.shape
    image_batch = np.reshape(image_batch, (image_shape[0], image_shape[1], image_shape[2], 1))
    label_batch = np.reshape(label_batch, (image_shape[0], image_shape[1], image_shape[2], 1)) #Expects a color channel
    print(image_batch.shape)

    data_generator = ImageDataGenerator(**aug_parameters)
    return data_generator.flow(image_batch, label_batch, batch_size=32)
    #Som jag fattar det, batch size här är hur många augmenterade bilder den genererar från grunddatan
     
def generator_for_training_load(image_path, label_path, aug_parameters = get_aug_parameters()):

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
    image_batch = reshape(image_batch, (image_batch.shape[0],image_batch.shape[1],image_batch.shape[2],1))
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

def get_particle_centers(label):
    from skimage import measure
    from statistics import mean
    from numpy import argwhere

    (label_id, number_of_particles) = measure.label(label[:,:,0], return_num=True)
    #Bra namn

    x_mean_list=[]
    y_mean_list=[]

    for particle_id in range(1,number_of_particles+1):
        x_list=[]
        y_list=[]
        coords = argwhere(label_id==particle_id)
        for coord in coords:
            x_list.append(coord[0]+label[coord[0],coord[1],1])
            y_list.append(coord[1]+label[coord[0],coord[1],2])

        x_mean_list.append(mean(x_list))
        y_mean_list.append(mean(y_list))

    return (x_mean_list, y_mean_list)



def get_particle_centers(label):
    from skimage import measure
    from statistics import mean
    from numpy import argwhere
    (label_id, number_of_particles) = measure.label(label[:,:,0], return_num=True)
    #Bra namn
    x_mean_list=[]
    y_mean_list=[]
    for particle_id in range(1,number_of_particles+1):
        x_list=[]
        y_list=[]
        coords = argwhere(label_id==particle_id)
        for coord in coords:
            x_list.append(coord[0]+label[coord[0],coord[1],1])
            y_list.append(coord[1]+label[coord[0],coord[1],2])
        x_mean_list.append(mean(x_list))
        y_mean_list.append(mean(y_list))
    return (x_mean_list, y_mean_list)


def get_particle_centers_pairs(label): #Returns on form [[x1,y1],[x2,y2]...]
    from skimage import measure
    from statistics import mean
    from numpy import argwhere
    (label_id, number_of_particles) = measure.label(label[:,:,0], return_num=True)
    #Bra namn
    particle_centers =[]
    for particle_id in range(1,number_of_particles+1):
        x_list=[]
        y_list=[]
        coords = argwhere(label_id==particle_id)
        for coord in coords:
            x_list.append(coord[0]+label[coord[0],coord[1],1])
            y_list.append(coord[1]+label[coord[0],coord[1],2])

        particle_centers.append([mean(x_list),mean(y_list)])

    return particle_centers
