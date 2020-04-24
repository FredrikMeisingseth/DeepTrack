def get_operating_characteristics(labels, predictions):
    """
    Method that returns the operating characteristics of a prediction.
    Input:
        labels: the batch_labels
        predictions: the batch_predictions. Sigmoid and cutoff should be applied BEFORE applying this method.
    Outputs:
        P:  condition positive - number of real positives in the label (pixels that are 1 in label)
        N:  condition negative - number of real negatives in the label (pixels that are 0 in label)
        TP: true positive - number of correct positive predictions (pixels that are 1 in both label and prediction)
        TN: true negative - number of correct negative predictions (pixels that are 0 in both label and prediction)
        FP: false positive - number of incorrect positive predictions (pixels that are 0 in label, but 1 in prediction)
        FN: false negative - number of incorrect negative predictions (pixels that are 1 in label, but 0 in prediction)
    """
    import numpy as np

    label_first_feature = np.ndarray.flatten(labels[:, :, :, 0])
    prediction_first_feature = np.ndarray.flatten(predictions[:, :, :, 0])

    P = sum(label_first_feature)
    N = sum(1 - label_first_feature)

    TP = sum(label_first_feature * prediction_first_feature)
    TN = sum((1 - label_first_feature) * (1 - prediction_first_feature))

    FP = sum((1 - label_first_feature) * prediction_first_feature)
    FN = sum(label_first_feature * (1 - prediction_first_feature))

    return P, N, TP, FP, TN, FN


def get_operating_characteristics_scanning_box(predicted_particle_positions_x, predicted_particle_positions_y,
                                               particle_positions_x, particle_positions_y,
                                               image_size_x, image_size_y,
                                               scanning_box_size_x, scanning_box_size_y,
                                               scanning_box_step_x, scanning_box_step_y):
    """ Method that returns the operating characteristics using a scanning box approach.
    Inputs:
    predicted_particle_positions_x: list of predicted x-coordinates of the particles
    predicted_particle_positions_y: list of predicted y-coordinates of the particles
    particle_positions_x: list of x-coordinates of the particles
    particle_positions_y: list of y-coordinates of the particles
    image_size_x: x-size of the image. We represent images as matrices, so this is usually the vertical side-length.
    image_size_y: y-size of the image. This is usually the horizontal side length.
    scanning_box_size_x: x-size of the scanning box.
    scanning_box_size_y: y-size of the scanning box.
    scanning_box_step_x: the x-coordinate of the start of the next scanning box relative to the current one.
    scanning_box_step_y: the y-coordinate of the start of the next scanning box relative to the current one.

    Notes:
        1) If a particle is exactly on the far edge of the image, it will not be counted

    Outputs:
        operating_characteristics
    """
    from numpy import meshgrid, arange, ndarray
    scanning_box_stops_x, scanning_box_stops_y = meshgrid(
        arange(scanning_box_size_x, image_size_x + scanning_box_step_x, scanning_box_step_x),
        arange(scanning_box_size_y, image_size_y + scanning_box_step_y, scanning_box_step_y),
        sparse=False,
        indexing='ij')

    operating_characteristics = [0, 0, 0, 0, 0, 0]
    for scanning_box_stop_x, scanning_box_stop_y in zip(ndarray.flatten(scanning_box_stops_x),
                                                        ndarray.flatten(scanning_box_stops_y)):
        scanning_box_start_x = scanning_box_stop_x - scanning_box_size_x
        scanning_box_start_y = scanning_box_stop_y - scanning_box_size_y
        particles_in_box = 0
        predicted_particles_in_box = 0

        for particle_position_x, particle_position_y in zip(particle_positions_x, particle_positions_y):
            if scanning_box_start_x <= particle_position_x < scanning_box_stop_x:
                if scanning_box_start_y <= particle_position_y < scanning_box_stop_y:
                    particles_in_box += 1

        for predicted_particle_position_x, predicted_particle_position_y in zip(predicted_particle_positions_x,
                                                                                predicted_particle_positions_y):
            if scanning_box_start_x <= predicted_particle_position_x < scanning_box_stop_x:
                if scanning_box_start_y <= predicted_particle_position_y < scanning_box_stop_y:
                    predicted_particles_in_box += 1

        # if particles_in_box != predicted_particles_in_box:
        #     print("Particles in box " + str(particles_in_box))
        #     print("Predicted particles in box " + str(predicted_particles_in_box))
        #     print(scanning_box_start_x, scanning_box_start_y)

        P_box = particles_in_box
        if particles_in_box == 0:
            N_box = 1
            TP_box = 0
            FN_box = 0
            if predicted_particles_in_box == 0:
                FP_box = 0
                TN_box = 1
            else:
                FP_box = predicted_particles_in_box
                TN_box = 0
        else:
            N_box = 0
            TN_box = 0
            if predicted_particles_in_box >= particles_in_box:
                TP_box = particles_in_box
                FP_box = predicted_particles_in_box - particles_in_box
                FN_box = 0
            else:
                TP_box = predicted_particles_in_box
                FP_box = 0
                FN_box = particles_in_box - predicted_particles_in_box

        operating_characteristics_box = [P_box, N_box, TP_box, FP_box, TN_box, FN_box]
        operating_characteristics = [sum(x) for x in zip(operating_characteristics, operating_characteristics_box)]
    return operating_characteristics


def get_operating_characteristics_scanning_box_optimized(predicted_particle_positions_x, predicted_particle_positions_y,
                                                         particle_positions_x, particle_positions_y,
                                                         image_size_x, image_size_y,
                                                         scanning_box_size_x, scanning_box_size_y,
                                                         scanning_box_step_x, scanning_box_step_y):
    """ Method that returns the operating characteristics using a scanning box approach.
    Inputs:
    predicted_particle_positions_x: list of predicted x-coordinates of the particles
    predicted_particle_positions_y: list of predicted y-coordinates of the particles
    particle_positions_x: list of x-coordinates of the particles
    particle_positions_y: list of y-coordinates of the particles
    image_size_x: x-size of the image. We represent images as matrices, so this is usually the vertical side-length.
    image_size_y: y-size of the image. This is usually the horizontal side length.
    scanning_box_size_x: x-size of the scanning box.
    scanning_box_size_y: y-size of the scanning box.
    scanning_box_step_x: the x-coordinate of the start of the next scanning box relative to the current one.
    scanning_box_step_y: the y-coordinate of the start of the next scanning box relative to the current one.

    Notes:
        1) If a particle is exactly on the far edge of the image, it will not be counted

    Outputs:
        operating_characteristics
    """
    from numpy import meshgrid, arange, ndarray, ceil, floor
    from scipy import sparse
    number_of_scanning_boxes_x = ceil((image_size_x + scanning_box_step_x - scanning_box_size_x) / scanning_box_step_x)
    number_of_scanning_boxes_y = ceil((image_size_y + scanning_box_step_y - scanning_box_size_y) / scanning_box_step_y)

    scanning_boxes_matrix_particles = sparse.lil_matrix((int(number_of_scanning_boxes_x),
                                                         int(number_of_scanning_boxes_y)))
    scanning_boxes_matrix_predicted_particles = sparse.lil_matrix((int(number_of_scanning_boxes_x),
                                                                   int(number_of_scanning_boxes_y)))

    for particle_position_x, particle_position_y in zip(particle_positions_x, particle_positions_y):
        start_index_x = int(ceil((particle_position_x - scanning_box_size_x) / scanning_box_step_x))
        stop_index_x = int(floor(particle_position_x / scanning_box_step_x))

        start_index_y = int(ceil((particle_position_y - scanning_box_size_y) / scanning_box_step_y))
        stop_index_y = int(floor(particle_position_y / scanning_box_step_y))

        for index_x in range(start_index_x, stop_index_x + 1):
            for index_y in range(start_index_y, stop_index_y + 1):
                try:
                    scanning_boxes_matrix_particles[index_x, index_y] += 1
                except IndexError:
                    continue

    for predicted_particle_position_x, predicted_particle_position_y in zip(predicted_particle_positions_x,
                                                                            predicted_particle_positions_y):
        start_index_x = int(ceil((predicted_particle_position_x - scanning_box_size_x) / scanning_box_step_x))
        stop_index_x = int(floor(predicted_particle_position_x / scanning_box_step_x))

        start_index_y = int(ceil((predicted_particle_position_y - scanning_box_size_y) / scanning_box_step_y))
        stop_index_y = int(floor(predicted_particle_position_y / scanning_box_step_y))

        for index_x in range(start_index_x, stop_index_x + 1):
            for index_y in range(start_index_y, stop_index_y + 1):
                try:
                    scanning_boxes_matrix_predicted_particles[index_x, index_y] += 1
                except IndexError:
                    continue

    n = sparse.csr_matrix(scanning_boxes_matrix_particles)
    m = sparse.csr_matrix(scanning_boxes_matrix_predicted_particles)

    P = sparse.csr_matrix.sum(n)
    N = number_of_scanning_boxes_x * number_of_scanning_boxes_y - sparse.csr_matrix.sum(n > 0)
    TP = sparse.csr_matrix.sum(sparse.csr_matrix.minimum(n, m))
    FP = sparse.csr_matrix.sum(sparse.csr_matrix.maximum(n, m) - n)
    n_nonzero_boolean = n > 0
    m_nonzero_boolean = m > 0
    TN = number_of_scanning_boxes_x * number_of_scanning_boxes_y - sparse.csr_matrix.sum(n_nonzero_boolean +
                                                                                         m_nonzero_boolean)

    FN = P - TP

    return P, N, TP, FP, TN, FN


def get_op_chars(predicted_positions,
                 particle_positions_and_radiuses,
                 image_size_x, image_size_y,
                 scanning_box_size_x, scanning_box_size_y,
                 scanning_box_step_x, scanning_box_step_y):
    
    operating_characteristics_sum = [0,0,0,0,0,0]
    for i in range(len(particle_positions_and_radiuses)):
        
        current_particle_positions_x = particle_positions_and_radiuses[i][0]
        current_particle_positions_y = particle_positions_and_radiuses[i][1]
        
        current_predicted_particle_positions_x = []
        current_predicted_particle_positions_y = []
        for pred in predicted_positions:
            if(pred[0] == i):
                current_predicted_particle_positions_x.append(pred[1])
                current_predicted_particle_positions_y.append(pred[2])
                
    
        current_operating_characteristics_optimized = get_operating_characteristics_scanning_box_optimized(current_predicted_particle_positions_x,
                                                                                                 current_predicted_particle_positions_y,
                                                                                                 current_particle_positions_x, 
                                                                                                 current_particle_positions_y,
                                                                                                 image_size_x, image_size_y,
                                                                                                 scanning_box_size_x, 
                                                                                                 scanning_box_size_y,
                                                                                                 scanning_box_step_x,
                                                                                                 scanning_box_step_y)
        operating_characteristics_sum = [sum(x) for x in zip(operating_characteristics_sum, current_operating_characteristics_optimized)]
    

    operating_characteristics = [x / len(particle_positions_and_radiuses) for x in operating_characteristics_sum]
        
    return tuple(operating_characteristics)


def get_optimal_parameters(predicted_positions_wrt_frame,
                           particle_positions_and_radiuses,
                           image_size_x, image_size_y,
                           sample_size=100,
                           number_of_iterations = 2,
                           x0=[20,10],
                           verbose = False):
    
    
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit

    
    def func(x, label, pred, sample_size,image_size_x, image_size_y):
        predicted_positions_DT = benchmark.get_predicted_positions_DT(x[0],
                                                                      x[1],
                                                                      sample_size,
                                                                      pred,
                                                                      verbose = False)
        
        scanning_box_size_x = image_size_x / 12
        scanning_box_size_y = image_size_y / 12
        scanning_box_step_x = scanning_box_size_x / 4
        scanning_box_step_y = scanning_box_size_y / 4
        
        operating_characteristics = get_op_chars(predicted_positions_DT,
                                                 label,
                                                 image_size_x, image_size_y,
                                                 scanning_box_size_x, scanning_box_size_y,
                                                 scanning_box_step_x, scanning_box_step_y)
        
        print(operating_characteristics)
        
        
        dist = benchmark.distance_from_upper_left_corner_ROC(operating_characteristics, FPR_weight=100.0)
        
        return dist

    sample_size = min([sample_size, predicted_positions_wrt_frame.shape[0]])

   
    label = particle_positions_and_radiuses
    pred = predicted_positions_wrt_frame
    current_guess = minimize(func, x0, args=(label, pred, sample_size,image_size_x, image_size_y), tol=1e-6, method='Nelder-Mead').x

    for k in range(number_of_iterations):
        temp = minimize(func, current_guess, args=(label, pred, sample_size, image_size_x, image_size_y), tol=1e-6, method='Nelder-Mead').x
        current_guess = temp
        if(verbose):
            func_value = func(current_guess, label, pred, sample_size,image_size_x, image_size_y)
            print("On iteration: {}, Current_guess: {} , func_value: {}".format(k,current_guess, func_value))
        

    return current_guess


def distance_from_upper_left_corner_ROC(operating_characteristics, FPR_weight=1):
    """
    Method that calculates the distance from the upper left corner of the ROC space for a given TPR and FPR
    Inputs:
        TPR - True Positive Rate
        FPR - False Positive Rate
        FPR_weight - instead of just FPR, FPR*FPR_weight is used in the calculation. This is usually a value between 0
                     and 1. A small value means that changes in FPR affect the distance more.
    """
    P, N, TP, FP, TN, FN = operating_characteristics

    TPR = TP / P
    FPR = FP / N

    distance = ((1 - TPR) ** 2 + (FPR * FPR_weight) ** 2) ** 0.5
    return distance


def distance_upper_left_corner_ROC_predictions_and_label_unet(cutoff, label, pred, sample_size,image_size_x, image_size_y,FPR_weight = 1):
        
        sample_size = min([sample_size, pred.shape[0]])
        
        predicted_positions_unet = []
        predicted_positions_unet = get_predicted_positions_unet(sample_size,
                                                                      pred,
                                                                      image_size_x,
                                                                      image_size_y,
                                                                      cutoff_value = cutoff)
        
        
        scanning_box_size_x = 3
        scanning_box_size_y = 3
        scanning_box_step_x = 1
        scanning_box_step_y = 1
        
        operating_characteristics = (0,0,0,0,0,0)
        operating_characteristics = get_op_chars(predicted_positions_unet,
                                                 label,
                                                 image_size_x, image_size_y,
                                                 scanning_box_size_x, scanning_box_size_y,
                                                 scanning_box_step_x, scanning_box_step_y)
        
        
        dist = distance_from_upper_left_corner_ROC(operating_characteristics, FPR_weight=FPR_weight)
              
        return dist


def distance_upper_left_corner_ROC_predictions_and_label_DT(params, label, pred, sample_size,image_size_x, image_size_y,FPR_weight = 1):
        
        sample_size = min([sample_size, pred.shape[0]])
        
        predicted_positions_DT = []
        predicted_positions_DT = get_predicted_positions_DT(params[0],
                                                               params[1],
                                                               sample_size,
                                                               pred)
        
        scanning_box_size_x = 3
        scanning_box_size_y = 3
        scanning_box_step_x = 1
        scanning_box_step_y = 1
        
        operating_characteristics = (0,0,0,0,0,0)
        operating_characteristics = get_op_chars(predicted_positions_DT,
                                                 label,
                                                 image_size_x, image_size_y,
                                                 scanning_box_size_x, scanning_box_size_y,
                                                 scanning_box_step_x, scanning_box_step_y)
        
        P, N, TP, FP, TN, FN = operating_characteristics

        TPR = TP / P
        FPR = FP / N
        
        
        dist = distance_from_upper_left_corner_ROC(operating_characteristics, FPR_weight=FPR_weight)
              
        return dist


def centroids_DT(
        particle_positions_x,
        particle_positions_y,
        particle_radial_distance,
        particle_interdistance,
    ):
    import numpy as np

    particle_number = 0
    particle_index = 0
    particle_numbers = -np.ones(len(particle_positions_x))

    # Sort all predicted points to correct particle 
    while particle_numbers[np.argmin(particle_numbers)] == -1:
        particle_index = np.argmin(particle_numbers)
        particle_numbers[particle_index] = particle_number
        particle_number += 1

        for j in range(len(particle_positions_x)):

            if (particle_positions_x[j] - particle_positions_x[particle_index]) ** 2 \
                    + (particle_positions_y[j] - particle_positions_y[particle_index]) ** 2 \
                    < particle_interdistance ** 2:
                particle_numbers[j] = particle_numbers[particle_index]

    centroid_x = np.zeros(int(np.amax(particle_numbers)) + 1)
    centroid_y = np.zeros(int(np.amax(particle_numbers)) + 1)

    particle_number = 0
    while max(particle_numbers) >= particle_number:
        points_x = particle_positions_x[np.where(particle_numbers == particle_number)]
        points_y = particle_positions_y[np.where(particle_numbers == particle_number)]
        distance_from_center = particle_radial_distance[np.where(particle_numbers == particle_number)]

        # Calculate centroids
        _len = len(points_x)
        centroid_x[particle_number] = sum(points_x) / _len
        centroid_y[particle_number] = sum(points_y) / _len

        particle_number += 1

    return (centroid_x, centroid_y)


def get_predicted_positions_DT(particle_radial_distance_threshold,
                               particle_maximum_interdistance,
                               number_frames_to_be_tracked,
                               predicted_positions_wrt_frame,
                               verbose = True):
    import numpy as np
    import matplotlib.pyplot as plt

    particle_radial_distance = []
    predicted_positions = []

    for i in range(number_frames_to_be_tracked):

        particle_positions_x = []
        particle_positions_y = []

        # Threshold the radial distance of the predicted points
        for j in range(0, predicted_positions_wrt_frame.shape[1]):
            for k in range(0, predicted_positions_wrt_frame.shape[2]):
                if predicted_positions_wrt_frame[i, j, k, 2] < particle_radial_distance_threshold:
                    particle_positions_x = \
                        np.append(particle_positions_x,
                                  predicted_positions_wrt_frame[i, j, k, 0])
                    particle_positions_y = \
                        np.append(particle_positions_y,
                                  predicted_positions_wrt_frame[i, j, k, 1])
                    particle_radial_distance = \
                        np.append(particle_radial_distance,
                                  predicted_positions_wrt_frame[i, j, k, 2])


        if (len(particle_positions_x) == 0):
            if(verbose):print('particle_radial_distance_threshold too small, no predictions passed')
        else:    
            # Calculate the centroid positions
            (centroids_x, centroids_y) = centroids_DT(particle_positions_x,
                                                    particle_positions_y,
                                                    particle_radial_distance,
                                                    particle_maximum_interdistance)

            for k in range(len(centroids_x)):
                predicted_positions.append((i, centroids_x[k], centroids_y[k]))

    return predicted_positions


def get_predicted_positions_unet(number_frames_to_be_tracked, batch_predictions,video_width, video_height, cutoff_value = 0.9):
    import imageGeneration as IG
    predicted_positions = []
    predictions = IG.cutoff(batch_predictions,cutoff_value, apply_sigmoid=True)

    for i in range(number_frames_to_be_tracked):

        (x_mean_list, y_mean_list, r_mean_list, i_mean_list) = IG.get_particle_positions_radii_and_intensities(predictions[i])

        for j in range(len(x_mean_list)):
            x_position = x_mean_list[j]
            y_position = y_mean_list[j]
            frame_index = i
            if(x_position <= video_width and y_position <= video_height):
                predicted_positions.append((frame_index,x_position,y_position))
    
    return predicted_positions


def hits_and_misses(number_frames_to_be_tracked, predicted_positions, particle_positions_and_radiuses, long_return = False):
    from numpy import sqrt, zeros

    nr_predictions = zeros(number_frames_to_be_tracked)
    nr_total_predictions = 0
    nr_real_particles = zeros(number_frames_to_be_tracked)
    nr_true_positives = zeros(number_frames_to_be_tracked)
    nr_false_positives = zeros(number_frames_to_be_tracked)
    true_positives = []
    true_positive_links = []
    false_positives = []
    MAE_distance = 0
    MSE_distance = 0


    for i in range(int(number_frames_to_be_tracked)):
        predictions_for_frame_i = []
        for j in range(len(predicted_positions)):
            if(predicted_positions[j][0] == i):
                predictions_for_frame_i.append(predicted_positions[j])
        
        nr_predictions[i] = len(predictions_for_frame_i)
        nr_real_particles[i] = len(particle_positions_and_radiuses[i][0])
        
        for j in range(int(nr_predictions[i])):
            hasHit = False
            best_match = (None,None,None)
            shortest_distance = 9999999
            
            for part in range(int(nr_real_particles[i])):
                distance_x = (predictions_for_frame_i[j][1] - particle_positions_and_radiuses[i][0][part])**2
                distance_y = (predictions_for_frame_i[j][2] - particle_positions_and_radiuses[i][1][part])**2
                distance = sqrt(distance_x + distance_y)

                if(distance < particle_positions_and_radiuses[i][2][part]):
                    hasHit = True
                    
                    
                    if(distance < shortest_distance):
                        shortest_distance = distance
                        best_match = predictions_for_frame_i[j]
                        best_match_index = part

            if(hasHit):
                MAE_distance += shortest_distance
                MSE_distance += shortest_distance**2
                nr_true_positives[i] += 1 
                true_positives.append(best_match)
                true_positive_links.append((i,int(nr_total_predictions + j),best_match_index))
            else: 
                nr_false_positives[i] += 1
                false_positives.append(predictions_for_frame_i[j])
            
        nr_total_predictions += nr_predictions[i]
                        
                    
    if(long_return):
        if(len(true_positives) != 0):
            MAE_distance = MAE_distance/len(true_positives)
            MSE_distance = MSE_distance/len(true_positives)
        else:
            MAE_distance = -1
            MSE_distance = -1
        return nr_real_particles,nr_predictions,nr_true_positives,nr_false_positives,true_positives, false_positives, true_positive_links,MAE_distance,MSE_distance
    else:
        return nr_real_particles,nr_predictions,nr_true_positives,nr_false_positives


def visualize_hits_and_misses(number_frames_to_be_tracked, frames,  particle_positions_and_radiuses, predicted_positions, FP,links):
    import matplotlib.pyplot as plt
    
    ### Visualize tracked frames
    for i in range(number_frames_to_be_tracked):
        
        # Show frame
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(frames[i], cmap='gray', vmin=0, vmax=1)
        nr_real_particles = len(particle_positions_and_radiuses[i][0])

        for j in range(int(len(predicted_positions))):
            if(predicted_positions[j][0] == i):
                if(predicted_positions[j] in FP):
                    # Plot the predicted points
                    plt.plot(predicted_positions[j][2],
                            predicted_positions[j][1], '.r')
                else:
                    # Plot the predicted points
                    plt.plot(predicted_positions[j][2],
                            predicted_positions[j][1], '.b')
                    
        found_particles = []
        for item in links:
            if(item[0] == i):
                found_particles.append((item[0],item[2]))
        
        for part in range(nr_real_particles):
            if((i,part) in found_particles):
                plt.plot(particle_positions_and_radiuses[i][1][part],
                                particle_positions_and_radiuses[i][0][part], '^g')
                circle = plt.Circle((particle_positions_and_radiuses[i][1][part], particle_positions_and_radiuses[i][0][part]), particle_positions_and_radiuses[i][2][part], color='g', fill=False)
                plt.gcf().gca().add_artist(circle)
            else:
                plt.plot(particle_positions_and_radiuses[i][1][part],
                                    particle_positions_and_radiuses[i][0][part], '^m')
                circle = plt.Circle((particle_positions_and_radiuses[i][1][part], particle_positions_and_radiuses[i][0][part]), particle_positions_and_radiuses[i][2][part], color='m', fill=False)
                plt.gcf().gca().add_artist(circle)
        
        
        for link in links:
            if(link[0] == i):
                plt.plot([predicted_positions[link[1]][2],particle_positions_and_radiuses[i][1][link[2]]],
                                [predicted_positions[link[1]][1],particle_positions_and_radiuses[i][0][link[2]]], 'g',linestyle = '-')
            

def construct_video_from_images(number_of_images_to_save = 10,pathIn = './images/',pathOut = 'output.mp4',fps = 25):
    import cv2
    import matplotlib.pyplot as plt
    import os
    from os.path import isfile, join

    size = 0
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    #for sorting the file names properly
    files.sort(key = lambda x: int(x[5:-4]))

    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        #inserting the frames into an image array
        frame_array.append(img)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])

    out.release()
    cv2.destroyAllWindows()


def get_optimal_parameters_DT(predicted_positions_wrt_frame,
                           particle_positions_and_radiuses,
                           image_size_x, image_size_y,
                           sample_size=100,
                           number_of_iterations = 2,
                           x0=[10,10],
                           verbose = False,
                           FPR_weight = 1):
    
    
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit

    sample_size = min([sample_size, predicted_positions_wrt_frame.shape[0]])

   
    label = particle_positions_and_radiuses
    pred = predicted_positions_wrt_frame
    current_guess = minimize(distance_upper_left_corner_ROC_predictions_and_label_DT, x0, args=(label, pred, sample_size,image_size_x, image_size_y,FPR_weight), tol=1e-6, method='Nelder-Mead').x
    if(verbose):
            func_value = distance_upper_left_corner_ROC_predictions_and_label_DT(current_guess, label, pred, sample_size,image_size_x, image_size_y,FPR_weight)
            print("On iteration: {}, Current_guess: {} , func_value: {}".format(0,current_guess, func_value))
            
    for k in range(number_of_iterations):
        temp = minimize(distance_upper_left_corner_ROC_predictions_and_label_DT, current_guess, args=(label, pred, sample_size, image_size_x, image_size_y,FPR_weight), tol=1e-6, method='Nelder-Mead').x
        current_guess = temp
        if(verbose):
            func_value = distance_upper_left_corner_ROC_predictions_and_label_DT(current_guess, label, pred, sample_size,image_size_x, image_size_y,FPR_weight)
            print("On iteration: {}, Current_guess: {} , func_value: {}".format(k+1,current_guess, func_value))
        

    return current_guess


def get_optimal_parameters_unet(batch_predictions,
                           particle_positions_and_radiuses,
                           image_size_x, image_size_y,
                           sample_size=100,
                           number_of_iterations = 2,
                           x0=0.8,
                           FPR_weight = 10,
                           verbose = False):
    
    
    import numpy as np
    from scipy.optimize import minimize
    
    sample_size = min([sample_size, batch_predictions.shape[0]])
   
    label = particle_positions_and_radiuses
    pred = batch_predictions 
    
    
    current_guess = minimize(distance_upper_left_corner_ROC_predictions_and_label_unet,
                             x0,
                             args=(label, pred, sample_size,image_size_x, image_size_y,FPR_weight),
                             tol=1e-6,
                             method = "Nelder-Mead"
                             ).x
    
    
    if(verbose):
            func_value = distance_upper_left_corner_ROC_predictions_and_label_unet(current_guess,
                                                                                   label,
                                                                                   pred,
                                                                                   sample_size,
                                                                                   image_size_x,
                                                                                   image_size_y,
                                                                                   FPR_weight)
            
            print("On iteration: {}, Current_guess: {} , func_value: {}".format(0,current_guess, func_value))
    
    if(current_guess > 1): current_guess = 0.97
    
    bnds = [(0,1)]
    for k in range(number_of_iterations):
        temp = minimize(distance_upper_left_corner_ROC_predictions_and_label_unet, 
                        current_guess,
                        args=(label, pred, sample_size, image_size_x, image_size_y,FPR_weight),
                        tol=1e-9,
                        method='L-BFGS-B',
                        bounds = bnds
                        ).x
        
        current_guess = temp
        if(verbose):
            func_value = distance_upper_left_corner_ROC_predictions_and_label_unet(current_guess,
                                                                                   label,
                                                                                   pred, 
                                                                                   sample_size,
                                                                                   image_size_x,
                                                                                   image_size_y,
                                                                                   FPR_weight)
            print("On iteration: {}, Current_guess: {} , func_value: {}".format(k+1,current_guess, func_value))
        

    return current_guess
