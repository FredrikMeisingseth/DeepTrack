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

    return P, N, TP, TN, FP, FN


def get_operating_characteristics_scanning_box(predicted_particle_positions_x, predicted_particle_positions_y,
                                               particle_positions_x, particle_positions_y,
                                               image_size_x, image_size_y,
                                               scanning_box_size_x, scanning_box_size_y):
    """ Method that returns the operating characteristics using a scanning box approach.
    Inputs:
    predicted_particle_positions_x: list of predicted x-coordinates of the particles
    predicted_particle_positions_y: list of predicted y-coordinates of the particles
    particle_positions_x: list of x-coordinates of the particles
    particle_positions_y: list of y-coordinates of the particles
    image_size_x: x-size of the image. We represent images as matrices, so this is usually the vertical side-length.
    image_size_y: y-size of the image. This is usually the horizontal side length.
    scanning_box_size_x: x-size of the scanning box. If image_size_x isn't divisible by scanning_box_size_x, the image
        is extended to make it fit the extra part of the scanning box.
    scanning_box_size_y: y-size of the scanning box. If image_size_y isn't divisible by scanning_box_size_y, the image
        is extended to make it fit the extra part of the scanning box.

    Notes:
        1) If a particle is exactly on the edge of the last scanning box, it will not be counted

    Outputs:
        operating_characteristics
    """
    from numpy import meshgrid, arange, ndarray
    scanning_box_coordinates_x, scanning_box_coordinates_y = meshgrid(arange(0, image_size_x, scanning_box_size_x),
                                                                      arange(0, image_size_y, scanning_box_size_y),
                                                                      sparse=False,
                                                                      indexing='ij')

    operating_characteristics = [0, 0, 0, 0, 0, 0]
    for scanning_box_start_x, scanning_box_start_y in zip(ndarray.flatten(scanning_box_coordinates_x),
                                                                                ndarray.flatten(scanning_box_coordinates_y)):
        scanning_box_stop_x = scanning_box_start_x + scanning_box_size_x
        scanning_box_stop__y = scanning_box_start_y + scanning_box_size_y
        particles_in_box = 0
        predicted_particles_in_box = 0

        for particle_position_x, particle_position_y in zip(particle_positions_x, particle_positions_y):
            if scanning_box_start_x <= particle_position_x < scanning_box_stop_x:
                if scanning_box_start_y <= particle_position_y < scanning_box_stop__y:
                    particles_in_box += 1

        for predicted_particle_position_x, predicted_particle_position_y in zip(predicted_particle_positions_x,
                                                                                predicted_particle_positions_y):
            if scanning_box_start_x <= predicted_particle_position_x < scanning_box_stop_x:
                if scanning_box_start_y <= predicted_particle_position_y < scanning_box_stop__y:
                    predicted_particles_in_box += 1

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


def distance_from_upper_left_corner_ROC(operating_characteristics, FPR_weight=1.0):
    """
    Method that calculates the distance from the upper left corner of the ROC space for a given TPR and FPR
    Inputs:
        TPR - True Positive Rate
        FPR - False Positive Rate
        FPR_weight - instead of just FPR, FPR*FPR_weight is used in the calculation. This is usually a value between 0
                     and 1. A small value means that changes in FPR affect the distance less.
    """
    P, N, TP, TN, FP, FN = operating_characteristics

    TPR = TP / P
    FPR = FP / N

    return ((1 - TPR) ** 2 + (FPR * FPR_weight) ** 2) ** (1 / 2)


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
                               predicted_positions_wrt_frame):
    import numpy as np
    import matplotlib.pyplot as plt

    particle_positions = []
    particle_centroids = []
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
            raise KeyError('particle_radial_distance_threshold too small, no predictions passed')

        particle_positions.append([])
        particle_positions[i].append(particle_positions_x)
        particle_positions[i].append(particle_positions_y)

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
            shortest_distance = 99999999
            
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
        MAE_distance = MAE_distance/len(true_positives)
        MSE_distance = MSE_distance/len(true_positives)
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
            

def construct_video_from_images(pathIn = './images/',pathOut = 'output.mp4',fps = 25):
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