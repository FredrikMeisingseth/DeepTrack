def operating_characteristics(labels, predictions):
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

        if(len(particle_positions_x) == 0):
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
            predicted_positions.append((i,centroids_x[k],centroids_y[k]))
        
    return predicted_positions
        

def get_predicted_positions_unet(number_frames_to_be_tracked, batch_predictions,video_width, video_height, cutoff_value = 0.9):
    import imageGeneration as IG
    predicted_positions = []
    predictions = IG.cutoff(batch_predictions,cutoff_value, apply_sigmoid=True)

    for i in range(number_frames_to_be_tracked):

        (x_mean_list, y_mean_list, r_mean_list, i_mean_list) = IG.get_particle_centers(predictions[i])

        for j in range(len(x_mean_list)):
            x_position = x_mean_list[j]
            y_position = y_mean_list[j]
            frame_index = i
            if(x_position <= video_width and y_position <= video_height):
                predicted_positions.append((frame_index,x_position,y_position))
    
    return predicted_positions


def hits_and_misses(number_frames_to_be_tracked, predicted_positions, particle_positions_and_radiuses, return_misses = False):
    from numpy import sqrt, zeros

    nr_predictions = zeros(number_frames_to_be_tracked)
    nr_real_particles = zeros(number_frames_to_be_tracked)
    nr_hits = zeros(number_frames_to_be_tracked)
    nr_misses = zeros(number_frames_to_be_tracked)
    misses = []


    for i in range(int(number_frames_to_be_tracked)):
        predictions_for_frame_i = []
        for j in range(len(predicted_positions)):
            if(predicted_positions[j][0] == i):
                predictions_for_frame_i.append(predicted_positions[j])
        
        nr_predictions[i] = len(predictions_for_frame_i)
        nr_real_particles[i] = len(particle_positions_and_radiuses[i][0])
        
        for j in range(int(nr_predictions[i])):
            hasHit = False
            for part in range(int(nr_real_particles[i])):
                distance_x = (predictions_for_frame_i[j][1] - particle_positions_and_radiuses[i][0][part])**2
                distance_y = (predictions_for_frame_i[j][2] - particle_positions_and_radiuses[i][1][part])**2
                distance = sqrt(distance_x + distance_y)

                if(distance < particle_positions_and_radiuses[i][2][part]):
                    hasHit = True  

            if(hasHit):nr_hits[i] += 1 
            else: misses.append(predictions_for_frame_i[j])
                        
        nr_misses[i] = nr_predictions[i] - nr_hits[i]          
    

    if(return_misses):
        return nr_real_particles,nr_predictions,nr_hits,nr_misses,misses
    else:
        return nr_real_particles,nr_predictions,nr_hits,nr_misses


def visualize_hits_and_misses(number_frames_to_be_tracked, frames,  particle_positions_and_radiuses, predicted_positions, misses):
    import matplotlib.pyplot as plt
    
    ### Visualize tracked frames
    for i in range(number_frames_to_be_tracked):
        
        # Show frame
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(frames[i], cmap='gray', vmin=0, vmax=1)
        nr_real_particles = len(particle_positions_and_radiuses[i][0])

        # Threshold the radial distance of the predicted points
        for j in range(int(len(predicted_positions))):
            if(predicted_positions[j][0] == i):
                if(predicted_positions[j] in misses):
                    # Plot the predicted points
                    plt.plot(predicted_positions[j][2],
                            predicted_positions[j][1], '.r')
                else:
                    # Plot the predicted points
                    plt.plot(predicted_positions[j][2],
                            predicted_positions[j][1], '.b')
        
        for part in range(nr_real_particles):
            plt.plot(particle_positions_and_radiuses[i][1][part],
                                particle_positions_and_radiuses[i][0][part], '^m')
            circle = plt.Circle((particle_positions_and_radiuses[i][1][part], particle_positions_and_radiuses[i][0][part]), particle_positions_and_radiuses[i][2][part], color='m', fill=False)
            plt.gcf().gca().add_artist(circle)
