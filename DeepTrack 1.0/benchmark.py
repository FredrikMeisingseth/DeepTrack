def calculate_variance(coordinate_batch):
    #input: coordinate_batch: (n,m,2)-list for n samples with m number of particles
    #
    for i in range:
        for t in range(1,coordinate_batch.shape[0]-1):
            result = result + (coordinate_batch[t,,i+1]-coordinate_batch[t-1])(coordinate_batch[i]-coordinate_batch[i-1])
	return result/coordinate_batch.len()

def simulated_loss(coordinate_batch, image_parameter):
    return coordinate_batch-image_parameter

def track_particle_method(batch):
	return []
    #Implementation of other tracking methods


def benchmark_SNR(network, batch):
    from time import time
    import imageGeneration

    batch = zip(imageGeneration.get_batch(return_image_parameters=True))
    result = []

    for image, label, image_parameters in batch:
        t0 = time()
        r = network.evaluate(image, label, batch_size=1)
        delta_t = time()-t0
        result.append(r, delta_t, image_parameters['Signal to Noise Ratio'])

    return result

def get_cost_matrix(coord_list, old_coord_list):
    from itertools import combinations
    from numpy import zeros, pad, delete, where

    max_len = 10 #Max distance to consider new particle as same particle as in previous frame

    cost_matrix=zeros((len(coord_list), len(old_coord_list)))
    old_coord_in_frame =zeros(len(old_coord_list),dtype=bool) #Keep track of if the old coordinate is still in frame

    for i in range(len(coord_list)):
        for j in range(len(old_coord_list)):
            dist = (coord_list[i][0]-old_coord_list[j][0])**2+(coord_list[i][1]-old_coord_list[j][1])**2
            cost_matrix[i,j]=dist
            
            if dist<max_len: #Has to be atleast one particle within max_len to be reasonable to consider as 
                old_coord_in_frame[j]=True

    #Delete rows where old tracksstop
    cost_matrix = delete(cost_matrix, where(old_coord_in_frame==False), 1)
    old_coord_list = delete(old_coord_list, where(old_coord_in_frame==False),0)

    #Add rows to create new tracks
    cost_matrix = pad(cost_matrix, ((0,0),(0, max(len(coord_list)-len(old_coord_list),0))))

    return (cost_matrix.tolist(), old_coord_list.tolist())


def sort_particles(coord_batch):
    #Takes cooridnates predicted from a batch and sorts them into a list of tracks containing all coordinate pairs belonging to that track in order of time.
    #Input: cooridnate batch: [[coord_list1]...[coord_listn]]
    #Output: track batch: []

    from scipy.optimize import linear_sum_assignment

    prev_list = []
    old_coord_list = coord_batch[0]

    for coord_list in coord_batch[1:]:
        (cost_matrix, old_coord_list) = get_cost_matrix(coord_list, old_coord_list)
        (current, prev) = linear_sum_assignment(cost_matrix)
        prev_list.extend(list(zip(prev,current)))
        old_coord_list = coord_list

    track_batch = []
    #while (not len(prev_list)==0):
    #    coord = prev.pop(0)
    #        while
    #            coord = prev.pop(where(coord = )) 

    #    index()
     #   track_batch.append()

    return prev_list