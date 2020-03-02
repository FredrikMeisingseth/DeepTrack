def calculate_variance(coordinate_batch):
    #input: coordinate_batch: (n,m,2)-list for n samples with m number of particles
    #
    for i in range:
        for t in range(1,coordinate_batch.shape[0]-1):
            Print('not done')
            # result = result + (coordinate_batch[t,,i+1]-coordinate_batch[t-1])(coordinate_batch[i]-coordinate_batch[i-1])
	
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
    #cost_matrix = pad(cost_matrix, ((0,0),(0, max(len(coord_list)-len(old_coord_list),0))))

    #print(cost_matrix)
    return (cost_matrix.tolist(), old_coord_list.tolist())


def sort_particles(coord_batch, max_dist=100):
    #Takes cooridnates predicted from a batch and sorts them into a list of tracks containing all coordinate pairs belonging to that track in order of time.
    #Input: cooridnate batch: [[coord_list1]...[coord_listn]]
    #Output: track batch: []
    #Invariant: index of coordinate_lists and matching lists should always match index of track_batch_temp

    from scipy.optimize import linear_sum_assignment
    from numpy import zeros, delete

    track_batch = []
    track_batch_temp = [[coord] for coord in coord_batch[0]]
    
    coord_list_prev = coord_batch[0]

    for coord_list in coord_batch[1:]:
        dist_matrix = zeros((len(coord_list), len(coord_list_prev)))
        matching_prev = zeros(len(coord_list_prev),dtype=bool) #Keep track of old coords without matching new coords -> Ending track
        matching_new = zeros(len(coord_list), dtype=bool) #Keep track of new coords without matching old -> Starting track

        for i in range(len(coord_list)):
            for j in range(len(coord_list_prev)):
                dist = (coord_list[i][0]-coord_list_prev[j][0])**2+(coord_list[i][1]-coord_list_prev[j][1])**2
                dist_matrix[i,j]=dist
                
                if dist<max_dist:
                    matching_prev[j]=True
                    matching_new[i]=True

        deleted_coords=0
        for match, index in zip(matching_prev, range(len(coord_list_prev))):
            if not match:
                track_batch.append(track_batch_temp.pop(index-deleted_coords))
                dist_matrix = delete(dist_matrix,index-deleted_coords,1)
                deleted_coords=deleted_coords+1

            #If there is no new matching coordinate, move that track from the temp tracks to tracks, remove column from dist_matrix

        new_coords = []
        for match, index in zip(matching_new, range(len(coord_list))):
            if not match:
                track_batch_temp.append([])
                new_coords.append(coord_list.pop(index-len(new_coords)))
                dist_matrix = delete(dist_matrix,index-len(new_coords),0)
            #If there is no previous matching coordinate, append that track to temp tracks, move element last in coord_list

        (row_index,col_index) = linear_sum_assignment(dist_matrix)

        coord_list = [coord_list[index] for index in col_index]
        coord_list.extend(new_coords)

        for track, coord in zip(track_batch_temp, coord_list):
            track.append(coord)
        
        coord_list_prev = coord_list

    while(track_batch_temp):
        track_batch.append(track_batch_temp.pop())

    return track_batch