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
    #Output: track batch: [track_1,track_2...] where track_n is a dictionary with three fields: start_frame, coordinates and particle_id
    #Invariant: index of coordinate_lists and matching lists should always match index of track_batch_temp

    from scipy.optimize import linear_sum_assignment
    from numpy import zeros, delete, pad

    track_batch = []
    track_batch_temp = [{'start_frame': 0,
                        'particle_id': particle_id,
                        'coordinates': [coord]} for particle_id, coord in enumerate(coord_batch[0])]
    
    coord_list_prev = coord_batch[0]
    particle_id=len(coord_batch[0])

    for frame_number, coord_list in enumerate(coord_batch[1:]):
        dist_matrix = zeros((len(coord_list), len(coord_list_prev)))
        matching_prev = zeros(len(coord_list_prev),dtype=bool) #Keep track of old coords without matching new coords -> Ending track

        for i in range(len(coord_list)):
            for j in range(len(coord_list_prev)):
                dist = (coord_list[i][0]-coord_list_prev[j][0])**2+(coord_list[i][1]-coord_list_prev[j][1])**2
                dist_matrix[i,j]=dist
                
                if dist<max_dist:
                    matching_prev[j]=True

        deleted_coords=0
        for match, index in zip(matching_prev, range(len(coord_list_prev))):
            if not match:
                track_batch.append(track_batch_temp.pop(index-deleted_coords))
                dist_matrix = delete(dist_matrix,index-deleted_coords,1)
                deleted_coords=deleted_coords+1

            #If there is no new matching coordinate, move that track from the temp tracks to tracks, remove column from dist_matrix

        (row_len, column_len) = dist_matrix.shape
        number_of_new_particles= max([0,row_len-column_len])
        dist_matrix = pad(dist_matrix, ((0,0),(0,number_of_new_particles)))        
        for j in range(number_of_new_particles):
            track_batch_temp.append({'start_frame': frame_number,
                                    'particle_id': particle_id,
                                    'coordinates':[]})
            particle_id = particle_id+1

        (row_index,col_index) = linear_sum_assignment(dist_matrix)
        print('ROW: ', len(row_index), ' COL: ', len(col_index), 'COORD: ', len(coord_list))
        coord_list = [coord_list[index] for index in col_index]
        
        for track, coord in zip(track_batch_temp, coord_list):
            track['coordinates'].append(coord)
        
        coord_list_prev = coord_list

    while(track_batch_temp):
        track_batch.append(track_batch_temp.pop())

    return track_batch