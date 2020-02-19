def calculate_variance(coordinate_batch):
    #input: coordinate_batch: (n,m,2)-list for n samples with m number of particles
    #
    for i in range 
    for t in range(1,coordinate_batch.shape[0]-1):
        result = result + (coordinate_batch[t,,i+1]-coordinate_batch[t-1])(coordinate_batch[i]-coordinate_batch[i-1])
	return result/coordinate_batch.len()

def simulated_loss(coordinate_batch, image_parameter):
    return coordinate_batch-image_parameter

def track_particle_method(batch):
	return []
    #Implementation of other tracking methods

def benchmark(track_method_list, loss_function, batch):
    from time import time

    result_list = []

    for track_method in track_method_list:
        time_start = time()
    	coordinate_batch = track_method(batch)
        time_total = time()-time_start()
        
        result_list.append(loss_function(), time_total)

    return result_list


def bechmark_snr():


def benchmark_gradient()
