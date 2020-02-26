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
