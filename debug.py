from process_constants import *
from utils import *
from phantom_generator import *
import tigre.algorithms as algs

ns = 984
print(f'nx, ny is {nx}, {ny} before setting them in debug.py.')
nx_phantom = nx
ny_phantom = ny
nx = 512
ny = 512
print(f'nx, ny is {nx}, {ny} after setting them in debug.py')
nu = int(nx * 1.3)
nl = ns * nu
intensity = 1e6
angles = np.linspace(0, 2 * np.pi, ns, endpoint=False, dtype=np.float32)
use_noise = True
use_clip = True  # making this True causes recon to miss the iodine

# try just 2 materials
x = x_true[...,1:3]
nm = 2
plt.figure(figsize=(12,4))
for i in range(nm):
    plt.subplot(1,3,i+1)
    plt.imshow(x[...,i])
    plt.colorbar()
plt.tight_layout()
plt.savefig('debug/0_x.jpeg')
plt.close()
mu = mu[1:3,...]
# clip mu to start at 20 keV
mu = mu[:,10:]
# rescale mu to be in mm instead of cm
mu = mu / 10
# try just 1 material
# x = x_true[...,2][...,None]
# nm = 1
# # mu = mu[2,...]
print(f'x shape is {x.shape}, mu shape is {mu.shape}')
temp = None
S = get_S(total_intensity=intensity, nl=nl)
S = S[...,10:]

# plt.figure()
# plt.plot(y=energies[])

# Forward model
def get_c_hat_by_energy(y, S, deriv=0, mu = mu):
    # c_hat[w,l,i] = E[# photons in (w,l) at energy i]
    if y.shape[1] == 1: #nm_algorithm
        mu = mu[2][None,...]
        # mu = np.sum(mu, axis = 0)[None,...] #np.mean converge after 34000 steps, also really slow
        
    elif y.shape[1] == nm:
        pass
    # print(f'mu {mu.shape}, y {y.shape}, dot {np.dot(y,mu).shape}')
    integral = qexp(-np.dot(y,mu),deriv)
    print(f'integral shape is {integral.shape}, min is {integral.min()}, max is {integral.max()}')
    plt.figure()
    plt.imshow(integral.mean(-1).reshape((ns,-1)))
    plt.colorbar()
    plt.savefig('debug/0_integral_exp.jpeg')
    plt.close()
    print(f'mu {mu.shape}, y {y.shape}, dot {np.dot(y,mu).shape}, dot range is {np.dot(y,mu).min()} to {np.dot(y,mu).max()}')
    print(f'just water is {np.outer(y[:,0], mu[0]).shape}')
    water = np.outer(y[:,0], mu[0]).mean(-1).reshape(tigre_shape).squeeze()
    iodine = np.outer(y[:,1], mu[1]).mean(-1).reshape(tigre_shape).squeeze()
    plt.figure()
    plt.imshow(water)
    plt.colorbar()
    plt.savefig('debug/0.5_waterdot.jpeg')
    plt.close()
    plt.figure()
    plt.imshow(iodine)
    plt.colorbar()
    plt.savefig('debug/0.5_iodinedot.jpeg')
    plt.close()
    plt.figure()
    plt.imshow(np.dot(y,mu).mean(-1).reshape((ns,-1)))
    plt.colorbar()
    plt.savefig('debug/1_integral_preexp.jpeg')
    plt.close()
    return (S * qexp(-np.dot(y,mu),deriv))

def get_c_hat(y, S, deriv=0):
    # c_hat[w,l] = E[# photons in (w,l), across all energies]
    return get_c_hat_by_energy(y=y, S=S, deriv=deriv).sum(2)



real_size = phantom_size + 1
geo = tigre.geometry(mode="fan")
# Source-to-Detector and Source-to-Origin distances (not relevant for parallel beam)
geo.DSO = 625.61  # Source-to-Origin Distance
geo.DSD = 1097.6  # Source-to-Detector Distance (not used in parallel beam)
# Offset settings (default is 0, can be adjusted if necessary)
geo.offOrigin = np.array([0, 0, 0])  # Offset of the image center
geo.offDetector = np.array([0, 0])  # Offset of the detector position
# Voxel settings (image size and resolution)
# geo.nVoxel =   # 2D image, so the z-dimension is 1
# geo.sVoxel = np.array([1, image_side * 10, image_side * 10]) # size is mm
# geo.dVoxel = np.array([0.2, 0.2, 1])
# geo.sVoxel = geo.dVoxel * geo.nVoxel
geo.sVoxel = np.array([1, 220, 220])  # mm diameter of phantom plus some border  # this is the max value in y
# geo.dVoxel = geo.sVoxel / geo.nVoxel
# print('detector length' ,geo.DSD * np.tan(theta/4) * 2) # detector length
# print('voxel size', geo.dVoxel)
geo.nDetector = np.array([1, nu])  # 1D detector with 1 × nu pixels
print(geo.DSO*theta/nu, 'detector')
geo.dDetector = np.array([1.0915, 1.0965])  # Set the size of each detector pixel (mm)
geo.sDetector = geo.nDetector * geo.dDetector  # total size of the detector    (mm)
# When we do the initial projections, we need to use the resolution of the phantom
geo.nVoxel = np.array([1, nx_phantom, ny_phantom])
geo.dVoxel = geo.sVoxel / geo.nVoxel

for i in range(nm):
        temp = tigre.Ax(x[...,i][None,...], geo, angles, gpuids=gpuids)
        if i == 0:
            y = temp
            print(f'y shape is {y.shape} right after tigre.Ax')
            tigre_shape = temp.shape
        else:
            y = np.concatenate([y, temp], axis = 1)
plt.figure(figsize=(12,4))
for i in range(nm):
    plt.subplot(1,3,i+1)
    plt.imshow(y[...,i,:])
    plt.colorbar()
plt.tight_layout()
plt.savefig('debug/2_y.jpeg')
plt.close()
y = y.transpose(0,2,1).reshape(-1, nm)


print(f'y shape is {y.shape}')

# temp = get_c_hat(y, S=S)

after_get_c = get_c_hat(y, S=S)
print(f'after get c shape is {after_get_c.shape}, before get c y has shape {y.shape}')
print(f'after get c max {after_get_c.max()}, after get c min {after_get_c.min()}')
plt.figure(figsize=(12,4))
for i in range(nw):
    plt.subplot(1,3,i+1)
    plt.imshow(after_get_c[i].reshape(tigre_shape).squeeze())
    plt.colorbar()
    # plt.clim(0,1000)
plt.tight_layout()
plt.savefig('debug/3_after_get_c.jpeg')
plt.close()
# temp = y
# print(after_get_c.shape, S.shape, after_get_c.max(), after_get_c.min())
# divided = (after_get_c)/(S+1e-8)
# print(divided.shape)
temp = after_get_c
S_div = S.sum(axis=-1)
print(f'S has shape {S.shape}, S_div has shape {S_div.shape}, after_get_c has shape {after_get_c.shape}')
print(f'S_div max {S_div.max()}, S_div min {S_div.min()}')
np.random.seed(0)
if use_noise:
    noisy = np.random.poisson(temp)/S_div
else:
    noisy = temp/S_div
plt.figure(figsize=(12,4))
for i in range(nw):
    plt.subplot(1,3,i+1)
    plt.imshow(noisy[i].reshape(tigre_shape).squeeze())
    plt.colorbar()
plt.tight_layout()
plt.savefig('debug/4_noise_before_clip.jpeg')
plt.close()
print(f'noisy sinogram max {noisy.max()}, min {noisy.min()}')
# temp = -np.log(temp/S_div)  # works without noise :)
clipval = np.exp(-0.2 * 100)  # largest amount of absorption; smallest expected value in sinogram. can adjust if needed
# clipval = np.finfo(np.float32).tiny
print(f'clipval is {clipval}')
if use_clip:
    noisy = np.clip(np.random.poisson(temp)/S_div, a_min=clipval, a_max=None)  # even when there is no noise, clipping causes a problem
plt.figure(figsize=(12,4))
for i in range(nw):
    plt.subplot(1,3,i+1)
    plt.imshow(noisy[i].reshape(tigre_shape).squeeze())
    plt.colorbar()
plt.tight_layout()
plt.savefig('debug/5_noise_after_clip.jpeg')
plt.close()
print(f'noisy sinogram max {noisy.max()}, min {noisy.min()}')
temp = -np.log(noisy)  # only fill in the zeros, don't add something to every pixel
# temp = -np.log(np.max([np.random.poisson(temp)/S_div, maxval2]))  # only fill in the zeros, don't add something to every pixel
# temp = -np.log(np.random.poisson(temp)/S_div + np.finfo(np.float32).tiny)
# noise in different energy bins is actually correlated (could matter if the energy bins actually overlap more, which they do in practice)
plt.figure(figsize=(12,4))
for i in range(nw):
    plt.subplot(1,3,i+1)
    plt.imshow(temp[i].reshape(tigre_shape).squeeze())
    plt.colorbar()
plt.tight_layout()
plt.savefig('debug/6_log_noise_after_clip.jpeg')
plt.close()

# When we do the reconstruction, we need to use the lower resolution to avoid discretization artifacts from the phantom
geo.nVoxel = np.array([1, nx, ny])
geo.dVoxel = geo.sVoxel / geo.nVoxel

result = None
for i in range(nw):
    # if i<2:
    #     continue
    # results_temp = tigre.Atb(np.array(y[:,i,:].reshape(tigre_shape), dtype = np.float32),geo,angles,backprojection_type="matched",gpuids=gpuids)
    method = 'ram_lak'#'hann'
    # results_temp = algs.fdk(np.array(temp[...,i].reshape(tigre_shape), dtype = np.float32), geo, angles, filter=method)
    results_temp = algs.fdk(np.array(temp[i].reshape(tigre_shape), dtype = np.float32), geo, angles, filter=method)
    # print(results_temp.shape)
    if i == 0:
        result = results_temp
    else:
        result = np.concatenate([result, results_temp])

plt.figure(figsize = (6,3))
plt.subplot(231)
plt.imshow(result[0])
plt.colorbar()
plt.subplot(232)
plt.imshow(result[1])
# import matplotlib.patches as patches
# box_size = 66
# x0 = (512 - box_size) // 2
# y0 = (512 - box_size) // 2
# rect = patches.Rectangle(
#     (x0, y0), box_size, box_size,
#     linewidth=2, edgecolor='red', facecolor='none'
# )
# plt.gca().add_patch(rect)
plt.colorbar()
plt.subplot(233)
# mid_point = result[2].shape[-1]//2
# plt.imshow(np.log(np.maximum(result[2],0)))
plt.imshow(result[2])
plt.colorbar()
plt.subplot(234)
plt.imshow(x[...,0])
plt.colorbar()
plt.subplot(235)
plt.imshow(x[...,1])
plt.colorbar()
plt.subplot(236)
# plt.imshow(x[...,2])
plt.imshow(result.sum(axis=0))
plt.colorbar()
plt.tight_layout()
plt.savefig(f'debug/7_{ns}_{intensity:0.0e}_FBP_result_before_sum.jpeg')
plt.close()


result = result.sum(axis=0).reshape(nx, ny, nm_algorithm)  # [nx, ny, nm]
# result = result / nl

plt.figure()
# plt.imshow(x[...,2])
plt.imshow(result)
plt.colorbar()
# plt.clim(0,1e-7)
plt.tight_layout()
plt.savefig(f'debug/8_{ns}_{intensity:0.0e}_FBP_result.jpeg')
plt.close()

"""
ns = 984
nx = 513
nu = nx * 2
nl = ns * nu
intensity = 1e6
angles = np.linspace(0, 2 * np.pi, ns, endpoint=False, dtype=np.float32)

# try just 2 materials
x = x_true[...,1:3]
nm = 2
mu = mu[1:3,...]
# try just 1 material
# x = x_true[...,2][...,None]
# nm = 1
# # mu = mu[2,...]
print(f'x shape is {x.shape}, mu shape is {mu.shape}')
temp = None
S = get_S(total_intensity=intensity, nl=nl)

# Forward model
def get_c_hat_by_energy(y, S, deriv=0, mu = mu):
    # c_hat[w,l,i] = E[# photons in (w,l) at energy i]
    if y.shape[1] == 1: #nm_algorithm
        mu = mu[2][None,...]
        # mu = np.sum(mu, axis = 0)[None,...] #np.mean converge after 34000 steps, also really slow
        
    elif y.shape[1] == nm:
        pass
    # print(f'mu {mu.shape}, y {y.shape}, dot {np.dot(y,mu).shape}')
    integral = qexp(-np.dot(y,mu),deriv)
    print(f'integral shape is {integral.shape}, min is {integral.min()}, max is {integral.max()}')
    plt.figure()
    plt.imshow(integral.sum(-1).reshape((ns,-1)))
    plt.colorbar()
    plt.savefig('basic_algorithms/integral_exp.jpeg')
    plt.close()
    plt.figure()
    plt.imshow(np.dot(y,mu).sum(-1).reshape((ns,-1)))
    plt.colorbar()
    plt.savefig('basic_algorithms/integral_preexp.jpeg')
    plt.close()
    return (S * qexp(-np.dot(y,mu),deriv))

def get_c_hat(y, S, deriv=0):
    # c_hat[w,l] = E[# photons in (w,l), across all energies]
    return get_c_hat_by_energy(y=y, S=S, deriv=deriv).sum(2)



real_size = phantom_size + 1
geo = tigre.geometry(mode="fan", nVoxel = np.array([1, nx, ny]))  # Set to parallel-beam geometry
# Source-to-Detector and Source-to-Origin distances (not relevant for parallel beam)
geo.DSO = 625.61  # Source-to-Origin Distance
geo.DSD = 1097.6  # Source-to-Detector Distance (not used in parallel beam)
# Offset settings (default is 0, can be adjusted if necessary)
geo.offOrigin = np.array([0, 0, 0])  # Offset of the image center
geo.offDetector = np.array([0, 0])  # Offset of the detector position
# Voxel settings (image size and resolution)
# geo.nVoxel =   # 2D image, so the z-dimension is 1
# geo.sVoxel = np.array([1, image_side * 10, image_side * 10]) # size is mm
# geo.dVoxel = np.array([0.2, 0.2, 1])
# geo.sVoxel = geo.dVoxel * geo.nVoxel
geo.sVoxel = np.array([1, 512, 512])
geo.dVoxel = geo.sVoxel / geo.nVoxel
# print('detector length' ,geo.DSD * np.tan(theta/4) * 2) # detector length
# print('voxel size', geo.dVoxel)
geo.nDetector = np.array([1, nu])  # 1D detector with 1 × nu pixels
print(geo.DSO*theta/nu, 'detector')
geo.dDetector = np.array([1.0915, 1.0965])  # Set the size of each detector pixel
geo.sDetector = geo.nDetector * geo.dDetector  # total size of the detector    (mm)


for i in range(nm):
        temp = tigre.Ax(x[...,i][None,...], geo, angles, gpuids=gpuids)
        if i == 0:
            y = temp
            tigre_shape = temp.shape
        else:
            y = np.concatenate([y, temp], axis = 1)
y = y.transpose(0,2,1).reshape(-1, nm)


print(f'y shape is {y.shape}')

# temp = get_c_hat(y, S=S)

after_get_c = get_c_hat(y, S=S)
print(f'after get c shape is {after_get_c.shape}, before get c y has shape {y.shape}')
print(f'after get c max {after_get_c.max()}, after get c min {after_get_c.min()}')
# temp = y
# print(after_get_c.shape, S.shape, after_get_c.max(), after_get_c.min())
# divided = (after_get_c)/(S+1e-8)
# print(divided.shape)
temp = after_get_c
S_div = S.sum(axis=-1)
print(f'S has shape {S.shape}, S_div has shape {S_div.shape}, after_get_c has shape {after_get_c.shape}')
print(f'S_div max {S_div.max()}, S_div min {S_div.min()}')
np.random.seed(0)
noisy = np.random.poisson(temp)/S_div
print(f'noisy sinogram max {noisy.max()}, min {noisy.min()}')
# temp = -np.log(temp/S_div)  # works without noise :)
maxval2 = np.exp(-0.2 * 100)  # largest amount of absorption; smallest expected value in sinogram. can adjust if needed
# print(f'maxval2 has shape {maxval2.shape}')
noisy = np.clip(np.random.poisson(temp)/S_div, a_min=maxval2, a_max=None)
print(f'noisy sinogram max {noisy.max()}, min {noisy.min()}')
temp = -np.log(noisy)  # only fill in the zeros, don't add something to every pixel
# temp = -np.log(np.max([np.random.poisson(temp)/S_div, maxval2]))  # only fill in the zeros, don't add something to every pixel
# temp = -np.log(np.random.poisson(temp)/S_div + np.finfo(np.float32).tiny)
# noise in different energy bins is actually correlated (could matter if the energy bins actually overlap more, which they do in practice)

result = None
for i in range(nw):
    # if i<2:
    #     continue
    # results_temp = tigre.Atb(np.array(y[:,i,:].reshape(tigre_shape), dtype = np.float32),geo,angles,backprojection_type="matched",gpuids=gpuids)
    method = 'ram_lak'#'hann'
    # results_temp = algs.fdk(np.array(temp[...,i].reshape(tigre_shape), dtype = np.float32), geo, angles, filter=method)
    results_temp = algs.fdk(np.array(temp[i].reshape(tigre_shape), dtype = np.float32), geo, angles, filter=method)
    # print(results_temp.shape)
    if i == 0:
        result = results_temp
    else:
        result = np.concatenate([result, results_temp])

plt.figure(figsize = (6,3))
plt.subplot(231)
plt.imshow(result[0])
plt.colorbar()
plt.subplot(232)
plt.imshow(result[1])
# import matplotlib.patches as patches
# box_size = 66
# x0 = (512 - box_size) // 2
# y0 = (512 - box_size) // 2
# rect = patches.Rectangle(
#     (x0, y0), box_size, box_size,
#     linewidth=2, edgecolor='red', facecolor='none'
# )
# plt.gca().add_patch(rect)
plt.colorbar()
plt.subplot(233)
mid_point = result[2].shape[-1]//2
# plt.imshow(np.log(np.maximum(result[2],0)))
plt.imshow(result[2])
plt.colorbar()
plt.subplot(234)
plt.imshow(x[...,0])
plt.colorbar()
plt.subplot(235)
plt.imshow(x[...,1])
plt.colorbar()
plt.subplot(236)
# plt.imshow(x[...,2])
plt.imshow(result.sum(axis=0))
plt.colorbar()
plt.tight_layout()
plt.savefig(f'basic_algorithms/{ns}_{intensity:0.0e}_FBP_result_2.jpeg')
plt.close()


result = result.sum(axis=0).reshape(nx, ny, nm_algorithm)  # [nx, ny, nm]
result = result / nl
"""