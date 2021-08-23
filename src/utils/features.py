import torch
import os
import shutil
import logging
from src import features, device, dataset_dir

# IDEA: implement a pytorch dataset

def load_all():
    positions_file = os.path.join(dataset_dir, 'user_positions.npy')
    samples_dir = os.path.join(dataset_dir, 'samples')

    indices = [x for x in range(NUM_SAMPLES)]
    dataset = CSIDataset(positions_file, samples_dir, indices)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader

def compute_features():
    """Compute and save features in temporary files."""
    
    os.mkdir('data/features')
    loader = load_all()
    idx = 0
    logging.info('Computing features...')
    for csi, _ in loader:
        csi= csi.to(device)
        N = csi.shape[0] # number of batches
        pdp = gen_PDP(csi)
        output = torch.zeros((N,0), device=device)
        for f in features:
            if f == 'rss':
                output = torch.cat((output, rss(pdp)), dim=-1)
            elif f == 'tof':
                output = torch.cat((output, tof(pdp)), dim=-1)
            elif f == 'pofp':
                output = torch.cat((output, power_first_path(pdp, tof(pdp))))
            elif f == 'ds':
                output = torch.cat((output, delay_spread(pdp)))
            else:
                logging.error(f'Features: unknown feature {f}')
                exit(-1)
        torch.save(output.reshape(-1), 'data/features/{:06d}.pt'.format(idx))
        logging.info('Sample {:06d} computed.'.format(idx))
        idx += 1

def del_tmp():
    """Remove saved features."""

    shutil.rmtree('data/features', ignore_errors=True)

def normalize(tensor):
    tensor = (tensor - tensor.mean()) / tensor.std()
    return tensor

def gen_PDP(csi: torch.Tensor, T: int=32*1024):
    """Return the power delay profile.

    Arguments:
    csi: tensor with dimensions (N,A,S), with batch size N, A antennas and S subcarriers
    T: size of the output time dimension

    Returns:
    pdp: power delay profile with dimensions (N,A,T)
    
    """

    cfr = csi
    cir = torch.fft.ifftshift(torch.fft.ifftn(cfr, s=(csi.shape[0], csi.shape[1], T)))
    pdp = torch.abs(cir)
    return pdp

def rss(pdp):
    """Return RSS computed from the PDP.
    
    Arguments:
    pdp: power delay profile of dimensions (N,A,T)

    Returns:
    rss: RSS with dimensionns (N, A)
    """
    
    rss = torch.sum(torch.square(pdp), dim=2)
    return rss

def tof(pdp):
    """Estimate ToF using the PDP.
    
    Arguments:
    pdp: power delay profile of dimensions (N,A,T)

    Returns:
    tof: ToF with dimensionns (N, A)
    """
    tof = torch.argmax(pdp, dim=2)
    return tof

def power_first_path(pdp, tof):
    """Estimates the power of the fist path.
    
    Estimates the power of the fist path using the ToF (WARNING: This function only does not
    work for multipath)

    Arguments:
    pdp: power delay profile of dimensions (N,A,T)
    tof: time of flight of dimenstions (N, A)

    Returns:
    pofp: power of the first path with dimensionns (N, A)
    """

    N = pdp.shape[0] # batch size
    A = pdp.shape[1] # number of antennas
    T = pdp.shape[2] # size of time dimension
    pofp = torch.zeros((N,A))
    for n in range(N):
        for a in range(A):
            pofp[n,a] = pdp[n,a,tof[n,a]]
    return pofp

def delay_spread(pdp):
    """Estimates the delay spread of the PDP.
    
    Arguments:
    pdp: power delay profile of dimensions (N,A,T)

    Returns:
    trms: the rms delay spread with dimensionns (N, A)
    """

    N = pdp.shape[0] # batch size
    A = pdp.shape[1] # number of antennas
    T = pdp.shape[2] # size of time dimension
    t = torch.arange(T)
    tau = torch.sum(torch.mul(t, pdp), dim=-1)/torch.sum(pdp, dim=-1)
    t_reshaped = t.tile((N),).reshape((N,T))
    tmp = torch.zeros((N,A,T))
    for n in range(N):
        for a in range(A):
            tmp[n,a,:] = torch.pow(t - tau[n,a], 2)
    trms = torch.sum(torch.mul(tmp, pdp), dim=2)/torch.sum(pdp, dim=-1)
    trms = torch.sqrt(trms)
    return trms
