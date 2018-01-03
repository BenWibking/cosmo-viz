import numpy as np
import pynbody

# cython import 'density' module; need to `pip install cython --user' if not done already
import pyximport; pyximport.install()
import cic_density as density

def compute_FFT_densities(particles_filename, output_filename, ngrid=2048):
    # read particles
    particles_file = pynbody.load(particles_filename)
    particles = particles_file['pos']

    # determine boxsize
    particles_x = particles['x'][mask]
    boxsize = np.abs(np.max(particles_x) - np.min(particles_x))

    # determine the line-of-sight depth of the 2d projection
    zmin = 0.
    zmax = 0.8 * boxsize
    mask = np.logical_and(particles['z'] < zmax, particles['z'] > zmin)
    particles_x = particles['x'][mask]
    particles_y = particles['y'][mask]

    # determine rectangle in x-y plane to plot
    xmin = 0.
    xmax = boxsize
    ymin = 0.
    ymax = boxsize
    xl = int(xmin/boxsize * ngrid)
    xh = int(xmax/boxsize * ngrid)
    yl = int(ymin/boxsize * ngrid)
    yh = int(ymax/boxsize * ngrid)
    xmin = xl / ngrid * boxsize
    xmax = xh / ngrid * boxsize
    ymin = yl / ngrid * boxsize
    ymax = yh / ngrid * boxsize

    # create plot
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    plt,ax = plt.subplots(1,1)

    # compute 2d density field
    rhogrid = np.empty((ngrid,ngrid),dtype=np.float32)
    rhogrid = np.array(density.cic_2d_grid(particles_x,particles_y,rhogrid,ngrid,boxsize))
    rhogrid = np.tanh(rhogrid) # apply density stretch so it looks nice

    # plot 2d density field
    image = ax.imshow(rhogrid, cmap=cm.get_cmap('magma'),
                      extent=[xmin,xmax,ymin,ymax],interpolation='bilinear')

    # add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(image,cax=cax)
    
    # set axes properties
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_xlabel(r'x ($h^{-1}$ Mpc)')
    ax.set_ylabel(r'y ($h^{-1}$ Mpc)')

    # remove whitespace around edges
    plt.tight_layout()

    # save in desired file format
    plt.savefig(output_filename)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_particles_filename')
    parser.add_argument('output_density_filename')
    args = parser.parse_args()

    compute_FFT_densities(args.input_particles_filename,
                          args.output_density_filename)
