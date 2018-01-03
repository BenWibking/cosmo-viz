import numpy as np
import pynbody

# cython import 'density' module; need to `pip install cython --user' if not done already
import pyximport; pyximport.install()
import cic_density as density

def compute_FFT_densities(particles_filename, output_filename,
                          ngrid=1000, rescale=1.0, do_tanh_stretch=True,
                          interpolation='lanczos',
                          pixel_size=2560):
    # read particles
    particles_file = pynbody.load(particles_filename)
    particles = particles_file['pos']

    # determine boxsize
    particles_x = particles[:,0]
    boxsize = np.abs(np.max(particles_x) - np.min(particles_x))

    # determine the line-of-sight depth of the 2d projection
    zmin = 0.8 * boxsize
    zmax = 1.0 * boxsize
    mask = np.logical_and(particles[:,2] < zmax, particles[:,2] > zmin)
    particles_x = particles[:,0][mask]
    particles_y = particles[:,1][mask]

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
    fig,ax = plt.subplots(1,1)
    dpi = float(fig.get_dpi())
    pixels = float(pixel_size)
    fig.set_size_inches([pixels/dpi,pixels/dpi])
    a=fig.gca()
    a.set_frame_on(False)
    a.set_xticks([])
    a.set_yticks([])
    plt.axis('off')

    # compute 2d density field
    rhogrid = np.empty((ngrid,ngrid),dtype=np.float32)
    rhogrid = np.array(density.cic_2d_grid(particles_x,particles_y,rhogrid,ngrid,boxsize))
    rhogrid *= rescale
    if do_tanh_stretch == True: # apply density stretch so it looks nice
        rhogrid = np.tanh(rhogrid)

    # plot 2d density field
    image = ax.imshow(rhogrid,
                      cmap=cm.get_cmap('magma'),
                      extent=[xmin,xmax,ymin,ymax],
                      interpolation=interpolation)

    # save as the given filename in desired file format (based on suffix)
    fig.savefig(output_filename, bbox_inches='tight', pad_inches=0, dpi='figure')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_particles_filename')
    parser.add_argument('output_density_filename')
    parser.add_argument('--tanh-stretch',default=True,action='store_true')
    parser.add_argument('--rescale',type=float,default=0.15)
    parser.add_argument('--ngrid',type=int,default=1000)
    parser.add_argument('--interpolation',default='lanczos')
    args = parser.parse_args()

    compute_FFT_densities(args.input_particles_filename,
                          args.output_density_filename,
                          ngrid=args.ngrid,
                          rescale=args.rescale,
                          do_tanh_stretch=args.tanh_stretch,
                          interpolation=args.interpolation)
