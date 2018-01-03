#!python
#cython: boundscheck=False
#cython: wraparound=False

cpdef float[:,:] boxwrap(float[:,:] pos, float boxsize) nogil:
    cdef Py_ssize_t i,j
    for i in xrange(pos.shape[0]):
        for j in xrange(3):
            if (pos[i,j] > boxsize):
                pos[i,j] -= boxsize
            elif (pos[i,j] < 0.):
                pos[i,j] += boxsize

    return pos

cpdef float[:,:,:] cic_grid(float[:] xpos, float[:] ypos, float[:] zpos, float[:,:,:] rhogrid, int ngrid, float boxsize) nogil:
    # return a binning of densities according to cloud-in-cell binning

    rhogrid[:,:,:] = -1.0  # must initialized manually
    
    # normalize by mean density of matter
    cdef float mass = (<float>ngrid)**3 / (xpos.shape[0])

    cdef float slab_fac = ngrid / boxsize
    cdef int ix,iy,iz,iix,iiy,iiz
    cdef float dx,dy,dz
    cdef Py_ssize_t i,j

    rhogrid[:,:,:] = -1.0

    for i in xrange(xpos.shape[0]):
        ix = <int>(slab_fac * xpos[i])
        iy = <int>(slab_fac * ypos[i])
        iz = <int>(slab_fac * zpos[i])

        iix = (ix + 1) % ngrid
        iiy = (iy + 1) % ngrid
        iiz = (iz + 1) % ngrid

        dx = slab_fac * xpos[i] - <float>ix
        dy = slab_fac * ypos[i] - <float>iy
        dz = slab_fac * zpos[i] - <float>iz

        ix = ix % ngrid
        iy = iy % ngrid
        iz = iz % ngrid

        rhogrid[ix,iy,iz]     += (1.-dx) * (1.-dy) * (1.-dz) * mass
        rhogrid[ix,iy,iiz]   += (1.-dx) * (1.-dy) * dz * mass
        rhogrid[ix,iiy,iz]   += (1.-dx) * dy * (1.-dz) * mass
        rhogrid[ix,iiy,iiz] += (1.-dx) * dy * dz * mass

        rhogrid[iix,iy,iz]     += dx * (1.-dy) * (1.-dz) * mass
        rhogrid[iix,iy,iiz]   += dx * (1.-dy) * dz * mass
        rhogrid[iix,iiy,iz]   += dx * dy * (1.-dz) * mass
        rhogrid[iix,iiy,iiz] += dx * dy * dz * mass

    return rhogrid

cpdef float[:,:] cic_2d_grid(float[:] xpos, float[:] ypos, float[:,:] rhogrid, int ngrid, float boxsize) nogil:
    # return a binning of densities according to cloud-in-cell binning

    rhogrid[:,:] = -1.0  # must initialized manually
    
    # normalize by mean density of matter
    cdef float mass = (<float>ngrid)**2 / (xpos.shape[0])

    cdef float slab_fac = (<float>ngrid) / boxsize
    cdef int ix,iy,iix,iiy
    cdef float dx,dy
    cdef Py_ssize_t i

    for i in xrange(xpos.shape[0]):
        ix = <int>(slab_fac * xpos[i])
        iy = <int>(slab_fac * ypos[i])

        iix = (ix + 1) % ngrid
        iiy = (iy + 1) % ngrid

        dx = slab_fac * xpos[i] - <float>ix
        dy = slab_fac * ypos[i] - <float>iy

        ix = ix % ngrid
        iy = iy % ngrid

        rhogrid[ix,iy]  += (1.-dx) * (1.-dy) * mass
        rhogrid[ix,iiy] += (1.-dx) * dy * mass

        rhogrid[iix,iy]  += dx * (1.-dy) * mass
        rhogrid[iix,iiy] += dx * dy * mass

    return rhogrid




