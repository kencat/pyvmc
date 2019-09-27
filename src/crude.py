#!/usr/bin/env python
# put everything here first; structurize later
# TODO: structurize...

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linspace

import numpy as np
from numpy.random import rand
from numpy import exp
from numpy import sqrt
from numpy import pi
from numpy import array

from numpy import prod
from numpy import sum

from numpy.linalg import norm

import unittest

def slater(a):
    def value(x):
        return exp(-a*x)
    return value

def normal(a):
    mu, sigsq = a
    def value(x):
        return exp(-0.5*(x-mu)**2/sigsq) / sqrt(2*pi*sigsq)
    return value

def gauss(a):
    def value(x):
        return exp(-a*x**2)
    return value

# define the classes first, then group it under one parent
# only s(000) gauss.
class GaussWF(object):
    def __init__(self, p):
        self.c = p[0]
        self.a = p[1]
        self.center = p[2]

    def value(self, t):
        r = norm(t - self.center, axis=-1)
        return self.c * exp(-self.a * r**2)

    # this is with respect to its own t
    #def derivative(self, t):
    #    r = norm(t - self.center, axis=-1)
    #    #return -2 * t * self.p1 * self.value(t)
    #    return self.c * -2 * self.a * self.value(t) * r

class SlaterWF(object):
    def __init__(self, p):
        self.c = p[0]
        self.a = p[1]
        self.center = p[2]

    def value(self, t):
        r = norm(t - self.center, axis=-1)
        return self.c * exp(-self.a * r)

    def local_energy(self):
        pass


    # this is with respect to its own t
    #def derivative(self, t):
    #    r = norm(t - self.center, axis=-1)
    #    #return -2 * t * self.p1 * self.value(t)
    #    return self.c * -self.a * self.value(t)

# TODO: only one term? but I don't know how the general terms looks like...
class FahyJastrowWF(object):
    def __init__(self, p):
        self.p = p

    def value(self, t1, t2):
        t12 = norm(t1 - t2)
        return exp(0.5*t12/(1+self.p*t12))

    # derivative to what? unclear.
    #def derivative(self, t1, t2):
    #    pass

class ProdWF(object):
    # each components are the class stuffs
    # TODO: needs something to ensure that
    # the arguments are WFs
    def __init__(self, *wfs):
        self.wfs = wfs # a list, remember

    def value(self, *t):
        # otherwise invalid
        assert len(t) == len(wfs)
    #    return prod([ wf.value(t) for wf in self.wfs ], axis=0)

    # not relevant isn't it???
    # also technically wrong as each deriv is of diffent variables
    #def derivative(self, t):
    #    dmat = [[ wfi.value(t) if wfi != wfj else wfi.derivative(t) for wfi in self.wfs ] for wfj in self.wfs ]
    #    return sum([ prod(m) for m in dmat ], axis=0)
        #return sum([  for wf in self.wfs ])
        #return prod(comp.deriv(t))

# remember that slater is antisymmetric sum of product wfs
def det_wf(object):
    def __init__(self, comp):
        self.comp = comp

# local energies
# note: only for gaussian; perhaps should be a part of a "wf" class?
# wait until I constructcted multi-basis wavefunctions
# todo: note the share a; e_l should be the part of the wavefunction

# some of them may be wv. dependent;
# reconsider the structure after a more general wv. implementation

# all should be a list/vectors; of atomic charges, atomic
def atom_coulomb(Z_list, ra_list, ri_list):
    # TODO: can be more pythonic
    # note that here there are no double count over the double summation
    e = 0.
    a = zip(Z_list, ra_list)
    for Za, ra in a:
        for ri in ri_list:
            e += Za / norm(ra - ri)
    return e

# here the double summation is self-iterated
def elec_coulomb(r_list):
    e = 0,
    # TODO: like atom_.. also can be more pythonic
    # maybe not the most efficient approach
    ij = list(range(len(r_list)))
    for i in ij:
        for j in ij[i:]:
            e =+ 1 / norm(r_list[i] - r_list[j])
    return e

# wv. dependent; so it should be a part of wavefunction class
# as is all energies
def kinetic(r_list):
    pass
    # this is the rather imaginative one

# Also TODO: enforcement of jastrow and orbital form by cusp condition

def harm_e_l(a, x):
    return a + (0.5 - 2*a**2) * x**2

def h_e_l(a, r):
    return (-1/r) - (a*(a-2/r)/2)

# based on this guess, then the local energy components of multi-coordinates laplc element should include all of the thigns
def he_e_l(a, r1, r2):
    r12 = norm( r1 - r2 )
    r1u = r1 / norm(r1)
    r2u = r2 / norm(r2)
    b = (1 + a*r12)
    # see the r12 part? that one only applies for helium
    # generally it should be summation over ij
    # only the kinetic is less clear for now
    #return -4 + np.dot((r1u-r2u), (r1-r2)) / ( r12 * b**2 ) - 1/(r12 * b**3) - 1/(4*b**4) + 1/r12
    return -4 + np.dot((r1u-r2u), (r1-r2)) / ( r12 * b**2 ) - 1/(r12 * b**3) - 1/(4*b**4) + 1/r12

def he_wf(a):
    def value(r1, r2):
        r1d = norm(r1)
        r2d = norm(r2)
        r12 = norm(r1-r2)
        return exp(-2*r1d)*exp(-2*r2d)*exp(r12/(2*(1+a*r12)))
    return value

def A(t):
    pass

# for the ratio I can get away with just amplitude (I know it is less efficient)
# for the local energy, however, I need to calculate it properly


if __name__ == '__main__':
    n = 100 # number of walkers
    nsamp = 300
    # to be splitted for each w
    # TODO: see the pattern and implement that way
    #w = array([ rand(3) for i in range(n) ]) #h
    w = array([ [rand(3),rand(3)] for i in range(n) ]) #he
    w0 = np.copy(w)
    #print(w)
    # not efficient I know
    #a = 0.8 #h
    a = 0.2 #he
    t = 1
    steps = 0
    accept = 0
    # to do: still finding the best way to accumulate local energies
    e_l_accumulator = array([[0., 0.] for i in range(nsamp)])
    for m in range(nsamp):
        #print(w) # checking random walk
        e_l = array([ 0. for i in range(n) ])
        for i in range(n):
            # increment for each walker (alg. to be separated for each walkers)
            # for calculating acceptance ratio
            # random walker updater must be generalized for vmc -> dmc transition, and for different wfs
            steps += 1

            #r = sqrt(w[i][0]**2 + w[i][1]**2 + w[i][2]**2) #h
            r1 = w[i][0] #he
            r2 = w[i][1] #he
            #e_l[i] = h_e_l(a, r) #h
            e_l[i] = he_e_l(a, r1, r2) #he
            #e_l[i] = h_e_l(a, w[i])
            #n_wi = w[i] + t*(rand(3) - array([.5, .5, .5])) # correction for -+ directions #h
            # all electron move; consider pbyp
            n_wi = [None, None]
            n_wi[0] = w[i][0] + t*(rand(3) - array([.5, .5, .5])) #he
            n_wi[1] = w[i][1] + t*(rand(3) - array([.5, .5, .5])) #he
            n_r1 = n_wi[0]
            n_r2 = n_wi[1]
            #n_r = sqrt(n_wi[0]**2 + n_wi[1]**2 + n_wi[2]**2)
            # n_wi = w[i] + t*(rand(1) - 0.5) # correction for -+ directions
            #wvr = ( slater(a)(n_r) / slater(a)(r) ) ** 2 # TODO: truly integrate
            wvr = ( he_wf(a)(n_r1, n_r2) / he_wf(a)(r1, r2) ) ** 2
            # wvr = ( gauss(a)(n_wi) / gauss(a)(w[i]) ) **2 # TODO: time to generalize?
            metro = rand(1)
            if metro < wvr:
                accept += 1
                #w[0] = n_wi #h
                w[i][0] = n_wi[0]
                w[i][1] = n_wi[1]
        #print(e_l.mean(), e_l.std())
        e_l_accumulator[m][0] = e_l.mean()
        e_l_accumulator[m][1] = e_l.var()
    print('VMC E: {} +- {}   V: {} +- {}'.format(e_l_accumulator[:,0].mean(), e_l_accumulator[:,0].std(), e_l_accumulator[:,1].mean(), e_l_accumulator[:,1].std()))
    print('Acceptance ratio: {}'.format(accept/steps))

    equil = 0
    # let's start with 1d
    # plotting
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    #fig, ax = plt.subplots(2)
    t = range(e_l_accumulator[equil:].shape[0])
    ax.plot(t, e_l_accumulator[equil:,0])
    #ax.plot(t, gauss(a)(t))
    #ax.plot(t, harm_e_l(a, t))

    # 3d plot
    #fig = plt.figure()
    ax3d = fig.add_subplot(2,1,2, projection='3d')

    #X = np.linspace(-5,5,100)
    #Y = np.linspace(-5,5,100)
    #X, Y = np.meshgrid(X, Y)
    #ax3d.contour(X, Y, slater(a)(X**2+Y**2)**2, offset=-4, stride=0.5)

    #x0 = w[:,0]
    #y0 = w[:,1]
    #z0 = w[:,2]
    ax3d.scatter(w[:,0,0], w[:,0,1], w[:,0,2], label='electron 1') #e1
    ax3d.scatter(w[:,1,0], w[:,1,1], w[:,1,2], label='electron 2') #e2
    ax3d.legend()
    #ax3d.scatter(w[1][:,0], w[1][:,1], w[1][:,2])
    #ax = fig.add_subplot(111, projection='3d')
    #ax3d.plot_wireframe(X, Y, gauss(a)(X)*gauss(a)(Y))

    plt.show()
