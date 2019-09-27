#!/usr/bin/env python

from crude import GaussWF
from crude import SlaterWF
from crude import ProdWF

from numpy import linspace
from numpy import array
from numpy import exp

import unittest
import matplotlib.pyplot as plt


class TestWF(unittest.TestCase):
    def test_GaussWF(self):
        center = array([0., 0., 0.])
        test_g = GaussWF([1, 1, center])
        #print(test_g.value(center))
        #print(test_g.derivative(center))
        self.assertEqual(test_g.value(center), 1)
        #self.assertEqual(test_g.derivative(center), 0)

    def test_SlaterWF(self):
        center = array([0., 0., 0.])
        test_g = SlaterWF([1, 1, center])
        #print(test_g.value(center))
        #print(test_g.derivative(center))
        self.assertEqual(test_g.value(center), 1)
        #self.assertEqual(test_g.derivative(center), 0)

    #    fig,ax = plt.subplots()
    #    t = linspace(-5,5,100)
    #    t = array([ [0.,0.,x] for x in t ])
    #    print(t[:,2])
    #    ax.plot(t[:,2], test_g.value(t))
    #    ax.plot(t[:,2], test_g.derivative(t))
    #    plt.show()
    def test_ProdWF(self):
        center1 = array([0., 0., 0.])
        center2 = array([0., 0., 1.])
        wf = GaussWF([1, 1, center1])
        wf2 = GaussWF([1, 1, center2])

        pwf = ProdWF(wf, wf2)
        print(wf.derivative(array([0., 0., .5])))
        print(wf2.derivative(array([0., 0., .5])))
        print(pwf.derivative(array([0., 0., .5])))
        #print(pwf.derivative(array([0., 0., 0.5])))

        center_gpt = array([0., 0., 0.5])
        m_gpt = exp(-0.5)
        self.assertEqual(pwf.value(center_gpt), m_gpt)
       #self.assertEqual(test_g.derivative(center), 0)

        fig,ax = plt.subplots()
        t = linspace(-5,5,100)
        t = array([ [0.,0.,x] for x in t ])
        #print(t[:,2])
        ax.plot(t[:,2], pwf.value(t))
        ax.plot(t[:,2], pwf.derivative(t))
        plt.show()

if __name__ == '__main__':
    unittest.main()
