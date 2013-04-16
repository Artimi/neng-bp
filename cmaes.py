#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np


class CMAES(object):

    def __init__(self, N, sigma=0.3, xmean=None):
        self.N = N
        if xmean is None:
            self.xmean = np.random.rand(N)
        else:
            self.xmean = xmean
        self.end = False
        self.sigma = sigma
        self.stopfitness = 1e-10
        self.stopeval = 1e3 * N ** 2
        # strategy parameter setting: selection
        self.lamda = 2 * int(4 + 3 * np.log(self.N))
        self.mu = self.lamda / 2
        self.weights = np.array([np.log(self.mu + 0.5) - np.log(i)
                                 for i in range(1, int(self.mu) + 1)])
        self.mu = int(self.mu)
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = 1 / np.sum(self.weights ** 2)
        # strategy parameter setting: adaptation
        self.cc = (4 + self.mueff / N) / (N + 4 + 2 * self.mueff / N)
        self.cs = (self.mueff + 2) / (N + self.mueff + 5)
        self.c1 = 2 / ((self.N + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) /
                       ((self.N + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) /
                                (self.N + 1)) - 1) + self.cs
        # initialize dynamic (internal) strategy parameters and constants
        self.pc = np.zeros((1, self.N))
        self.ps = np.zeros_like(self.pc)
        self.B = np.eye(self.N)
        self.D = np.eye(self.N)
        self.C = np.identity(self.N)
        self.eigenval = 0
        self.chiN = self.N ** 0.5 * (1 - 1 / (4 * self.N) + 1 /
                                     (21 * self.N ** 2))
        # generation loop
        self.counteval = 0
        self.iteration = 0
        #self.arx = np.empty([self.N, self.lamda])
        #self.arz = np.empty_like(self.arx)
        #self.arfitness = np.empty(self.lamda)

    def ask(self, lamda=None, xmean=None):
        if lamda is not None:
            self.lamda = lamda
        self.arz = np.random.randn(self.lamda, self.N)
        self.arx = self.xmean + self.sigma * np.dot(np.dot(self.B, self.D), self.arz.T).T
        return self.arx


    def tell(self, arfitness):
        self.arfitness = arfitness
        self.iteration += 1
        self.counteval += self.lamda
        self.arindex = np.argsort(self.arfitness)
        # sort by fitness and compute weighted mean into xmean
        self.arindex = np.argsort(self.arfitness)
        self.arfitness = self.arfitness[self.arindex]
        self.xmean = np.dot(self.arx[self.arindex[:self.mu]].T, self.weights)
        self.zmean = np.dot(self.arz[self.arindex[:self.mu]].T, self.weights)
        self.ps = np.dot((1 - self.cs), self.ps) + np.dot((np.sqrt(self.cs * (2 - self.cs) * self.mueff)), np.dot(self.B, self.zmean))
        self.hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.counteval / self.lamda)) / self.chiN < 1.4 + 2 / (self.N + 1)
        self.pc = np.dot((1 - self.cc), self.pc) + np.dot(np.dot(self.hsig, np.sqrt(self.cc * (2 - self.cc) * self.mueff)), np.dot(np.dot(self.B, self.D), self.zmean))
        # adapt covariance matrix C
        self.C = np.dot((1 - self.c1 - self.cmu), self.C) \
            + np.dot(self.c1, ((self.pc * self.pc.T)
            + np.dot((1 - self.hsig) * self.cc * (2 - self.cc), self.C))) \
            + np.dot(self.cmu,
                     np.dot(np.dot(np.dot(np.dot(self.B, self.D), self.arz[self.arindex[:self.mu]].T),
                            np.diag(self.weights)), (np.dot(np.dot(self.B, self.D), self.arz[self.arindex[:self.mu]].T)).T))
        # adapt step size sigma
        self.sigma = self.sigma * np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        # diagonalization
        if self.counteval - self.eigenval > self.lamda / (self.c1 + self.cmu) / self.N / 10:
            self.eigenval = self.counteval
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            self.D, self.B = np.linalg.eig(self.C)
            self.D = np.diag(np.sqrt(self.D))
        if self.arfitness[0] <= self.stopfitness:
            self.end = True
            self.result = self.arx[self.arindex[0]]
        

def fmin(func, N):
    cma = CMAES(N)
    while not cma.end:
        pop = cma.ask()
        values = np.empty(pop.shape[0])
        for i in xrange(pop.shape[0]):
            values[i] = func(pop[i])
        cma.tell(values)
    print "Result: {0}\nFunction value: {1}".format(cma.result, cma.arfitness[0])

