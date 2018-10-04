#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Equations for PFHub Benchmark 6 v2 with visualization of the
# chemical free energy density after convex splitting

from matplotlib import pylab as plt
import numpy as np
from sympy import collect, diff, expand, factor, simplify, symbols

c, ca, cb, p, w = symbols("c ca cb p w")

F = w * (c - ca) ** 2 * (cb - c) ** 2

f = diff(F, c)

print("F =", simplify(expand(F)))

print("f =", simplify(expand(f)))

print("f =", collect(expand(f), c))


def Fcon(c):
    ca = 0.3
    cb = 0.7
    w = 5
    return w * (c ** 4 + (ca ** 2 + 4 * ca * cb + cb ** 2) * c ** 2 + ca ** 2 * cb ** 2)


def Fexp(c):
    ca = 0.3
    cb = 0.7
    w = 5
    k = 0.09
    return w * (-2 * (ca + cb) * c ** 3 - 2 * ca * cb * (ca + cb) * c) - 0.5 * k * c


x = np.linspace(-0.1, 1.1, 201)
ycon = Fcon(x)
yexp = Fexp(x)

plt.figure()
plt.plot(x, ycon, label="con")
plt.plot(x, yexp, label="exp")
plt.plot(x, ycon + yexp, label="tot")
plt.xlim([-0.1, 1.1])
plt.ylim([-0.5, 0.5])
plt.xlabel(r"$c$")
plt.ylabel(r"$f_{\mathrm{chem}}$")
plt.legend(loc="best")
plt.savefig("fchem.png", bbox_inches="tight", dpi=400)

print("Figures generated. Doing math...")

from sympy.abc import A, B, C
from sympy.vector import CoordSys3D, Del, divergence
R = CoordSys3D('R')
delop = Del()

Phi = A*R.x*R.y + B*R.x + C*R.y
print("Phi = ", Phi)
print("gradPhi = ", delop(Phi).doit())
print("lapPhi = ", divergence(delop(Phi).doit()))
