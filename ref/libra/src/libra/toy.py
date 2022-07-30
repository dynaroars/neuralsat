"""
Toy Example
===========

:Author: Caterina Urban
"""
import faulthandler
faulthandler.enable()
from libra.engine.bias_analysis import BiasAnalysis, AbstractDomain

spec = 'tests/toy.txt'
nn = 'tests/toy.py'
domain = AbstractDomain.BOXES1
L = 0.25
U = 2
BiasAnalysis(spec, domain=domain, startL=L, startU=U).main(nn)
