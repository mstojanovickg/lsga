#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Find the global maximum for binary function: f(x) = y*sim(2*pi*x) + x*cos(2*pi*y)
'''

from math import sin, cos, pi

from lsga import GAEngine
from lsga.components import BinaryIndividual
from lsga.components import Population
from lsga.operators import TournamentSelection
from lsga.operators import UniformCrossover
from lsga.operators import FlipBitBigMutation

# Built-in best fitness analysis.
from lsga.analysis.fitness_store import FitnessStore
from lsga.analysis.console_output import ConsoleOutput

# Define population.
indv_template = BinaryIndividual(ranges=[(-2, 2), (-2, 2)], eps=0.001)
population = Population(indv_template=indv_template, size=50).init()

# Create genetic operators.
#selection = RouletteWheelSelection()
selection = TournamentSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitBigMutation(pm=0.1, pbm=0.55, alpha=0.6)

# Create genetic algorithm engine.
# Here we pass all built-in analysis to engine constructor.
engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation,
                  analysis=[ConsoleOutput, FitnessStore])

# Define fitness function.
@engine.fitness_register
def fitness(indv):
    x, y = indv.solution
    return y*sin(2*pi*x) + x*cos(2*pi*y)

if '__main__' == __name__:
    engine.run(ng=100)

