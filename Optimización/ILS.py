import numpy as np
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.random as nr

#objective function Ackley
def objective(v):
	x, y = v
	return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

#verificar que las soluciones generadas estén dentro del espacio de búsqueda.
def dentro_limites(punto, limites):
	#enumerar todas las dimensiones del punto
	for d in range(len(limites)):
		#comprobar si está fuera de los límites de esta dimensión
		if punto[d] < limites[d, 0] or punto[d] > limites[d, 1]:
			return False
	return True

#algoritmo de escalada estocástica como soporte de ILS
def hillclimbing(objective, limites, n_iterations, step_size, punto_perturbado):
	#almacenar el punto inicial
	solution = punto_perturbado
	#evaluar el punto inicial
	solution_eval = objective(solution)
	#correr la subida de la colina
	for i in range(n_iterations):
		#tomar un punto nuevo
		candidate = None
		while candidate is None or not dentro_limites(candidate, limites):
			candidate = solution + nr.randn(len(limites)) * step_size
		#evaluar punto candidato
		candidate_eval = objective(candidate)
		#comprobar si debemos mantener el nuevo punto si no, regresa el mismo
		if candidate_eval <= solution_eval:
			solution, solution_eval = candidate, candidate_eval
	return [solution, solution_eval]

#iterated local search
def iterated_local_search(objective, limites, n_iter, step_size, n_restarts, p_size):
	#definir punto de partida
	best = None
	while best is None or not dentro_limites(best, limites):
		best = limites[:, 0] + nr.rand(len(limites)) * (limites[:, 1] - limites[:, 0])
	#evaluar el mejor punto actual
	best_eval = objective(best)
	#inicio del algoritmo ILS
	for n in range(n_restarts):
        #generar un punto inicial como una version perturbada de la ultima mejor
		punto_perturbado = None
		while punto_perturbado is None or not dentro_limites(punto_perturbado, limites):
			punto_perturbado = best + nr.randn(len(limites)) * p_size
		#realizar una búsqueda estocástica de escalada de colinas
		solution, solution_eval = hillclimbing(objective, limites, n_iter, step_size, punto_perturbado)
		#buscar nuevos mejores
		if solution_eval < best_eval:
			best, best_eval = solution, solution_eval
			print('Restart %d, best: f(%s) = %.5f' % (n, best, best_eval))
	return [best, best_eval]

if __name__ == '__main__':
    #sembrar el generador de números pseudoaleatorios
    nr.seed(1)
    #definir el rango de entrada
    Limites = np.asarray([[-5.0, 5.0], [-5.0, 5.0]])
    #definir las iteraciones totales para hill climb
    n_iter = 1000
    #definir el tamaño de paso máximo hiil climb
    s_size = 0.05
    #número total de reinicios aleatorios para ILS 
    n_restarts = 30
    #tamaño de paso de perturbación ILS
    p_size = 1.0
    #realizar la búsqueda de escalada
    best, score = iterated_local_search(objective, Limites, n_iter, s_size, n_restarts, p_size)
    print('Done!')
    print('f(%s) = %f' % (best, score))
    
    '''
    # define range for input
    r_min, r_max = -5.0, 5.0
    # sample input range uniformly at 0.1 increments
    xaxis = np.arange(r_min, r_max, 0.1)
    yaxis = np.arange(r_min, r_max, 0.1)
    # create a mesh from the axis
    x, y = meshgrid(xaxis, yaxis)
    # compute targets
    v=x,y
    results = objective(v)
    # create a surface plot with the jet color scheme
    figure = plt.figure()
    axis = figure.gca(projection='3d')
    axis.plot_surface(x, y, results, cmap='jet')
    # show the plot
    plt.show()
    '''




