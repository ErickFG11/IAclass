import numpy.random as nr
import numpy as np
import math
from matplotlib import pyplot as plt

def graficos():
	#define rango de entrada
	r_min, r_max = -5.0, 5.0
	#muestra de entrada en incrementos de 0.1
	inputs = np.arange(r_min, r_max, 0.1)
	#objetivos
	results = [objective([x]) for x in inputs]
	#crear una linea de entradas contra resultados
	plt.plot(inputs, results)
	#define un valor de entrada optimo
	x_optima = 0.0
	#dibujar una linea vertical optima en la entrada optima
	plt.axvline(x=x_optima, ls='--', color='red')
	plt.show()

	#total de iteraciones
	iterations = 100
	#temperatura inicial
	initial_temp = 10
	#arreglo de iteraciones de 0 a 1
	iterations = [i for i in range(iterations)]
	#temperaturas para cada iteracion
	temperatures = [initial_temp/float(i + 1) for i in iterations]
	#graficar iteraciones vs temperatura
	plt.plot(iterations, temperatures)
	plt.xlabel('Iteration')
	plt.ylabel('Temperature')
	plt.show()

	#explore metropolis acceptance criterion for simulated annealing
	iterations = 100
	#temperatura inicial
	initial_temp = 10
	#arreglo de iteraciones de 0 a 1
	iterations = [i for i in range(iterations)]
	#temperaturas por iteracion
	temperatures = [initial_temp/float(i + 1) for i in iterations]
	#criterio de aceptacion metropolis
	differences = [0.01, 0.1, 1.0]
	for d in differences:
		metropolis = [math.exp(-d/t) for t in temperatures]
		#graficar iteraciones vs metropolis
		label = 'diff=%.2f' % d
		plt.plot(iterations, metropolis, label=label)
	# inalize plot
	plt.xlabel('Iteration')
	plt.ylabel('Metropolis Criterion')
	plt.legend()
	plt.show()

# objective function
def objective(x):
	return x[0]**2.0

# simulated annealing algorithm
def simulated_annealing(objective, limites, n_iterations, step_size, temp):
	#generar un punto inicial
	mejor = limites[:, 0] + nr.rand(len(limites)) * (limites[:, 1] - limites[:, 0])
	#evaluar el punto inicial
	mejor_eval = objective(mejor)
	#solucion actual
	curr, curr_eval = mejor, mejor_eval
	#para graficar los mejores
	scores = list()
	#algoritmo
	for i in range(n_iterations):
		#nuevo valor candidato aleatorio
		candidate = curr + nr.randn(len(limites)) * step_size
		#evaluar candidato
		candidate_eval = objective(candidate)
		#revisar mejor solucion (minimizar)
		if candidate_eval < mejor_eval:
			#guardar el nuevo punto
			mejor, mejor_eval = candidate, candidate_eval
			scores.append(mejor_eval)
			# reportar progreso
			print('>%d f(%s) = %.5f' % (i, mejor, mejor_eval))

		#si no es mejor el candidato se genera uno nuevo (vecino)
		#calcular temoeratura para epoca actual
		t = temp / float(i + 1)
		diff = candidate_eval - curr_eval
		#calcular criterio de aceptacion metropolis
		metropolis = math.exp(-diff / t)
		#revisar si se guarda el nuevo punto
		if diff < 0 or nr.rand() < metropolis:
			#guardar punto actual
			curr, curr_eval = candidate, candidate_eval
	return [mejor, mejor_eval, scores]

if __name__ == '__main__':
    #definir rangeo de entrada
	Limites = np.asarray([[-5.0, 5.0]])
	#semilla para obtener los mismo numeros aleatorios
	nr.seed(1)
	n_iterations = 1000
	step_size = 0.1
	temp = 10

	#desempeño del recocido simulado
	best, score, scores = simulated_annealing(objective, Limites, n_iterations, step_size, temp)
	print('Done!')
	print('f(%s) = %f' % (best, score))
	#graficar mejores scores
	plt.plot(scores, '.-')
	plt.xlabel('Improvement Number')
	plt.ylabel('Evaluation f(x)')
	plt.show()
