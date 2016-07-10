# Python code for solving the Schrodinger Equation in 
# one-dimentional problems using the B-splines functions
# for the implementation of the variational Rayleight-Rithz
# Method.
#######################################################################
#######################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA
from bspline import Bspline
from sys import exit

## Defino funcion que calcula los knots
def knots_sequence(grado, type, N_intervalos, beta, a, b):
	N_puntos = N_intervalos + 1;
	if type=='uniform':
		knots = np.hstack((np.hstack((np.tile(a, grado), np.linspace(a, b, N_puntos))), np.tile(b, grado)));
	elif type=='exp':
		x = np.linspace(a, b, N_puntos);
		y = np.sign(x)*np.exp(beta*np.abs(x));
		y = (x[N_puntos-1]-x[0])/(y[N_puntos-1]-y[0])*y;
		knots = np.hstack((np.hstack((np.tile(a, grado), y)), np.tile(b, grado)));

	return knots
#############################

## Defino el potencial del pozo
def V_potencial_ho(me, omega, x):
	U = 0.5*omega**2*me*x**2;
	return U

#############################

## Energia cinetica del momento angular
def L_angular(me, l_angular, x):
	L = 0.5*l_angular*(l_angular+1)/(me*x**2);
	return L

#######################################
############### MAIN ##################
#######################################

## Parametros fisicos del problema
me				 = 1; ## Masa de la particula
l_angular	 = 0.0; ## Momento angular

omega_vec = np.linspace(0.0, 10.0, 100);

## Separo las coordenadas y tomo distinta base en r y en z
Xmin = 0.0;
Xmax = 100.0;

N_intervalos_x = 100;
N_cuad = 200;
grado = 4;
kord = grado + 1;
beta = 0.0065; ## Cte de decaimiento para los knots en la distribucion exponencial
N_splines_x = N_intervalos_x + grado;
N_base_x = N_splines_x - 2;

N_dim = N_base_x; # Tamano de las matrices

archivo = "./resultados/E_vs_omega.dat"

f = open(archivo, 'w')
f.write("# Intervalo de integracion en r [{0}, {1}]\n".format(Xmin, Xmax))
f.write("# Grado de los B-splines {0}\n".format(grado))
f.write("# Num de intervalos {0} y tamano de base {1} en x\n".format(N_intervalos_x, N_base_x))
f.write("# Dimension total del espacio N_dim = N_base_r = {0}\n".format(N_dim))
f.write("# Cte de separacion de knots en dist exp beta = {0}\n".format(beta))
f.write("# Orden de la cuadratura N_cuad = {0}\n".format(N_cuad))
f.write("# Masa de la particula me = {0} en UA\n".format(me))
f.write("# Momento angular l_angular = {0}\n".format(l_angular))
f.write("# Autovalores calculados\n")


## Vector de knots para definir los B-splines, distribucion uniforme
knots = knots_sequence(grado, 'uniform', N_intervalos_x, beta, Xmin, Xmax);

## Pesos y nodos para la cuadratura de Gauss-Hermite
x, w = np.polynomial.legendre.leggauss(N_cuad);

## Escribo bien los nodos y los pesos para las integrales en r
x_nodos = np.array([]);
wx_pesos = np.array([]);
for i in range(N_intervalos_x+1):
	aux_x = 0.5*(knots[i+grado+1]-knots[i+grado])*x + 0.5*(knots[i+grado+1]+knots[i+grado]);
	aux_w = 0.5*(knots[i+grado+1]-knots[i+grado])*w;

	x_nodos = np.hstack((x_nodos, aux_x));
	wx_pesos = np.hstack((wx_pesos, aux_w));

wx_pesos = np.tile(wx_pesos, (N_splines_x, 1));

## B-splines en la coordenada r
basis = Bspline(knots, grado);
## Calculo la matriz con los bsplines y sus derivadas en r
bsx  = [basis._Bspline__basis(i, basis.p) for i in x_nodos]; # evaluo los bsplines
dbsx = [basis.d(i) for i in x_nodos];                        # evaluo las derivadas de los bsplines


## Matriz de solapamiento en r
Sx = np.dot(np.transpose(bsx), (np.transpose(wx_pesos)*bsx));
Sx = np.array([[Sx[i][j] for i in range(1,N_splines_x-1)] for j in range(1,N_splines_x-1)]);

## Matriz de energia cinetica en r
Tx = 0.5/me*np.dot(np.transpose(dbsx), (np.transpose(wx_pesos)*dbsx));
Tx = np.array([[Tx[i][j] for i in range(1,N_splines_x-1)] for j in range(1,N_splines_x-1)]);

## Matriz de momento angular al cuadrado
L = L_angular(me, l_angular, x_nodos);
L = np.tile(L, (N_splines_x, 1));
VL = np.dot(np.transpose(bsx), (np.transpose(L)*np.transpose(wx_pesos)*bsx));
VL = np.array([[VL[i][j] for i in range(1,N_splines_x-1)] for j in range(1,N_splines_x-1)]);

auval = np.zeros(N_dim);
for omega in omega_vec:

	## Matriz de energia de potencial del pozo de potencial
	# Primero en la variable r
	U = V_potencial_ho(me, omega, x_nodos);
	U = np.tile(U, (N_splines_x, 1));
	Vp_x = np.dot(np.transpose(bsx), (np.transpose(U)*np.transpose(wx_pesos)*bsx));
	Vp_x = np.array([[Vp_x[i][j] for i in range(1,N_splines_x-1)] for j in range(1,N_splines_x-1)]);
	
	## El hamiltoniano en la coordenada r es
	Hx = Tx + VL + Vp_x;

	e, auvec = LA.eigh(Hx, Sx);

	auval = np.vstack((auval, e));

	f.write("{:13.9e}   ".format(omega))
	for i in range(N_dim):
		f.write("{:13.9e}   ".format(e[i]))

	f.write("\n")

auval = np.array([[auval[i][j] for i in range(1,np.size(omega_vec)+1)] for j in range(N_dim)]);

for i in range(10):
	estado = np.zeros(np.size(omega_vec));
	for j in range(np.size(omega_vec)):
 		estado[j] = auval[i][j];

 	plt.plot(omega_vec, estado, '-')

plt.show()
f.close()
