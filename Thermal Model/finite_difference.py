import numpy as np
import scipy
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from matplotlib import cm
import matplotlib.pyplot as plt

class FiniteDifference_1D:
    def __init__(self, Nr, R, t_final, dt, rho, cp, k, h, T_amb, Q, T_in):
        self.Nr = Nr
        self.R = R
        self.r = np.linspace(0, self.R, self.Nr) #Create radial array
        self.dr = self.r[1]-self.r[0] #Distance between node points
        self.t_final = t_final 
        self.dt = dt
        self.time_steps = int(self.t_final/self.dt)
        self.rho = rho
        self.cp = cp
        self.k = k
        self.h = h
        self.T_amb = T_amb
        self.Q = Q
        self.alpha = self.k / (self.rho * self.cp) #Thermal diffusivity
        self.S_in = self.Q/(rho*cp) #Source Term
        self.F = dt*self.alpha/self.dr**2
        self.T = T_in

    def BackwardEuler(self):

        #Implicit Scheme for time stepping the simulation

        #Check if source and temp terms are arrays or scalars, need to be arrays to solve with this implicit scheme
        if isinstance(self.S_in, float) == True:
            self.S = np.zeros(self.Nr)
            self.S[:] = self.S_in #Need to turn the source term into an array for the explciit solving
        else:
            self.S = self.S_in

        if isinstance(self.T, float):
            self.T = np.ones(self.Nr) * self.T

        #Setup the diagonals of the matrix for solving (tridiagonal for 1D)
        diagonal = np.zeros(self.Nr)
        lower = np.zeros(self.Nr-1)
        upper = np.zeros(self.Nr-1)
        b = np.zeros(self.Nr)

        #Precompute the diagonals (2nd order accurate)
        diagonal[:] = 1 + 2*self.F
        lower[:] =  self.F *self.dr/(self.r[1:]*2) - self.F
        upper[:] = -self.F * self.dr/(self.r[:-1]*2) - self.F

        #Setup the sparse matrix with the diagonals we have generated
        A = scipy.sparse.diags(
        diagonals=[diagonal, lower, upper],
        offsets = [0, -1, 1],
        shape=(self.Nr,self.Nr),
        format="lil") #Set to "lil" so we can easily insert insert boundary conditions

        #Insert left axisymmetric boundary condition representing no flux for the cylinder core (2nd order accurate)
        A[0, 0] = 1 + 7*self.F
        A[0, 1] = -8*self.F
        A[0, 2] = self.F

        #Insert right convective robin boudnary condition for cylinder edge (2nd order accurate)
        A[-1, -3] = self.F/2
        A[-1, -2] = -4*self.F
        A[-1, -1] = 1 + 3*self.dr*self.F*self.h/self.k + self.dt*self.h*self.alpha/(self.R*self.k) +7/2*self.F

        #Convert A to csr for efficient sparse solving
        A = A.tocsr()

        #Create time loop for time stepping the simulation
        for t in range(0, self.time_steps - 1):

            #Create b matrix interior nodes   
            b[1:-1] = self.T[1:-1] + self.S[1:-1]*self.dt #Add the source term to the intial temps

            #Insert left boundary condition b term
            b[0] = self.T[0] + self.S[0]*self.dt

            #Insert right boundary condition b term
            b[-1] = self.T[-1] + 3*self.dr*self.h*self.F*self.T_amb/self.k + self.dt *self.h*self.alpha*self.T_amb/(self.R*self.k) + self.S[-1] * self.dt
    
            #Solve (Using the efficient sparse solver from Scipy)
            self.T[:] = scipy.sparse.linalg.spsolve(A, b)

        return self.T
    
    def Plot(self, temp = "Celcius", OneD = True, TwoD = True,  spy = False):

        #Plot Celcius, Kelvin, or Fahrenheit (if you really want to...)
        if temp == "Celcius":
            self.T = self.T - 273.15
        elif temp == "Kelvin":
            self.T == self.T
        elif temp == "Fahrenheit":
            self.T = (self.T - 273.15) * 1.8 + 32

        #Plot the temperature as a function of radial distance at the end of the simulation
        if OneD == True:
            self.r = self.r*1000 #Convert m to mm for plot)

            plt.plot(self.r, self.T)
            plt.title("1D Radial Temperature Profile")
            plt.ylabel(f"Temperature ({temp})")
            plt.xlabel("Radial Distance from Core (mm)")

        #Create "2D" plot (considering axisymmetric diffusion) of the temperature at the end of the simulation
        if TwoD == True:

            #Get min and max temperatures
            self.min = np.min(self.T)
            self.max = np.max(self.T)

            #Get a colormap for displaying the temperature 
            self.cmap = cm.get_cmap("rainbow")
            self.norm = Normalize(vmin = self.min, vmax = self.max)

            #Create figure for the plot
            fig, ax = plt.subplots()
            fig.patch.set_facecolor("none")

            #Create concentric overlapping circles to represent the "nodes" from the FDM discretisation
            for i, rad in reversed(list(enumerate(self.r))):
                self.r_norm = self.r[i]/self.r[-1]
                self.color = self.cmap(self.norm(self.T[i]))
                self.circle = Circle((0.5,0.5), self.r_norm/2, color=self.color)
                ax.add_patch(self.circle)
                ax.axis("equal")
                ax.axis("off")

            #Add Colorbar for temp distribution
            self.cbar = plt.colorbar(plt.cm.ScalarMappable(norm = self.norm, cmap=self.cmap), ax=ax)
            self.cbar.ax.set_title("Temperature (C)")

            #Add title
            ax.set_title("2D Radial Temperature Profile")

        #Display the matrix sparseness
        if spy == True:
            plt.figure()
            plt.spy(self.A, marker='o')
            plt.title("Matrix A Coefficent Positions")

        plt.show()

    def Analytic_Test(self):
        pass
