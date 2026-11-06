import os
import time
import warnings
import numpy as np
from scipy.linalg import eigh
from numba import njit, prange
import matplotlib.pyplot as plt
from IPython.display import HTML
import matplotlib.colors as colors
import matplotlib.animation as animation

warnings.filterwarnings("ignore")

class waveguide:
    def __init__(self, params, B_signal, pump_data=None, wg_type="Planar"):
        #Parametros basicos
        self.N = params["N"]
        self.g = params["g"]
        self.alpha = params["alpha"]
        self.B_s = params["B_s"]
        self.C0 = params["C0"]
        self.points = params["points"]
        self.deltaB = params["deltaB"]
        self.wg_type = wg_type
        self.pump = pump_data
        

        self.L = params["L"] / self.C0
        self.deltaB_dis = 3 * self.C0
        self.deltaC = 0.9 * self.C0
        self.z = np.linspace(0, self.L, self.points)
        self.B_slope = self.deltaB / self.C0
        
        #Parametros avanzados
        self.B_signal = B_signal(self.N, self.B_slope, self.B_s)
        self.C = self.C0 * np.ones(self.N-1) #Homogeneous coupling profile
        if self.wg_type == "Planar":
            self.C_corner = 0

        elif self.wg_type == "Circular":
            self.C_corner = self.C0

        else:
            print("WARNING: invalid wg_type - The code will assume Planar waveguide")
            self.wg_type = "Planar"
            self.C_corner = 0
            
        #Calcula los eigenmodes y eigenvals para la configuracion inicial
        self.transform()

        
    def set_pump(self, data):
        self.pump = data

        
    def transform(self,disorder=False):
        if disorder:
            C = self.C_dis
            B_signal = self.B_signal_dis
            C_corner = self.C_corner_dis
            
        else:
            C = self.C
            B_signal = self.B_signal
            C_corner = self.C_corner
            
        #Configura el hamiltoniano
        omega = np.diag(B_signal) + np.diag(C, k=-1) + np.diag(C, k=1)
    
        #Configura la condicion de frontera periodica (si aplica el caso)
        if self.wg_type == "Circular":
            omega[0, -1] = C_corner
            omega[-1, 0] = C_corner
    
        #Calcula los eigenvectores y eigenvalores
        eigenvals, eigenmodes = eigh(omega,check_finite=False, driver='evr')
        
        if disorder:
            self.eigenvals_dis = eigenvals[::-1]
            self.eigenmodes_dis = eigenmodes[:, ::-1].T
            self.Lambda_dis = self.eigenvals_dis[:, None] + self.eigenvals_dis[None, :]
        else:
            self.eigenvals = eigenvals[::-1]
            self.eigenmodes = eigenmodes[:, ::-1].T
            self.Lambda = self.eigenvals[:, None] + self.eigenvals[None, :]


    def simulate(self, basis='individual', calc_schmidt=False, disorder=False):
        if disorder:
            B_signal = self.B_signal_dis
            eigenvals = self.eigenvals_dis
            eigenmodes = self.eigenmodes_dis
            Lambda = self.Lambda_dis
        else:
            B_signal = self.B_signal
            eigenvals = self.eigenvals
            eigenmodes = self.eigenmodes
            Lambda = self.Lambda
    
        if self.pump is None:
            print("Missing pump data")
            return
    
        #
        Q = Qs_parallel(B_signal, eigenvals, eigenmodes, Lambda, self.points, self.N, self.z, self.pump, self.g, self.alpha)
    
        #
        if basis == 'individual':
            Q = eigenmodes.conj().T @ Q @ eigenmodes
    
        #
        res = self.generate_data(Q, calc_schmidt=calc_schmidt)
    
        #
        suffix = '_dis' if disorder else ''
        for key, val in res.items():
            setattr(self, key + suffix, val)

    
    def linear(self, calc_schmidt=False, disorder=False):
        #Selecciona los eigenvalores/eigenmodos con o sin desorden
        if disorder:
            eigenvals = self.eigenvals_dis
            eigenmodes = self.eigenmodes_dis
        else:
            eigenvals = self.eigenvals
            eigenmodes = self.eigenmodes

        if self.pump is None:
            print("Missing pump data")
            return
    
        #
        phases = np.exp(1j * eigenvals[None, :] * self.z[:, None])
        idx = np.arange(self.N)
        U_s = np.zeros((self.points, self.N, self.N), dtype=np.complex128)
        U_s[:, idx, idx] = phases
    
        #
        U_i = eigenmodes.conj().T @ U_s @ eigenmodes
    
        #
        P = 2 ** (-np.eye(self.N) / 2) * self.pump
    
        #
        Q = U_i @ P @ U_i.transpose(0, 2, 1)
    
        #Calcula K, dist_prob, corr maps, part_ratio, std, std_der (y schmidt si se pide)
        res = self.generate_data(Q, calc_schmidt=calc_schmidt)
    
        #Guarda los resultados
        suffix = '_dis' if disorder else ''
        for key, val in res.items():
            setattr(self, key + suffix, val)

    
    def generate_data(self, Q, calc_schmidt=False):
        """Toma el estado de dos fotones Q ya propagado y calcula K, dist_prob,
        matrices de correlacion, part_ratio, std y std_der."""
        
        K = 2 ** (1 - np.eye(Q.shape[-1]) / 2) * Q
    
        n = np.sum((1 + np.eye(K.shape[-1])) * np.abs(K) ** 2, axis=-1)
        dist_prob = n / np.sum(n, axis=-1, keepdims=True)
    
        norm_Q = np.sum(2 ** (np.eye(Q.shape[-1]) - 1) * np.abs(Q) ** 2, axis=(-1, -2), keepdims=True)
        Q_corr = np.abs(Q) ** 2 / norm_Q
    
        norm_K = np.sum(2 ** (np.eye(K.shape[-1]) - 1) * np.abs(K) ** 2, axis=(-1, -2), keepdims=True)
        K_corr = np.abs(K) ** 2 / norm_K
    
        prob_no_nans = np.where(np.isnan(dist_prob), 0, dist_prob)
        part_ratio = 1 / np.sum(prob_no_nans ** 2, axis=-1)
    
        #Calcula la desviacion estandar planar
        if self.wg_type == "Planar":
            M = np.arange(1, dist_prob.shape[-1] + 1)
            std = np.sqrt(np.sum(M**2 * dist_prob, axis=-1) - np.sum(M * dist_prob, axis=-1)**2)
            """
        else:
            #Crea un array donde asigna la distancia angular a cada wg, partiendo de la wg[0]
            theta = 2 * np.pi * np.arange(dist_prob.shape[-1]) / dist_prob.shape[-1]
    
            #Obtiene el promedio de la distribucion
            prom = np.angle(np.sum(dist_prob * np.exp(1j * theta), axis=-1))
    
            #Calcula la diferencia angular entre las posiciones de las wg y el promedio
            delta_theta = np.angle(np.exp(1j * (theta - prom[:, np.newaxis])))
    
            #Calcula la std
            std = np.sqrt(np.sum(delta_theta**2 * dist_prob, axis=-1))
    
            #Reconvierte de notacion angular a wg
            std = std * dist_prob.shape[-1] / (2 * np.pi)
            """
        else:
            # Guía de entrada
            j0 = np.argmax(self.pump)
        
            # Índices de las guías
            j = np.arange(self.N)
        
            # Distancia geodésica en el anillo
            delta = np.abs(j - j0)
            r = np.minimum(delta, self.N - delta)
        
            # Anchura de transporte
            std = np.sqrt(np.sum(r**2 * dist_prob, axis=-1))
    
        #Calcula la derivada de la desviacion estandar
        std_der = np.gradient(np.log(std), np.log(self.C0 * self.z), edge_order=2)
    
        result = dict(Q=Q, K=K, dist_prob=dist_prob, Q_corr=Q_corr, K_corr=K_corr,
                      part_ratio=part_ratio, std=std, std_der=std_der)
    
        if calc_schmidt:
            #Calcula la matriz diagonal
            _, D, _ = np.linalg.svd(Q, full_matrices=False)
    
            #Normaliza los valores obtenidos
            lambda_vals = D**2 / np.sum(D**2, axis=-1, keepdims=True)
    
            #Genera el parametro k
            result['schmidt'] = 1 / np.sum(lambda_vals**2, axis=-1)
    
        return result
    

    def disorder(self, kap_B, kap_C, runsQty=200, linear=False, calc_schmidt=False):

        #Guarda los desordenes
        self.kap_B = kap_B
        self.kap_C = kap_C

        #Define el perfil de las constantes de acople
        f_0 = np.ones(self.N-1)  #Homogeneo

        def rand_kappa(kappa, N):
            """Genera N valores aleatorios uniformes entre -kappa y kappa."""
            return kappa * (2 * np.random.rand(N) - 1)

        #Lista donde se guardan los datos calculados
        std_der_av   = np.zeros((len(kap_C), len(kap_B), self.points))
        std_av       = np.zeros((len(kap_C), len(kap_B), self.points))
        Q_corr_av    = np.zeros((len(kap_C), len(kap_B), self.points, self.N, self.N))
        K_corr_av    = np.zeros((len(kap_C), len(kap_B), self.points, self.N, self.N))
        dist_prob_av = np.zeros((len(kap_C), len(kap_B), self.points, self.N))

        if calc_schmidt:
            schmidt_av = np.zeros((len(kap_C), len(kap_B), self.points))

        #Contadores de la duracion de la simulacion
        time_diag = 0
        time_sim = 0

        for i in range(len(kap_C)):
            for k in range(len(kap_B)):
                for j in range(runsQty):
                    
                    t0 = time.time()
    
                    #Crea el perfil de constantes de propagacion con desorden
                    self.B_signal_dis = self.B_signal + rand_kappa(kap_B[k], self.N) * self.deltaB_dis
    
                    #Crea el perfil de constantes de acople con desorden
                    self.C_dis = self.C0 * f_0 + rand_kappa(kap_C[i], self.N - 1) * self.deltaC

                    #En caso de un arreglo circular, agrega desorden a la constante periodica
                    if self.wg_type == "Circular":
                        self.C_corner_dis = self.C0 + rand_kappa(kap_C[i], 1)[0] * self.deltaC
                    else:
                        self.C_corner_dis = 0
    
                    #Obtiene los autovalores y supermodos con desorden
                    self.transform(disorder=True)
                    
                    time_diag += time.time() - t0
                    t0 = time.time()
    
                    #Realiza la simulacion de propagacion no lineal
                    if linear:
                        self.linear(disorder=True, calc_schmidt=calc_schmidt)

                    else:
                        self.simulate(disorder=True, calc_schmidt=calc_schmidt)
    
                    #Guarda los datos de la simulacion
                    std_av[i, k]       += self.std_dis
                    std_der_av[i, k]   += self.std_der_dis
                    Q_corr_av[i, k]    += self.Q_corr_dis
                    K_corr_av[i, k]    += self.K_corr_dis
                    dist_prob_av[i, k] += self.dist_prob_dis
                    if calc_schmidt:
                        schmidt_av[i, k] += self.schmidt_dis
    
                    time_sim += time.time() - t0
        
                    if kap_C[i] == 0 and kap_B[k] == 0:
                        break
                    
                if kap_C[i] == 0 and kap_B[k] == 0:
                    continue
                    
                else:
                    std_av[i, k]       = std_av[i, k]       / runsQty
                    std_der_av[i, k]   = std_der_av[i, k]   / runsQty
                    Q_corr_av[i, k]    = Q_corr_av[i, k]    / runsQty
                    K_corr_av[i, k]    = K_corr_av[i, k]    / runsQty
                    dist_prob_av[i, k] = dist_prob_av[i, k] / runsQty
                    if calc_schmidt:
                        schmidt_av[i, k] = schmidt_av[i, k] / runsQty

        #Muestra los tiempo de duracion de la simulacion
        print('Diagonalization time =', time_diag)
        print('Simulation time =', time_sim)
        print('Total time =', time_diag + time_sim)

        #Guarda los promedios obtenidos
        self.std_av       = std_av.squeeze()
        self.std_der_av   = std_der_av.squeeze()
        self.Q_corr_av    = Q_corr_av.squeeze()
        self.K_corr_av    = K_corr_av.squeeze()
        self.dist_prob_av = dist_prob_av.squeeze()
        if calc_schmidt:
            self.schmidt_av = schmidt_av.squeeze()

class B_profiles:
    @staticmethod
    def wannier_stark_beta(N, B_slope, B_s, b0=0):
        """Perfil de constantes de propagación efectiva (potencial de Wannier-Stark)"""
        return b0 + B_slope * B_s * np.linspace(0, 1, N)
        
    @staticmethod
    def uniform_beta(N, B_slope, B_s, b0=0):
        """Perfil de constantes de propagación efectiva homogéneo"""
        return b0 + B_s * np.ones(N)

class plotter():
    def __init__(self, output_dir):
        self.output_dir = output_dir
        
    def plot_eigen(self, wg_list, wg_number, var, var_name, case=None, save=False):
        """Genera una grafica con los eigenvalores y supermodos de una waveguide"""
    
        #Configuracion inicial de las graficas
        fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharey=False)
    
        #Etiqueta de cada grafica
        labels = ['(a)', '(b)']
    
        for i in range(len(wg_list)):

            #Numero de guia de onda
            index = np.arange(wg_list[i].N) + 1
            
            #Obtiene los eigenvalores, supermodos y beta_slope de cada guia de onda
            vals = wg_list[i].eigenvals / wg_list[i].C0
            modes = wg_list[i].eigenmodes[wg_number]
            Bslope = wg_list[i].B_slope
    
            #Plotea los eigenvaloes y supermodos
            axes[0].plot(index, vals,  marker='s',    linestyle='None', label = f'{var_name} = {var[i]}')
            axes[1].plot(index, np.abs(modes), marker='none', linestyle='-',    label = f'{var_name} = {var[i]}')
    
        #Configura los ejes de las graficas
        axes[0].set_ylabel(f'Eigenvalue/$C_0$', fontsize=18)
        axes[1].set_ylabel(f'Eigenmode amplitude', fontsize=18)
        axes[0].set_xlabel('Supermode', fontsize=18)
        axes[1].set_xlabel('Waveguide index', fontsize=18)
    
        #Configuracion extra de las graficas, agrega las labels
        for i in range(2):
            axes[i].legend(fontsize=17)
            axes[i].tick_params(axis='x', labelsize=17)
            axes[i].tick_params(axis='y', labelsize=17)
            axes[i].text(
            0.02, 0.98,
            labels[i],
            transform=axes[i].transAxes,
            fontsize=18,
            fontweight='bold',
            va='top',
            ha='left',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8)
        )
            
        #Si tiene nombre, se guarda
        if save:
            plt.savefig(rf'{self.output_dir}/eigenmodes_{case}.pdf', dpi=600, bbox_inches = "tight")
        plt.show()

    
    
    def plot_propagation(self, wg_list, case=None, save=False):
        fig, axes = plt.subplots(1, len(wg_list), figsize=(5*len(wg_list), 5), sharey=True)
    
        #Define la cantidad de graficas
        nQty = len(wg_list)
        
        labels = [f"({chr(97 + i)})" for i in range(nQty)]
        
        for i in range(nQty):
            N = wg_list[i].N
    
            axes[i].text(
            0.02, 0.08,
            labels[i],
            transform=axes[i].transAxes,
            fontsize=18,
            fontweight='bold',
            va='top',
            ha='left',
            color='white'
            )
    
            propagation = axes[i].pcolor(wg_list[i].dist_prob, norm=colors.LogNorm(vmin=1e-4, vmax=1))
    
            #Configura la grafica en caso de un numero impar o par de wg
            if N % 2 == 0:
                xticks = np.array([0.5, round(N/3) - 0.5, round(2*N/3) - 0.5, N - 0.5])
                xlabels = np.array([1, round(N/3), round(2*N/3), N]).astype(int)
            else:
                xticks = np.array([0.5, round(N/4) + 0.5, N/2, round(3*N/4) - 0.5, N - 0.5])
                xlabels = (xticks - N/ 2).astype(int)
    
                #Ajusta los ejes de la grafica
            axes[i].set_xticks(xticks)
            axes[i].set_xticklabels(xlabels, fontsize=18)
            axes[i].set_yticks(np.linspace(0, wg_list[i].points, 6))
            axes[i].set_yticklabels(np.round(wg_list[i].C0 * np.linspace(0, wg_list[i].L, 6), 2).astype(int), fontsize=18)
            axes[i].set_xlabel('Waveguide number', fontsize=18)
        
            #Solo el primer subplot tiene ylabel
            if i == 0:
                axes[i].set_ylabel('$C_0 z$', fontsize=18)
                
            #Título por subplot
            axes[i].set_title(f'$\\Delta\\beta / C_0  = {wg_list[i].B_slope}$', fontsize=18)
        
        #Configura la barra lateral
        cbar_ax = fig.add_axes([0.90, 0.145, 0.02, 0.78])
        cbar = fig.colorbar(propagation, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=18) 
    
        if case:
            if save:
                plt.tight_layout(rect=[0, 0, 0.9, 0.99])
                plt.savefig(rf'{self.output_dir}/{case}_propagation.pdf', dpi=600, bbox_inches = "tight")
            else:
                title = fig.suptitle(f"{case} case",fontsize=18)
                plt.tight_layout(rect=[0, 0, 0.9, 0.99])
            
        plt.show()
    
        

    def plot_corr(self, wg_list, frame=-1, corr_type="Q", case=None, save=False):
        z = wg_list[0].C0 * np.linspace(0, wg_list[0].L, wg_list[0].points)
        
        #Define la cantidad de graficas
        nQty = len(wg_list)
        
        labels = [f"({chr(97 + i)})" for i in range(nQty)]
    
        #En caso de que solo exista una waveguode
        if nQty == 1:
            axes = [axes]
    
        images = [None] * nQty

        #Crea una fila de subplots
        fig, axes = plt.subplots(
            1,
            nQty,
            figsize=(5 * nQty, 5),
            sharex=False,
            sharey=True
        )

        # Configura el plot 
        for i in range(nQty):

            #Selecciona el tipo de matriz de correlacion
            if corr_type == "Q":
                corr_data = wg_list[i].Q_corr
    
            elif corr_type == "K":
                corr_data = wg_list[i].K_corr
    
            else:
                corr_data = wg_list[i].K_corr


            images[i] = axes[i].imshow(
                corr_data[frame],
                norm=colors.LogNorm(vmin=1e-4, vmax=1),
                cmap='YlGnBu',
                origin='lower',
                aspect='auto'
            )
    
            axes[i].set_title(f'$\\Delta\\beta / C_0  = {wg_list[i].B_slope}$', fontsize=18)
            axes[i].set_xlabel('Waveguide r', fontsize=18)
            axes[i].tick_params(axis="both", labelsize=16)

            axes[i].text(
                0.01, 0.98,
                labels[i],
                transform=axes[i].transAxes,
                fontsize=18,
                fontweight='bold',
                va='top',
                ha='left',
                color='black'
            )
            
        # Colorbar compartida
        axes[0].set_ylabel('Waveguide q', fontsize=18)
        fig.tight_layout(rect=[0, 0, 0.96, 1])
        cbar = fig.colorbar(images[0], ax=axes, location="right", fraction=0.02, pad=0.02)
        cbar.set_label(r'$\Gamma_{q,r}$', fontsize=18)
        cbar.ax.tick_params(labelsize=14)
        if case:
            if save:
                plt.savefig(rf'{self.output_dir}/{case}_{corr_type}corr.pdf', dpi=600, bbox_inches = "tight")
            else:
                title = fig.suptitle(
                    rf'{case} case, {corr_type} correlation matrix, $C_0z={z[frame]:.2f}$',
                    fontsize=18
                )
        plt.show()


    def animate_corr_list(self, wg_list, corr_type="Q", case=None):
        """Anima, en paralelo, las matrices de correlacion de varios waveguides
        distintos (uno por B_slope, L, etc.) - igual que plot_corr pero animado."""
        z = wg_list[0].C0 * np.linspace(0, wg_list[0].L, wg_list[0].points)
    
        nQty = len(wg_list)
        fig, axes = plt.subplots(1, nQty, figsize=(5*nQty, 5), sharex=False, sharey=True)
        if nQty == 1:
            axes = [axes]
    
        images = [None] * nQty
        corr_data_list = []
    
        #Selecciona la matriz de correlacion de cada waveguide y arma el frame inicial
        for i in range(nQty):
            if corr_type == "Q":
                corr_data = wg_list[i].Q_corr
            elif corr_type == "K":
                corr_data = wg_list[i].K_corr
            else:
                corr_data = wg_list[i].K_corr
            corr_data_list.append(corr_data)
    
            images[i] = axes[i].imshow(
                corr_data[0],
                norm=colors.LogNorm(vmin=1e-4, vmax=1),
                cmap='YlGnBu',
                origin='lower',
                aspect='auto'
            )
            axes[i].set_title(f'$\\Delta\\beta / C_0  = {wg_list[i].B_slope}$', fontsize=18)
            axes[i].set_xlabel('Waveguide r', fontsize=18)
            axes[i].tick_params(axis="both", labelsize=16)
    
        axes[0].set_ylabel('Waveguide q', fontsize=18)
    
        if case:
            title = fig.suptitle(rf'{case} case, {corr_type} correlation matrix, $C_0z={z[0]:.2f}$', fontsize=18)
        else:
            title = fig.suptitle(rf'{corr_type} correlation matrix, $C_0z={z[0]:.2f}$', fontsize=18)
    
        fig.tight_layout(rect=[0, 0, 0.96, 1])
        cbar = fig.colorbar(images[0], ax=axes, location="right", fraction=0.02, pad=0.02)
        cbar.set_label(r'$\Gamma_{q,r}$', fontsize=18)
        cbar.ax.tick_params(labelsize=14)
    
        #Actualiza los nQty paneles a la vez en cada frame
        def update(frame):
            for i in range(nQty):
                images[i].set_data(corr_data_list[i][frame])
            if case:
                title.set_text(rf'{case} case, {corr_type} correlation matrix, $C_0z={z[frame]:.2f}$')
            else:
                title.set_text(rf'{corr_type} correlation matrix, $C_0z={z[frame]:.2f}$')
            return images
    
        ani = animation.FuncAnimation(fig, update, frames=wg_list[0].points, interval=50, blit=False)
        ani.save(f"correlation_{corr_type}_{case}.gif", writer="pillow", fps=20)
        plt.close(fig)
        plt.rcParams['animation.embed_limit'] = 100
        display(HTML(ani.to_jshtml()))
    

    
    def plot_Schmidt(self,wg_list, case=None, save=False):

        plt.figure(figsize=(10, 5))
    
        z = wg_list[0].C0 * np.linspace(0, wg_list[0].L, wg_list[0].points)

        for i in range(len(wg_list)):
        
            k_data = wg_list[i].schmidt
            plt.plot(z, k_data, label=f'$\\Delta\\beta / C_0  = {wg_list[i].B_slope}$')
        
        plt.grid()
        plt.legend(fontsize=17)
        plt.tick_params(axis='x', labelsize=17)
        plt.tick_params(axis='y', labelsize=17)
        plt.xlabel("$C_0Z$", fontsize=18)
        plt.ylabel("Parameter $K$", fontsize=18)

        if save:
            plt.savefig(rf'{self.output_dir}/{case}_Schmidt.pdf', dpi=600, bbox_inches = "tight")
        plt.show()

    
    def plot_propagation_line(self,wg_list, wg_number, case=None, save=False):

        plt.figure(figsize=(10, 5))
    
        z = wg_list[0].C0 * np.linspace(0, wg_list[0].L, wg_list[0].points)

        for i in range(len(wg_list)):
        
            k_data = wg_list[i].dist_prob[:, wg_number]
            plt.plot(z, k_data, label=f'$\\Delta\\beta / C_0  = {wg_list[i].B_slope}$')
        
        plt.grid()
        plt.legend(fontsize=17)
        plt.tick_params(axis='x', labelsize=17)
        plt.tick_params(axis='y', labelsize=17)
        plt.xlabel("$C_0Z$", fontsize=18)
        plt.ylabel("Prob_distribution", fontsize=18)

        if save:
            plt.savefig(rf'{self.output_dir}/{case}_Propagation_Line.pdf', dpi=600, bbox_inches = "tight")
        plt.show()


        
    def plot_disorder_corr(self, wg, disorder=False, frame=-1, case=None, corr_type="Q", output_dir="figures",save=False):
        z = wg.C0 * np.linspace(0, wg.L, wg.points)

        labels = ['(a)', '(b)', '(c)', '(d)']
        
        #Selecciona el tipo dematriz de correlacion
        if corr_type == "Q":
            corr_data = wg.Q_corr_av

        elif corr_type == "K":
            corr_data = wg.K_corr_av

        else:
            print("WARNING: invalid corr_type - The code will assume K correlation matrices")
            corr_data = wg.K_corr_av

        #Define la cantidad de graficas
        nQty = len(wg.kap_C)
        
        #Crea una fila de subplots
        fig, axes = plt.subplots(
            1,
            nQty,
            figsize=(5 * nQty, 5),
            sharex=False,
            sharey=True
        )
    
        #En caso de que solo exista un kappa
        if nQty == 1:
            axes = [axes]
    
        images = [None] * nQty


        # Configura el plot 
        for i in range(nQty):

            images[i] = axes[i].imshow(
                corr_data[i, frame],
                norm=colors.LogNorm(vmin=1e-4, vmax=1),
                cmap='YlGnBu',
                origin='lower',
                aspect='auto'
            )
    
            axes[i].set_title(rf'$\kappa_C={wg.kap_C[i]}$', fontsize=18)
            axes[i].set_xlabel('Waveguide r', fontsize=18)
            axes[i].tick_params(axis="both", labelsize=16)

            axes[i].text(
                0.01, 0.98,
                labels[i],
                transform=axes[i].transAxes,
                fontsize=18,
                fontweight='bold',
                va='top',
                ha='left',
                color='black'
            )
            
        #Titulo
        if case:
            title = fig.suptitle(
                rf'{case} case, {corr_type} correlation matrix, $C_0z={z[frame]:.2f}$',
                fontsize=18
            )
        else:
            title = fig.suptitle(
                rf'{corr_type} correlation matrix, $C_0z={z[frame]:.2f}$',
                fontsize=18
            )
        
        # Colorbar compartida
        axes[0].set_ylabel('Waveguide q', fontsize=18)
        fig.tight_layout(rect=[0, 0, 0.96, 1])
        cbar = fig.colorbar(images[0], ax=axes, location="right", fraction=0.02, pad=0.02)
        cbar.set_label(r'$\Gamma_{q,r}$', fontsize=18)
        cbar.ax.tick_params(labelsize=14)
        
        if save:
            fig.savefig(f"{self.output_dir}/corr_matrix/Matrix{corr_type}_{case}_L{z[frame]:.0f}.pdf", dpi=300, bbox_inches="tight")
        
        plt.show()

    
        
    def animate_disorder_corr(self, wg, corr_type="Q", case=None):
        z = wg.C0 * np.linspace(0, wg.L, wg.points)
        
        #Selecciona el tipo dematriz de correlacion
        if corr_type == "Q":
            corr_av_data = wg.Q_corr_av

        elif corr_type == "K":
            corr_av_data = wg.K_corr_av

        else:
            print("WARNING: invalid corr_type - The code will assume K correlation matrices")
            corr_av_data = wg.K_corr_av

        #Define la cantidad de graficas
        nQty = len(wg.kap_C)
        
        #Crea una fila de subplots
        fig, axes = plt.subplots(
            1,
            nQty,
            figsize=(5 * nQty, 5),
            sharex=False,
            sharey=False
        )
    
        #En caso de que solo exista un kappa
        if nQty == 1:
            axes = [axes]
    
        images = [None] * nQty


        # Configura un plot inicial
        for i in range(nQty):

            images[i] = axes[i].imshow(
                corr_av_data[i, 0],
                norm=colors.LogNorm(vmin=1e-4, vmax=1),
                cmap='YlGnBu',
                origin='lower',
                aspect='auto'
            )
    
            axes[i].set_title(rf'$\kappa_C={wg.kap_C[i]}$', fontsize=18)
            axes[i].set_xlabel('Waveguide r', fontsize=18)
            axes[i].tick_params(axis="both", labelsize=16)
            
        #Titulo inicial
        if case:
            title = fig.suptitle(
                rf'{case} case, {corr_type} correlation matrix, $C_0z={z[0]:.2f}$',
                fontsize=18
            )
        else:
            title = fig.suptitle(
                rf'{corr_type} correlation matrix, $C_0z={z[0]:.2f}$',
                fontsize=18
            )
        
        #Colorbar compartida
        axes[0].set_ylabel('Waveguide q', fontsize=18)
        fig.tight_layout(rect=[0, 0, 0.96, 1])
        cbar = fig.colorbar(images[0], ax=axes, location="right", fraction=0.02, pad=0.02)
        cbar.set_label(r'$\Gamma_{q,r}$', fontsize=18)
        cbar.ax.tick_params(labelsize=14)

        

        def update(frame):
            for i in range(nQty):
                images[i].set_data(corr_av_data[i, frame])
    
                # Actualiza el título
                if case:
                    title.set_text(
                        rf'{case} case, {corr_type} correlation matrix, $C_0z={z[frame]:.2f}$'
                    )
                else:
                    title.set_text(
                        rf'{corr_type} correlation matrix, $C_0z={z[frame]:.2f}$'
                    )
    
            return images

        # Genera la animación
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=wg.points,
            interval=50,
            blit=False
        )

        plt.close(fig)
        
        plt.rcParams['animation.embed_limit'] = 100
        display(HTML(ani.to_jshtml()))        

        
    
    def std_comparison(self, wg_lin, wg_nonlin, kappaC, case=None, save=False):
        """Compara std no lineal vs lineal para todos los niveles de desorden en kappaC."""
    
        z = wg_nonlin.C0 * np.linspace(0, wg_nonlin.L, wg_nonlin.points)
        labels = ['(a)', '(b)', '(c)']
    
        fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=False)
    
        for i in range(len(kappaC)):
            std_nl = wg_nonlin.std_av[i]
            std_l  = wg_lin.std_av[i]
    
            #G1: no lineal en funcion de lineal (una linea por kappa_C)
            axes[0].plot(std_l, std_nl, label=rf'${kappaC[i]}$')

            #G2: no lineal en funcion de z
            axes[1].plot(z, std_nl, label=rf'${kappaC[i]}$')
    
            #G3: lineal en funcion de z
            axes[2].plot(z, std_l, label=rf'${kappaC[i]}$')
        
        axes[0].set_xlabel(r'$\sigma_{Linear}$', fontsize=18)
        axes[0].set_ylabel(r'$\sigma_{Nonlinear}$', fontsize=18)
    
        axes[1].set_xlabel('$C_0 z$', fontsize=18)
        axes[1].set_ylabel(r'$\sigma_{Nonlinear}$', fontsize=18)
    
        axes[2].set_xlabel('$C_0 z$', fontsize=18)
        axes[2].set_ylabel(r'$\sigma_{Linear}$', fontsize=18)

        lims = [
            min(np.nanmin(wg_lin.std_av), np.nanmin(wg_nonlin.std_av)),
            max(np.nanmax(wg_lin.std_av), np.nanmax(wg_nonlin.std_av))
        ]
        axes[0].plot(lims, lims, color='gray', linestyle=':', linewidth=1)
    
        for j in range(len(axes)):
            axes[j].grid()
            axes[j].tick_params(axis='both', labelsize=17)
            axes[j].text(
                0.02, 0.98, labels[j],
                transform=axes[j].transAxes, fontsize=18, fontweight='bold',
                va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8)
            )
            
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            title=r"$\kappa_C$",
            title_fontsize=18,
            fontsize=16
        )

        if case:
            title_name = f"{case} case, Nonlinear vs Linear std comparison"
        else:
            title_name = f"Nonlinear vs Linear std comparison"
            
        title = fig.suptitle(title_name,fontsize=18)
    
        plt.tight_layout(w_pad=3)

        if save:
            fig.savefig(f"{self.output_dir}/std_lin_nonlin_comparison_{case}.pdf", dpi=300, bbox_inches="tight")
        plt.show()


    def plot_disorder_data(self, wg_list, data_type, case=None, save=False, ncols=3, xlim=None, ylim=None):
        
        labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
    
        # Cantidad de arreglos
        nQty = len(wg_list)
    
        # Cantidad de filas necesaria
        nrows = int(np.ceil(nQty / ncols))
    
        # Crea subplots en varias filas
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(6 * ncols, 6 * nrows),
            sharex=False,
            sharey=True
        )
    
        # Convierte axes a arreglo 1D
        axes = np.atleast_1d(axes).flatten()
    
        N_name = ""
    
        # Recorre cada arreglo
        for j in range(nQty):
    
            wg = wg_list[j]
    
            N_name += str(wg.N) + "_"
    
            z = wg.C0 * np.linspace(0, wg.L, wg.points)
    
            # Recorre los niveles de desorden
            for i in range(len(wg.kap_C)):
                if data_type == "schmidt":
                    k_data = wg.schmidt_av[i]
                    name = r"Schmidt parameter $K$"
                
                elif data_type == "std":
                    k_data = wg.std_av[i]
                    name = r"Standard deviation $\sigma$"
                
                elif data_type == "std_der":
                    k_data = wg.std_der_av[i]
                    name = r"Transport width $\gamma$"
                                    
                axes[j].plot(z, k_data, label=rf'$\kappa_C = {wg.kap_C[i]}$')
    
            # Título según número de guías
            axes[j].set_title(rf'$N={wg.N}$',fontsize=18)
    
            axes[j].grid()
    
            axes[j].tick_params(axis='both',labelsize=17)
    
            axes[j].set_xlabel(r"$C_0z$",fontsize=18)
    
            axes[j].text(
                0.02,
                0.97,
                labels[j],
                transform=axes[j].transAxes,
                fontsize=18,
                fontweight='bold',
                va='top',
                ha='left'
            )

            if xlim:
                axes[j].set_xlim(xlim)
            if ylim:
                axes[j].set_ylim(ylim)
                
    
        # Elimina subplots vacíos
        for j in range(nQty, len(axes)):
            fig.delaxes(axes[j])
    
        # Y-label en la primera columna de cada fila
        for row in range(nrows):
            idx = row * ncols
    
            if idx < nQty:
                axes[idx].set_ylabel(name,fontsize=18)
    
        # Leyenda común
        handles, legend_labels = axes[0].get_legend_handles_labels()
    
        fig.legend(
            handles,
            wg_list[0].kap_C,
            loc="center left",
            bbox_to_anchor=(0.9, 0.5),
            frameon=False,
            title=r"$\kappa_C$",
            title_fontsize=18,
            fontsize=16
        )
    
        # Título general
        if case:
            fig.suptitle(
                f'{case} case, {name} under coupling disorder',fontsize=18)
        else:
            fig.suptitle(f'{name} under coupling disorder',fontsize=18)
    
        fig.tight_layout(rect=[0, 0, 0.90, 0.96])
    
        if save:
            fig.savefig(f"{self.output_dir}/{data_type}/{data_type}_{case}_N{N_name}.pdf",dpi=300,bbox_inches="tight")
    
        plt.show()

    
    def plot_disorder_propagation(self, wg, zlim=None, case=None):
    
        #Define la cantidad de graficas
        nQty = len(wg.kap_C)
        N = wg.N

        fig, axes = plt.subplots(1, nQty, figsize=(5*nQty, 5), sharey=True)
        
        labels = [f"({chr(97 + i)})" for i in range(nQty)]
        
        for i in range(nQty):
            
            axes[i].text(
            0.02, 0.08,
            labels[i],
            transform=axes[i].transAxes,
            fontsize=18,
            fontweight='bold',
            va='top',
            ha='left',
            color='white'
            )
    
            propagation = axes[i].pcolor(wg.dist_prob_av[i], norm=colors.LogNorm(vmin=1e-4, vmax=1))
    
            #Configura la grafica en caso de un numero impar o par de wg
            if N % 2 == 0:
                xticks = np.array([0.5, round(N/3) - 0.5, round(2*N/3) - 0.5, N - 0.5])
                xlabels = np.array([1, round(N/3), round(2*N/3), N]).astype(int)
            else:
                xticks = np.array([0.5, round(N/4) + 0.5, N/2, round(3*N/4) - 0.5, N - 0.5])
                xlabels = (xticks - N/ 2).astype(int)

            if zlim is not None:
                frame_max = zlim / (wg.C0 * wg.L) * wg.points
                axes[i].set_ylim(0, frame_max)
                lim = zlim
            else:
                frame_max = wg.points
                lim = wg.C0 * wg.L
    
            #Ajusta los ejes de la grafica
            axes[i].set_xticks(xticks)
            axes[i].set_xticklabels(xlabels, fontsize=18)
            axes[i].set_yticks(np.linspace(0, frame_max, 6))
            axes[i].set_yticklabels(np.round(np.linspace(0, lim, 6), 2), fontsize=18)
            axes[i].set_xlabel('Waveguide number', fontsize=18)
        
            #Solo el primer subplot tiene ylabel
            if i == 0:
                axes[i].set_ylabel('$C_0 z$', fontsize=18)
                
            #Título por subplot
            axes[i].set_title(fr'$\kappa_C = {wg.kap_C[i]}$', fontsize=18)
        
        #Configura la barra lateral
        cbar_ax = fig.add_axes([0.90, 0.145, 0.02, 0.78])
        cbar = fig.colorbar(propagation, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=18) 
    
        if case:
            title = fig.suptitle(f"{case} case",fontsize=18)
            #plt.savefig(rf'Figures/nonlin_prop_{case}.png', dpi=600, bbox_inches = "tight")
            
        plt.tight_layout(rect=[0, 0, 0.9, 0.99])
        plt.show()



    
@njit(parallel=True)
def Qs_parallel(B_signal, eigenvals, eigenmodes, lambda_data, points, N, z, pump, g, alpha):
    
    B_pump = 2 * B_signal
    
    Q_nm = np.zeros((points, N, N), dtype=np.complex128)

    for p in prange(points):
        temp = np.zeros((N, N), dtype=np.complex128)

        for j in range(N):
            S_n = eigenmodes[:,j]

            P_nm = pump[j] * np.outer(S_n, S_n)

            phase = (lambda_data + B_pump[j]) * z[p] / 2
            arg = (lambda_data - B_pump[j]) * z[p] / (2 * np.pi)
            T_nm = np.exp(1j * phase) * np.sinc(arg)
            temp += P_nm * T_nm

        Q_nm[p] = 1j * z[p] * g * alpha * temp
        
    return Q_nm