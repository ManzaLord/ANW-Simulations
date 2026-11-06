import time
import pickle
import warnings
import numpy as np
from pathlib import Path
from scipy.linalg import eigh
from numba import njit, prange
from tqdm.auto import tqdm

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

        self.pump = None
        if pump_data is not None:
            self.set_pump(pump_data)

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
        pump = np.asarray(data, dtype=complex)
        norm = np.linalg.norm(pump)

        self.pump = pump / norm

    def save(self, name):

        folder = Path("waveguides data")
        folder.mkdir(parents=True, exist_ok=True)

        filename = folder / f"{name}.pkl"

        with open(filename, "wb") as f:
            pickle.dump(self, f)

        print(f"Data saved: {filename}")


    @staticmethod
    def load(name):
        filename = Path("waveguides data") / f"{name}.pkl"

        with open(filename, "rb") as f:
            return pickle.load(f)

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
        
        #Promedio de fotones por guia
        n = np.sum((1 + np.eye(K.shape[-1])) * np.abs(K) ** 2, axis=-1)
        
        #Promedio total de fotones
        photon = np.sum(n, axis=-1)

        #Distribucion normalizada
        dist_prob = n / np.sum(n, axis=-1, keepdims=True)
    
        norm_Q = np.sum(2 ** (np.eye(Q.shape[-1]) - 1) * np.abs(Q) ** 2, axis=(-1, -2), keepdims=True)
        Q_corr = np.abs(Q) ** 2 / norm_Q
    
        norm_K = np.sum(2 ** (np.eye(K.shape[-1]) - 1) * np.abs(K) ** 2, axis=(-1, -2), keepdims=True)
        K_corr = np.abs(K) ** 2 / norm_K
    
        prob_no_nans = np.where(np.isnan(dist_prob), 0, dist_prob)
        part_ratio = 1 / np.sum(prob_no_nans ** 2, axis=-1)
    
        #Calcula la desviacion estandar planar normalizada
        if self.wg_type == "Planar":
            M = np.arange(1, dist_prob.shape[-1] + 1)
            std_norm = np.sqrt(np.sum(M**2 * dist_prob, axis=-1) - np.sum(M * dist_prob, axis=-1)**2)

        else:
            # Guía de entrada
            j0 = np.argmax(self.pump)
        
            # Índices de las guías
            j = np.arange(self.N)
        
            # Distancia geodésica en el anillo
            delta = np.abs(j - j0)
            r = np.minimum(delta, self.N - delta)
        
            # Anchura de transporte
            std_norm = np.sqrt(np.sum(r**2 * dist_prob, axis=-1))
        
        std = std_norm * photon

        #Calcula las derivadas
        std_der      = np.gradient(np.log(std)     , np.log(self.C0 * self.z), edge_order=2)
        std_norm_der = np.gradient(np.log(std_norm), np.log(self.C0 * self.z), edge_order=2)
        photon_der   = np.gradient(np.log(photon)  , np.log(self.C0 * self.z), edge_order=2)
    
        result = dict(
                Q=Q,
                K=K,
                dist_prob=dist_prob,
                Q_corr=Q_corr,
                K_corr=K_corr,
                part_ratio=part_ratio,
                std=std,
                std_der=std_der,
                std_norm=std_norm,
                std_norm_der=std_norm_der,
                photon=photon,
                photon_der=photon_der
            )
    
        if calc_schmidt:
            #Calcula la matriz diagonal
            _, D, _ = np.linalg.svd(Q, full_matrices=False)
    
            #Normaliza los valores obtenidos
            lambda_vals = D**2 / np.sum(D**2, axis=-1, keepdims=True)
    
            #Genera el parametro k
            result['schmidt'] = 1 / np.sum(lambda_vals**2, axis=-1)
    
        return result
    

    def disorder(self, kap_B, kap_C, runsQty=200, linear=False, calc_schmidt=False, phase_matching=False):

        #Guarda los desordenes
        self.kap_B = kap_B
        self.kap_C = kap_C

        #Define el perfil de las constantes de acople
        f_0 = np.ones(self.N-1)  #Homogeneo

        def rand_kappa(kappa, N=1):
            """Genera N valores aleatorios uniformes entre -kappa y kappa."""
            return kappa * (2 * np.random.rand(N) - 1)

        #Lista donde se guardan los datos calculados

        #Desviacion estandar normalizada
        std_norm_der_av = np.zeros((len(kap_C), len(kap_B), self.points))
        std_norm_av     = np.zeros((len(kap_C), len(kap_B), self.points))
        
        #Desviacion estandar sin normalizar
        std_der_av      = np.zeros((len(kap_C), len(kap_B), self.points))
        std_av          = np.zeros((len(kap_C), len(kap_B), self.points))
        
        #Promedio de fotones
        photon_av       = np.zeros((len(kap_C), len(kap_B), self.points))
        photon_der_av   = np.zeros((len(kap_C), len(kap_B), self.points))

        #Matrices de correlacion
        Q_corr_av       = np.zeros((len(kap_C), len(kap_B), self.points, self.N, self.N))
        K_corr_av       = np.zeros((len(kap_C), len(kap_B), self.points, self.N, self.N))

        #Distribucion de probabilidad
        dist_prob_av    = np.zeros((len(kap_C), len(kap_B), self.points, self.N))

        if calc_schmidt:
            schmidt_av  = np.zeros((len(kap_C), len(kap_B), self.points))

        #Contadores de la duracion de la simulacion
        time_diag = 0
        time_sim = 0

        #Calcula la cantidad total de simulaciones
        total_runs = sum(
            1 if c == 0 and b == 0 else runsQty
            for c in kap_C for b in kap_B
        )

        #Inicializa la barra de progreso
        pbar = tqdm(total=total_runs, desc="Disorder", unit="run")

        for i in range(len(kap_C)):
            for k in range(len(kap_B)):

                pbar.set_postfix(kap_B=kap_B[k], kap_C=kap_C[i])

                for j in range(runsQty):
                    
                    t0 = time.time()
    
                    #Crea el perfil de constantes de propagacion con desorden
                    self.B_signal_dis = self.B_signal + rand_kappa(kap_B[k], self.N) * self.deltaB_dis
                    
                    if phase_matching:
                        ind = np.nonzero(self.pump)[0]
                        self.B_signal_dis[self.pump != 0] = self.B_signal_dis[ind[0]]

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
                    std_av[i, k]          += self.std_dis
                    std_der_av[i, k]      += self.std_der_dis
                    std_norm_av[i,k]      += self.std_norm_dis
                    std_norm_der_av[i, k] += self.std_norm_der_dis
                    photon_av[i, k]       += self.photon_dis
                    photon_der_av[i, k]   += self.photon_der_dis
                    Q_corr_av[i, k]       += self.Q_corr_dis
                    K_corr_av[i, k]       += self.K_corr_dis
                    dist_prob_av[i, k]    += self.dist_prob_dis
                    
                    if calc_schmidt:
                        schmidt_av[i, k]  += self.schmidt_dis
    
                    time_sim += time.time() - t0
                    pbar.update(1)
        
                    if kap_C[i] == 0 and kap_B[k] == 0:
                        break
                    
                if kap_C[i] == 0 and kap_B[k] == 0:
                    continue
                    
                else:

                    std_av[i, k]         = std_av[i, k]         / runsQty
                    std_der_av[i, k]     = std_der_av[i, k]     / runsQty
                    std_norm_av[i, k]    = std_norm_av[i, k]    / runsQty
                    std_norm_der_av[i,k] = std_norm_der_av[i,k] / runsQty
                    photon_av[i, k]      = photon_av[i, k]      / runsQty
                    photon_der_av[i, k]  = photon_der_av[i, k]  / runsQty
                    Q_corr_av[i, k]      = Q_corr_av[i, k]      / runsQty
                    K_corr_av[i, k]      = K_corr_av[i, k]      / runsQty
                    dist_prob_av[i, k]   = dist_prob_av[i, k]   / runsQty
                    if calc_schmidt:
                        schmidt_av[i, k] = schmidt_av[i, k] / runsQty

        pbar.close()

        #Muestra los tiempo de duracion de la simulacion
        print('Diagonalization time =', time_diag)
        print('Simulation time =', time_sim)
        print('Total time =', time_diag + time_sim)

        #Guarda los promedios obtenidos
        self.std_av          = std_av.squeeze()
        self.std_der_av      = std_der_av.squeeze()
        self.std_norm_av     = std_norm_av.squeeze()
        self.std_norm_der_av = std_norm_der_av.squeeze()
        self.photon_av       = photon_av.squeeze()
        self.photon_der_av   = photon_der_av.squeeze()
        self.Q_corr_av       = Q_corr_av.squeeze()
        self.K_corr_av       = K_corr_av.squeeze()
        self.dist_prob_av    = dist_prob_av.squeeze()
        if calc_schmidt:
            self.schmidt_av  = schmidt_av.squeeze()


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
