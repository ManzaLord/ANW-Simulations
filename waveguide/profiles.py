import numpy as np

class B_profiles:
    @staticmethod
    def wannier_stark_beta(N, B_slope, B_s, b0=0):
        """Perfil de constantes de propagación efectiva (potencial de Wannier-Stark)"""
        return b0 + B_slope * B_s * np.linspace(0, 1, N)
        
    @staticmethod
    def uniform_beta(N, B_slope, B_s, b0=0):
        """Perfil de constantes de propagación efectiva homogéneo"""
        return b0 + B_s * np.ones(N)