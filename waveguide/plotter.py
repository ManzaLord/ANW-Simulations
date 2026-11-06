from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from IPython.display import HTML, display
from cycler import cycler

class plotter():

    def __init__(self, output_dir="figures"):

        #Directorio principal del proyecto
        root = Path(__file__).resolve().parent.parent

        #Directorios de salida
        output_dir = Path(output_dir)
        self.output_dir = output_dir if output_dir.is_absolute() else root / output_dir
        self.animation_dir = root / "animations"

        #Crea las carpetas si no existen
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.animation_dir.mkdir(parents=True, exist_ok=True)

        self.colors = [
            "#7B00FF",  # Morado
            "#75E300",  # Verde
            "#FF0084",  # Rosado
            "#E5C07B",  # Amarillo
            "#C678DD",  # Morado
            "#56B6C2",  # Cyan
            ]

        plt.rcParams["axes.prop_cycle"] = cycler(color=self.colors)
    
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
            plt.savefig(rf'{self.output_dir}/eigenmodes_{case}.png', dpi=600, bbox_inches = "tight")
        plt.show()

    
    
    def plot_propagation(self, wg_list, iter_item=None, iter_name=None, case=None, save=False):
        
        if not isinstance(wg_list, (list, tuple)):
            wg_list = [wg_list]

        fig, axes = plt.subplots(1, len(wg_list), figsize=(1+5*len(wg_list), 5), sharey=True)
        axes = np.atleast_1d(axes)
    
        #Define la cantidad de graficas
        nQty = len(wg_list)
        
        labels = [f"({chr(97 + i)})" for i in range(nQty)]
        
        for i in range(nQty):
            N = wg_list[i].N
            
            if len(labels) > 1:
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
    
            # Configura los ticks de las guías
            #Guía central
            center = (N - 1) // 2

            #Máximo índice simétrico
            ind_max = min(center, N - 1 - center)

            if ind_max <= 4:
                tick_names = np.arange(N) - center

            else:
                max_space = ind_max / 2

                #Busca un intervalo adecuado
                scale = 10 ** np.floor(np.log10(max_space))
                steps = np.array([1, 2, 4, 5, 10]) * scale
                step = int(steps[steps <= max_space].max())

                tick_names = np.arange(-2, 3) * step

            #Posiciones reales en la matriz
            tick_vals = tick_names + center + 0.5

            axes[i].set_xticks(tick_vals)
            axes[i].set_xticklabels(tick_names, fontsize=18)

            axes[i].set_yticks(np.linspace(0, wg_list[i].points, 6))
            axes[i].set_yticklabels(np.round(wg_list[i].C0 * np.linspace(0, wg_list[i].L, 6), 2).astype(int), fontsize=18)
            axes[i].set_xlabel('Waveguide number', fontsize=18)
        
            #Solo el primer subplot tiene ylabel
            if i == 0:
                axes[i].set_ylabel('$C_0 z$', fontsize=18)
                
            #Título por subplot
            if iter_name == None:
                pass
            else:
                axes[i].set_title(f'${iter_name}  = {iter_item[i]}$', fontsize=18)
        
        #Configura la barra lateral
        plt.tight_layout(rect=[0, 0, 0.9, 0.99])
        pos = axes[-1].get_position()
        cbar_ax = fig.add_axes([0.90, pos.y0, 0.2 / fig.get_figwidth(), pos.height])
        cbar = fig.colorbar(propagation, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=18) 
    
        if case:
            if save:
                plt.savefig(rf'{self.output_dir}/{case}_propagation.pdf', dpi=600, bbox_inches = "tight")
                plt.savefig(rf'{self.output_dir}/{case}_propagation.png', dpi=600, bbox_inches = "tight")

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
                plt.savefig(rf'{self.output_dir}/{case}_{corr_type}corr.png', dpi=600, bbox_inches = "tight")
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

    def plot_disorder_corr(self, wg, iter_item, iter_name, z_val=None, case=None, corr_type="Q",save=False, limits=None):
        z = wg.C0 * np.linspace(0, wg.L, wg.points)

        labels = ['(a)', '(b)', '(c)', '(d)']

        #Selecciona el frame mas cercano a z_value
        if z_val is None:
            frame = -1
        else:
            frame = np.argmin(np.abs(z - z_val))
        
        #Selecciona el tipo dematriz de correlacion
        if corr_type == "Q":
            corr_data = wg.Q_corr_av

        elif corr_type == "K":
            corr_data = wg.K_corr_av

        else:
            print("WARNING: invalid corr_type - The code will assume K correlation matrices")
            corr_data = wg.K_corr_av

        #Define la cantidad de graficas
        nQty = len(iter_item)
        
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
            if limits:

                images[i] = axes[i].imshow(
                    corr_data[i, frame, limits[0]:limits[1], limits[0]:limits[1]],
                    norm=colors.LogNorm(vmin=1e-4, vmax=1),
                    cmap='YlGnBu',
                    origin='lower',
                    aspect='auto'
                )

            else:
                images[i] = axes[i].imshow(
                    corr_data[i, frame],
                    norm=colors.LogNorm(vmin=1e-4, vmax=1),
                    cmap='YlGnBu',
                    origin='lower',
                    aspect='auto'
                )

            #Configura los ticks de las guías
            #Guia central
            center = (wg.N - 1) // 2

            #Ajusta el centro según el recorte
            if limits is not None:
                N_plot = limits[1] - limits[0]
                center = center - limits[0]
            else:
                N_plot = wg.N

            #Máximo índice simétrico
            ind_max = min(center, N_plot - 1 - center)

            if ind_max <= 4:
                tick_names = np.arange(N_plot) - center

            else:
                max_space = ind_max / 2

                #Busca un intervalo adecuado
                scale = 10 ** np.floor(np.log10(max_space))
                steps = np.array([1, 2, 4, 5, 10]) * scale
                step = int(steps[steps <= max_space].max())

                tick_names = np.arange(-2, 3) * step

            #Posiciones de los ticks
            tick_vals = tick_names + center

            axes[i].set_xticks(tick_vals)
            axes[i].set_xticklabels(tick_names, fontsize=18)

            axes[i].set_yticks(tick_vals)
            axes[i].set_yticklabels(tick_names, fontsize=18)
    
            axes[i].set_title(rf'${iter_name}={iter_item[i]}$', fontsize=18)
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
        #Colorbar compartida
        axes[0].set_ylabel('Waveguide q', fontsize=18)
        fig.tight_layout(rect=[0, 0, 0.96, 1])
        cbar = fig.colorbar(images[0], ax=axes, location="right", fraction=0.02, pad=0.02)
        cbar.set_label(r'$\Gamma_{q,r}$', fontsize=18)
        cbar.ax.tick_params(labelsize=14)
        
        if save:
            if case:
                filename = f"{case}_disorder_corr_{corr_type}_N{wg.N}_z{z[frame]:.2f}.pdf"
            else:
                filename = f"disorder_corr_{corr_type}_N{wg.N}_z{z[frame]:.2f}.pdf"
            
            fig.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")

        plt.show()

    
    def animate_disorder_corr(self, wg, corr_type="Q", case=None, save=False):
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
        if save:
            filename = f"disorder_corr_{corr_type}_N{wg.N}.gif"
            ani.save(self.animation_dir / filename, writer="pillow", fps=20)

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
    
        if save:
            fig.tight_layout(rect=[0, 0, 0.90, 0.96])

            N_name = "_".join(str(wg.N) for wg in wg_list)
            filename = f"disorder_{data_type}_N{N_name}.pdf"
            fig.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")

        else:
            if case:
                fig.suptitle(f'{case} case, {name} under coupling disorder',fontsize=18)
            else:
                fig.suptitle(f'{name} under coupling disorder',fontsize=18)

            fig.tight_layout(rect=[0, 0, 0.90, 0.96])

        plt.show()

    
    def plot_disorder_propagation(self, wg, iter_item, iter_name, zlim=None, case=None, save=False):
    
        #Define la cantidad de graficas
        nQty = len(iter_item)
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
            axes[i].set_title(fr'${iter_name} = {iter_item[i]}$', fontsize=18)
        
        #Configura la barra lateral
        cbar_ax = fig.add_axes([0.90, 0.145, 0.02, 0.78])
        cbar = fig.colorbar(propagation, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=18) 
    
        
        if save:
            plt.tight_layout(rect=[0, 0, 0.9, 0.99])
            filename = f"disorder_propagation_N{N}.pdf"
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
        else:
            if case:
                title = fig.suptitle(f"{case} case",fontsize=18)
            plt.tight_layout(rect=[0, 0, 0.9, 0.99])
        plt.show()
