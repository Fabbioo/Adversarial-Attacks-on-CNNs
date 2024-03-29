from .hub_imports import warnings, torch, plt, os

warnings.filterwarnings('ignore')

device: str = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

x_img_resize: int = 224
y_img_resize: int = 224
img_resize: tuple = (x_img_resize, y_img_resize)

x_figure_size: int = 16
y_figure_size: int = 8
fig_size: tuple = (x_figure_size, y_figure_size)

parametri_grafici: dict = {
    'figure.figsize': fig_size, # Dimensione della figura.
    'figure.autolayout': True,  # Regolazione automatica delle dimensioni della figura.
    'figure.titlesize': 20,     # Dimensione del titolo associato ad ogni figura (plt.suptitle()).
    'axes.titlesize': 20,       # Dimensione del titolo associato ad ogni grafico all'interno di una figura (plt.title()).
    'axes.labelsize': 20,       # Dimensione delle etichette sia sull'asse x sia sull'asse y.
    'xtick.labelsize': 15,      # Dimensione dei riferimenti sull'asse x.
    'ytick.labelsize': 15,      # Dimensione dei riferimenti sull'asse y.
    'legend.fontsize': 20,      # Dimensione dei caratteri della legenda.
}
plt.rcParams.update(parametri_grafici)

working_directory_path: str = os.getcwd()
images_path: str = working_directory_path + '/images'
results_path: str = working_directory_path + '/res'

epsilons: list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
alphas: list = [elem/100 for elem in epsilons]
iters: list = [5, 10, 25, 50, 75, 100]