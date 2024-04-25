from .hub_imports import warnings, torch, plt, os

warnings.filterwarnings("ignore")

device: str = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

x_img_resize: int = 224
y_img_resize: int = 224
img_resize: tuple[int] = (x_img_resize, y_img_resize)

x_figure_plot_size: int = 16
y_figure_plot_size: int = 8
fig_size: tuple[int] = (x_figure_plot_size, y_figure_plot_size)

parametri_grafici: dict[tuple[int] | bool | int] = {
    "figure.figsize": fig_size,         # Dimensione della figura.
    "figure.autolayout": True,          # Regolazione automatica delle dimensioni della figura.
    "figure.titlesize": 20,             # Dimensione del titolo associato ad ogni figura (plt.suptitle()).
    "axes.titlesize": 20,               # Dimensione del titolo associato ad ogni grafico all'interno di una figura (plt.title()).
    "axes.labelsize": 20,               # Dimensione delle etichette sia sull'asse x sia sull'asse y.
    "xtick.labelsize": 15,              # Dimensione dei riferimenti sull'asse x.
    "ytick.labelsize": 15,              # Dimensione dei riferimenti sull'asse y.
    "legend.fontsize": 20,              # Dimensione dei caratteri della legenda.
    "font.family": "times new roman",   # Font utilizzata per i testi.
}
plt.rcParams.update(parametri_grafici)

working_directory_path: str = os.getcwd()
images_path: str = working_directory_path + "/images"
results_path: str = working_directory_path + "/res"
perturbed_images_path: str = working_directory_path + "/perturbed_images"