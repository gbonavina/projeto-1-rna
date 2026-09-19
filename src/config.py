from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataset_projeto1.csv"
MODELS_DIR = PROJECT_ROOT / "models"
IMGS_DIR = PROJECT_ROOT / "imgs"
REPORT_FIGURES_DIR = (
    PROJECT_ROOT / "report" / "redes_neurais_template_projetos" / "figures"
)

RANDOM_STATE = 123
TEST_SIZE = 0.8
VAL_FRACTION_OF_DEV = 0.5

NEURONS = [1, 100, 100, 100, 1]
NUM_LAYERS = 4

NUM_EPOCHS = 10_000
LEARNING_RATE = 1e-2
BATCH_SIZE = 8

DROPOUT = 0.05
L2_WEIGHT_DECAY = 1e-3
L1_LAMBDA = 1e-4
MOMENTUM = 0.9
