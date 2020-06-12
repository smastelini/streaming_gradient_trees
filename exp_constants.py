DATASETS_NOM_COLS = {
    'abalone': [0],
    'ailerons': None,
    # 'airlines07_08': [1, 2, 3, 6, 9, 10, 12],
    'bike': [0, 1, 2, 3, 4, 5, 6, 7],
    'cal_housing': None,
    'elevators': None,
    'fried': None,
    'house_8L': None,
    'house_16H': None,
    'metro_interstate_traffic': [0, 5, 6],
    'mv_delve': [2, 6, 7],
    'pol': None,
    'wind': [1, 2],
    'winequality': None
}

METHODS_VARIANTS = {
    # 'htr_mean': 'HTR$_m$',
    # 'htr_perceptron': 'HTR$_p$',
    'sgt': 'SGT'
}

FORMATED_DATASETS_NAMES = {
    'abalone': 'Abalone',
    'ailerons': 'Ailerons',
    # 'airlines07_08': 'Airlines07-08',
    'bike': 'Bike',
    'cal_housing': 'CalHousing',
    'elevators': 'Elevators',
    'fried': 'Fried',
    'house_8L': 'House8L',
    'house_16H': 'House16H',
    'metro_interstate_traffic': 'MetroTraffic',
    'mv_delve': 'MVDelve',
    'pol': 'Pol',
    'wind': 'Wind',
    'winequality': 'WineQuality'
}

INPUT_PATH = '/home/mastelini/stream_benchmarks/datasets/str'
RAW_OUTPUT_PATH = '../../outputs/sgt/str/raw_results'
OUTPUT_PATH = '../../outputs/sgt/str'
N_REPETITIONS = 10
MAIN_SEED = 42
