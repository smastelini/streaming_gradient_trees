import os
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.utils import check_random_state
from skmultiflow.utils.file_scripts import clean_header

from utils import FileStreamPlus
from streaming_gradient_tree import StreamingGradientTree
from exp_constants import DATASETS_NOM_COLS, INPUT_PATH, RAW_OUTPUT_PATH, OUTPUT_PATH, \
    MAIN_SEED, N_REPETITIONS


if not os.path.exists(RAW_OUTPUT_PATH):
    os.makedirs(RAW_OUTPUT_PATH)

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

random_state = check_random_state(MAIN_SEED)
for data_name, nom_cols in DATASETS_NOM_COLS.items():
    data_path = '{}/{}.csv'.format(INPUT_PATH, data_name)
    random_seeds = random_state.randint(0, 4294967295, size=N_REPETITIONS, dtype='u8')
    for n, seed in enumerate(random_seeds):
        stream = FileStreamPlus(data_path, n_targets=1, mode='future', random_state=seed,
                                shuffle_data=True)

        sgt = StreamingGradientTree(mode='future')

        raw_out_file = '{}/results_sgt_{}_rep{:02d}.csv'.format(RAW_OUTPUT_PATH, data_name, n)
        out_file = '{}/results_sgt_{}_rep{:02d}.csv'.format(OUTPUT_PATH, data_name, n)
        evaluator = EvaluatePrequential(max_samples=float('Inf'), pretrain_size=200, batch_size=1,
                                        metrics=['mean_absolute_error', 'mean_square_error',
                                                 'running_time', 'model_size'],
                                        output_file=raw_out_file)

        evaluator.evaluate(stream=stream, model=sgt, model_names=['SGT'])
        clean_header(raw_out_file, out_file)
