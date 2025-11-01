
from modelling_function import run_bayes_shap_for_loop


for i in [6,7]:   
    run_bayes_shap_for_loop(
        6,
        input_csv_template="/Users/rezadoobary/Documents/MLCORRELATORS/ML-correlator/Tree_classifier_for_graphs/new_stuff/features/merged/new_merged/{i}loops_merged.csv",
        output_root="/Users/rezadoobary/Documents/MLCORRELATORS/ML-correlator/Tree_classifier_for_graphs/new_stuff/results/bayes_shap_result",
        n_calls=20,
        n_splits=5,
        n_workers=8,
        random_state=42,
        save_plot_threshold_curves=True,
        shap_sample_size = 10000
    )