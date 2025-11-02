from modelling_function_instance import run_bayes_shap_for_loop


#for i in [10, 11]:   
for i in [10,11]:
    run_bayes_shap_for_loop(
        i,
        input_csv_template="/home/ec2-user/ML-correlator/Tree_classifier_for_graphs/new_stuff/features/new_features/new2_merged/{i}loops_merged.csv",
        output_root="/home/ec2-user/ML-correlator/Tree_classifier_for_graphs/new_stuff/results/bayes_shap_result",
        n_calls=20,
        n_splits=5,
        n_workers=4,
        random_state=42,
        save_plot_threshold_curves=True,
        shap_sample_size = 10000
    )