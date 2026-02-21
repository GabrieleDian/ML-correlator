from modelling_function_12_ts_lp1_reweighting import BayesConfig, setup_logging, run_bayes_on_merged_parquet
import json
from datetime import datetime
from pathlib import Path

# Define all runs
runs = [
    # Run 1: Train on [5,6,7], test on loop 8 with validation_step=[1]
    {
        "train_loops": [5, 6, 7],
        "test_loop": 8,
        "validation_step": [1],
        "label": "train_567_test8_vstep1"
    },
    # Run 2: Train on [5,6,7], test on loop 9 with validation_step=[1]
    {
        "train_loops": [5, 6, 7],
        "test_loop": 9,
        "validation_step": [1],
        "label": "train_567_test9_vstep1"
    },
    # Run 3: Train on [5,6,7,8], test on loop 9 with validation_step=[1]
    {
        "train_loops": [5, 6, 7, 8],
        "test_loop": 9,
        "validation_step": [1],
        "label": "train_5678_test9_vstep1"
    },
    # Run 4: Train on [5,6,7,8], test on loop 9 with validation_step=[2]
    {
        "train_loops": [5, 6, 7, 8],
        "test_loop": 9,
        "validation_step": [1,2],
        "label": "train_5678_test9_vstep12"
    },
    # Run 5: Train on [5,6,7,8], test on loop 10 with validation_step=[1]
    {
        "train_loops": [5, 6, 7, 8],
        "test_loop": 10,
        "validation_step": [1],
        "label": "train_5678_test10_vstep1"
    },
    # Run 6: Train on [5,6,7,8], test on loop 10 with validation_step=[1,2]
    {
        "train_loops": [5, 6, 7, 8],
        "test_loop": 10,
        "validation_step": [1, 2],
        "label": "train_5678_test10_vstep12"
    },
    # Run 7: Train on [5,6,7,8,9], test on loop 10 with validation_step=[1]
    {
        "train_loops": [5, 6, 7, 8, 9],
        "test_loop": 10,
        "validation_step": [1],
        "label": "train_56789_test10_vstep1"
    },
    # Run 8: Train on [5,6,7,8,9], test on loop 10 with validation_step=[1,2]
    {
        "train_loops": [5, 6, 7, 8, 9],
        "test_loop": 10,
        "validation_step": [1, 2],
        "label": "train_56789_test10_vstep12"
    },
    # Run 9: Train on [5,6,7,8,9], test on loop 11 with validation_step=[1]
    {
        "train_loops": [5, 6, 7, 8, 9],
        "test_loop": 11,
        "validation_step": [1],
        "label": "train_56789_test11_vstep1"
    },
    # Run 10: Train on [5,6,7,8,9], test on loop 11 with validation_step=[1,2]
    {
        "train_loops": [5, 6, 7, 8, 9],
        "test_loop": 11,
        "validation_step": [1, 2],
        "label": "train_56789_test11_vstep12"
    },
    # Run 11: Train on [5,6,7,8,9,10], test on loop 11 with validation_step=[1]
    {
        "train_loops": [5, 6, 7, 8, 9, 10],
        "test_loop": 11,
        "validation_step": [1],
        "label": "train_5678910_test11_vstep1"
    },
    # Run 12: Train on [5,6,7,8,9,10], test on loop 11 with validation_step=[1,2]
    {
        "train_loops": [5, 6, 7, 8, 9, 10],
        "test_loop": 11,
        "validation_step": [1, 2],
        "label": "train_5678910_test11_vstep12"
    },
    # Run 13: Train on [5,6,7,8,9,10,11] without test, validation_step=[1]
    {
        "train_loops": [5, 6, 7, 8, 9, 10, 11],
        "test_loop": None,
        "validation_step": [1],
        "label": "train_567891011_notest_vstep1"
    },
    # Run 14: Train on [5,6,7,8,9,10,11] without test, validation_step=[1,2]
    {
        "train_loops": [5, 6, 7, 8, 9, 10, 11],
        "test_loop": None,
        "validation_step": [1, 2],
        "label": "train_567891011_notest_vstep12"
    },
]

def run_experiment(run_config, run_number, total_runs):
    """Run a single experiment with the given configuration."""
    train_loops = run_config["train_loops"]
    test_loop = run_config["test_loop"]
    validation_step = run_config["validation_step"]
    label = run_config["label"]
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create descriptive run name
    train_str = "".join(map(str, train_loops))
    test_str = f"_test{test_loop}" if test_loop is not None else "_notest"
    vstep_str = "".join(map(str, validation_step))
    run_name = f"{timestamp}_{label}"
    
    print("\n" + "="*80)
    print(f"RUN {run_number}/{total_runs}: {run_name}")
    print("="*80)
    print(f"Configuration:")
    print(f"  Training loops: {train_loops}")
    print(f"  Test loop: {test_loop}")
    print(f"  Validation step: {validation_step}")
    print(f"  Label: {label}")
    print("="*80 + "\n")
    
    # Create config with custom results directory that includes the label
    base_results_dir = "/home/ec2-user/ML-correlator/Tree_classifier_for_graphs/new_stuff/bayes_results/bayes_results_merged"
    results_dir = Path(base_results_dir) / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    config = BayesConfig(
        n_calls=30,
        random_state=42,
        results_dir=str(results_dir),
        save_plot=True,
        threshold_metric="f1",
        device="cpu",
        use_oof_memmap=True,
        shap_sample_size=10000,
        validation_step=validation_step,
        test_loop=test_loop
    )
    
    # Setup logging with descriptive prefix
    log_prefix = f"run{run_number:02d}_{label}"
    _lf = setup_logging(str(results_dir), log_prefix=log_prefix)
    print(f"Log file: {_lf}")
    print(f"Results directory: {results_dir}")
    print(f"Using training loops: {train_loops}")
    if test_loop is not None:
        print(f"Test loop: {test_loop}")
    print(f"Validation step: {validation_step}\n")
    
    try:
        out = run_bayes_on_merged_parquet(
            dataset_dir="/home/ec2-user/ML-correlator/Tree_classifier_for_graphs/new_stuff/features/merged/dataset/dataset/dataset",
            loops=train_loops,
            log_root=str(results_dir),
            cfg=config
        )
        
        print(f"\n=== Run {run_number} Results ===")
        print(f"Run name: {run_name}")
        print(f"Best CV AUC: {out['best_cv_auc']:.4f}")
        print(f"CV AUC Mean: {out['cv_auc_mean']:.4f} ± {out['cv_auc_std']:.4f}")
        print(f"Decision threshold: {out['decision_threshold']:.6f}")
        if out.get('test_auc') is not None:
            print(f"Test AUC: {out['test_auc']:.4f}")
            print(f"Test scores saved to: {out.get('test_scores_path', 'N/A')}")
        print(f"Artifacts directory: {out['artifacts_dir']}")
        print("="*80 + "\n")
        
        # Save run summary
        run_summary = {
            "run_number": run_number,
            "run_name": run_name,
            "timestamp": timestamp,
            "label": label,
            "train_loops": train_loops,
            "test_loop": test_loop,
            "validation_step": validation_step,
            "best_cv_auc": out['best_cv_auc'],
            "cv_auc_mean": out['cv_auc_mean'],
            "cv_auc_std": out['cv_auc_std'],
            "test_auc": out.get('test_auc'),
            "artifacts_dir": out['artifacts_dir']
        }
        
        summary_path = results_dir / "run_summary.json"
        with open(summary_path, "w") as f:
            json.dump(run_summary, f, indent=2)
        print(f"Run summary saved to: {summary_path}\n")
        
        return run_summary
        
    except Exception as e:
        print(f"\nERROR in run {run_number} ({run_name}): {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("\n" + "="*80)
    print("STARTING EXPERIMENT RUNNER")
    print("="*80)
    print(f"Total runs to execute: {len(runs)}")
    print("="*80 + "\n")
    
    all_summaries = []
    start_time = datetime.now()
    
    for i, run_config in enumerate(runs, 1):
        summary = run_experiment(run_config, i, len(runs))
        if summary:
            all_summaries.append(summary)
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds() / 60  # minutes
    
    # Save master summary
    master_summary = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_minutes": elapsed,
        "total_runs": len(runs),
        "successful_runs": len(all_summaries),
        "runs": all_summaries
    }
    
    master_summary_path = Path("/home/ec2-user/ML-correlator/Tree_classifier_for_graphs/new_stuff/bayes_results/bayes_results_merged") / f"master_summary_{start_time.strftime('%Y%m%d-%H%M%S')}.json"
    master_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(master_summary_path, "w") as f:
        json.dump(master_summary, f, indent=2)
    
    print("\n" + "="*80)
    print("ALL RUNS COMPLETED")
    print("="*80)
    print(f"Total runs: {len(runs)}")
    print(f"Successful runs: {len(all_summaries)}")
    print(f"Failed runs: {len(runs) - len(all_summaries)}")
    print(f"Total elapsed time: {elapsed:.1f} minutes")
    print(f"Master summary saved to: {master_summary_path}")
    print("="*80 + "\n")
