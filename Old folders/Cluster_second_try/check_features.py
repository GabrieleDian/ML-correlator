"""
Utility script to check and manage pre-computed features.
"""

import numpy as np
from pathlib import Path
import argparse
from load_features import get_available_features, check_feature_consistency


def check_all_loops(data_dir='Graph_Edge_Data'):
    """Check all available loops and their features."""
    print("Checking all available loops and features...")
    print("=" * 60)
    
    # Find all feature directories
    feature_dirs = list(Path(data_dir).glob('features_loop_*'))
    
    if not feature_dirs:
        print("No pre-computed features found!")
        return
    
    for feature_dir in sorted(feature_dirs):
        loop = int(feature_dir.name.split('_')[-1])
        features = get_available_features(loop, data_dir)
        
        print(f"\nLoop {loop}:")
        if features:
            print(f"  Available features: {', '.join(features)}")
            
            # Check first feature for stats
            first_feat = np.load(feature_dir / f"{features[0]}.npy")
            print(f"  Number of graphs: {first_feat.shape[0]}")
            print(f"  Max nodes: {first_feat.shape[1]}")
        else:
            print("  No features computed yet")


def feature_statistics(loop_order, feature_name, data_dir='Graph_Edge_Data'):
    """Show statistics for a specific feature."""
    feature_path = Path(data_dir) / f'features_loop_{loop_order}' / f'{feature_name}.npy'
    
    if not feature_path.exists():
        print(f"Feature {feature_name} not found for loop {loop_order}")
        return
    
    # Load feature
    features = np.load(feature_path)
    
    print(f"\nStatistics for {feature_name} (loop {loop_order}):")
    print(f"  Shape: {features.shape}")
    print(f"  Min: {np.min(features):.4f}")
    print(f"  Max: {np.max(features):.4f}")
    print(f"  Mean: {np.mean(features):.4f}")
    print(f"  Std: {np.std(features):.4f}")
    print(f"  % zeros (padding): {100 * np.mean(features == 0):.1f}%")
    
    # Show example
    print(f"\nFirst graph {feature_name} values (first 20 nodes):")
    print(features[0][:20])


def delete_feature(loop_order, feature_name, data_dir='Graph_Edge_Data'):
    """Delete a specific feature file."""
    feature_path = Path(data_dir) / f'features_loop_{loop_order}' / f'{feature_name}.npy'
    
    if feature_path.exists():
        response = input(f"Delete {feature_path}? (y/n): ")
        if response.lower() == 'y':
            feature_path.unlink()
            print(f"Deleted {feature_path}")
    else:
        print(f"Feature {feature_name} not found for loop {loop_order}")


def estimate_memory_usage(loop_order, data_dir='Graph_Edge_Data'):
    """Estimate memory usage for features."""
    features_dir = Path(data_dir) / f'features_loop_{loop_order}'
    
    if not features_dir.exists():
        print(f"No features found for loop {loop_order}")
        return
    
    total_size = 0
    print(f"\nMemory usage for loop {loop_order}:")
    
    for feature_file in features_dir.glob('*.npy'):
        size_mb = feature_file.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  {feature_file.stem}: {size_mb:.1f} MB")
    
    print(f"\nTotal: {total_size:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description='Check and manage pre-computed features')
    parser.add_argument('--check-all', action='store_true', 
                       help='Check all loops and features')
    parser.add_argument('--loop', type=int, help='Loop order')
    parser.add_argument('--feature', type=str, help='Feature name')
    parser.add_argument('--stats', action='store_true', 
                       help='Show statistics for a feature')
    parser.add_argument('--delete', action='store_true', 
                       help='Delete a feature')
    parser.add_argument('--memory', action='store_true',
                       help='Show memory usage')
    parser.add_argument('--consistency', action='store_true',
                       help='Check feature consistency')
    
    args = parser.parse_args()
    
    if args.check_all:
        check_all_loops()
    elif args.stats and args.loop and args.feature:
        feature_statistics(args.loop, args.feature)
    elif args.delete and args.loop and args.feature:
        delete_feature(args.loop, args.feature)
    elif args.memory and args.loop:
        estimate_memory_usage(args.loop)
    elif args.consistency and args.loop:
        check_feature_consistency(args.loop)
    else:
        # Default: show available features for specified loop
        if args.loop:
            features = get_available_features(args.loop)
            print(f"Available features for loop {args.loop}: {features}")
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
