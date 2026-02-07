import os
import gc
import random
import argparse
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve

from profiler_utils import SOURCE_DICT, MODEL_ZOO, data_generation, load_features


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Profiler: AI-Generated Text Origin Detection')
    
    # Basic settings
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    parser.add_argument('--task', type=str, required=True,
                       choices=['Arxiv', 'Code', 'Yelp', 'Essay', 'Creative', 'GCJ'],
                       help='Task/domain for detection')
    
    # Model settings
    parser.add_argument('--base_model', type=str, default='all',
                       help='Detection model(s) to use: all, llama2-7b, llama2-13b, llama3-8b, gemma-2b, gemma-7b, mistral-7b, or comma-separated list (e.g., "llama3-8b,gemma-7b")')
    parser.add_argument('--sample_clip', type=int, default=4000,
                       help='Maximum sample length (characters)')
    parser.add_argument('--context_window', type=int, default=6,
                       help='Full context window size (will be divided by 2 for positions on each side). Common values: 2, 4, 6, 8. Should be even.')
    
    # Dataset arguments
    parser.add_argument('--train_dataset', type=str, required=True,
                       help='Training dataset: {normal or paraphrased}_{task}')
    parser.add_argument('--test_dataset', type=str, required=True,
                       help='Testing dataset: {normal or paraphrased}_{task}')
    
    # Feature generation
    parser.add_argument('--data_generation', type=int, default=0,
                       help='1 to generate features, 0 to skip')

    return parser.parse_args()


def parse_dataset_arg(ds):
    """
    Parse dataset string with expected format:
      {normal or paraphrased}_{task}
    For example: "normal_Arxiv" or "paraphrased_Yelp"
    Returns a tuple: (dataset_type, task)
    """
    parts = ds.split('_')
    if len(parts) != 2 or parts[0] not in ['normal', 'paraphrased']:
        raise ValueError("Dataset must be in format {normal or paraphrased}_{task}")
    return parts[0], parts[1]


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_tree_parameters(tree):
    """Count parameters in a decision tree (for logging)"""
    tree_ = tree.tree_
    num_nodes = tree_.node_count
    num_leaf_nodes = np.sum(tree_.children_left == -1)
    num_split_nodes = num_nodes - num_leaf_nodes
    
    split_node_params = 2  # feature and threshold
    
    if hasattr(tree_, 'value'):
        leaf_node_params = tree_.value.shape[1] * tree_.value.shape[2]
    else:
        leaf_node_params = 1
    
    total_params = (num_split_nodes * split_node_params) + (num_leaf_nodes * leaf_node_params)
    return total_params


def main():
    args = parse_args()
    set_seed(args.seed)

    # Parse base_model argument to get list of models to use
    if args.base_model == 'all':
        base_models = list(MODEL_ZOO.keys())
    else:
        # Support comma-separated list of models
        base_models = [m.strip() for m in args.base_model.split(',')]
        # Validate all models are in MODEL_ZOO
        for model in base_models:
            if model not in MODEL_ZOO:
                raise ValueError(f"Unknown model: {model}. Available models: {list(MODEL_ZOO.keys())}")

    # Print configuration
    print('='*60)
    print('Detection Method: Profiler')
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print(f'Using detection models: {base_models}')
    print('='*60)

    # Parse dataset arguments
    train_type, train_task = parse_dataset_arg(args.train_dataset)
    test_type, test_task = parse_dataset_arg(args.test_dataset)

    # Validate task consistency
    if train_task != args.task or test_task != args.task:
        raise ValueError(f"Task mismatch: args.task={args.task}, train_task={train_task}, test_task={test_task}")

    # Create output directory structure
    # Include base_model info in path for ablation studies
    model_suffix = args.base_model.replace(',', '+') if args.base_model != 'all' else 'all'
    base_out_dir = os.path.join('./results',
        f"{args.train_dataset}_vs_{args.test_dataset}_model_{model_suffix}_context_window_{args.context_window}")
    os.makedirs(base_out_dir, exist_ok=True)
    
    # Feature Generation Phase
    if args.data_generation:
        print("Starting feature generation...")
        for base_model in base_models:
            print(f"Generating features using {base_model}...")

            # Generate training features
            success = data_generation(args, base_model, dataset_type=train_type)
            if not success:
                print(f"Error in training feature generation for {base_model}")
                return

            # Generate test features if different from training
            if args.train_dataset != args.test_dataset:
                success = data_generation(args, base_model, dataset_type=test_type)
                if not success:
                    print(f"Error in test feature generation for {base_model}")
                    return

        print("Feature generation complete.")
        gc.collect()
        torch.cuda.empty_cache()

    # Load training features for all sources
    print("Loading training features...")
    train_features = load_features(args, base_models, dataset_type=train_type)

    # Initialize results dictionary
    roc_auc_scores = {}

    # CASE 1: Same dataset → 5-fold Cross-Validation with One-vs-All
    if args.train_dataset == args.test_dataset:
        print("Evaluating with 5-fold cross-validation (one-vs-all for each source)...")

        # For EACH source, train a separate one-vs-all classifier
        for source in SOURCE_DICT.keys():
            print(f"\n{'='*60}")
            print(f"Training one-vs-all classifier for source: {source}")
            print(f"{'='*60}")

            # Create one-vs-all dataset for this source
            X = np.concatenate([
                train_features[source],
                np.concatenate([train_features[s] for s in SOURCE_DICT.keys() if s != source])
            ])
            Y = np.array(
                [1]*len(train_features[source]) +
                [0]*sum([len(train_features[s]) for s in SOURCE_DICT.keys() if s != source])
            )

            print(f"X shape: {X.shape}, Y shape: {Y.shape}")
            print(f"Positive samples ({source}): {np.sum(Y == 1)}")
            print(f"Negative samples (others): {np.sum(Y == 0)}")

            # 5-fold cross-validation for this source
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
            fold_scores = []

            for fold_idx, (train_index, test_index) in enumerate(skf.split(X, Y)):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]

                # Train classifier with fixed hyperparameters
                clf = RandomForestClassifier(
                    n_estimators=200,
                    random_state=args.seed,
                    bootstrap=False,
                    criterion='entropy',
                    max_depth=7,
                    n_jobs=8
                )
                clf.fit(X_train, Y_train)

                # Log model parameters (first fold only)
                if fold_idx == 0:
                    total_params = sum(count_tree_parameters(tree) for tree in clf.estimators_)
                    print(f"Total number of parameters: {total_params}")

                # Predict probabilities and compute ROC-AUC
                val_predicted_prob = clf.predict_proba(X_test)[:, 1]
                fold_score = roc_auc_score(Y_test, val_predicted_prob)
                fold_scores.append(fold_score)
                print(f"Fold {fold_idx + 1} ROC-AUC: {fold_score:.4f}")

            # Average ROC-AUC across folds for this source
            roc_auc_scores[source] = np.mean(fold_scores)
            print(f"\n{source} average ROC-AUC score: {roc_auc_scores[source]:.4f}")

        # Print overall average
        avg_score = np.mean(list(roc_auc_scores.values()))
        print(f"\n{'='*60}")
        print(f"Average ROC-AUC score across all sources: {avg_score:.4f}")
        print(f"{'='*60}")

        # Save results
        with open(os.path.join(base_out_dir, 'roc_auc_scores.txt'), 'w') as f:
            f.write(f'average roc_auc score: {avg_score}\n')
            for source, score in roc_auc_scores.items():
                f.write(f'{source} roc_auc score: {score}\n')

    # CASE 2: Different datasets → Train/Test Split with One-vs-All (OOD)
    else:
        print("Evaluating OOD setting (one-vs-all for each source)...")

        # Load test features
        print("Loading test features...")
        test_features = load_features(args, base_models, dataset_type=test_type)

        roc_curves = {}

        # For EACH source, train a separate one-vs-all classifier
        for source in SOURCE_DICT.keys():
            print(f"\n{'='*60}")
            print(f"Training one-vs-all classifier for source: {source}")
            print(f"{'='*60}")

            # Create one-vs-all training data for this source
            X_train = np.concatenate([
                train_features[source],
                np.concatenate([train_features[s] for s in SOURCE_DICT.keys() if s != source])
            ])
            Y_train = np.array(
                [1]*len(train_features[source]) +
                [0]*sum([len(train_features[s]) for s in SOURCE_DICT.keys() if s != source])
            )

            # Create one-vs-all test data for this source
            X_test = np.concatenate([
                test_features[source],
                np.concatenate([test_features[s] for s in SOURCE_DICT.keys() if s != source])
            ])
            Y_test = np.array(
                [1]*len(test_features[source]) +
                [0]*sum([len(test_features[s]) for s in SOURCE_DICT.keys() if s != source])
            )

            print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
            print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")
            print(f"Train positive samples ({source}): {np.sum(Y_train == 1)}")
            print(f"Test positive samples ({source}): {np.sum(Y_test == 1)}")

            # Train classifier on all training data with fixed hyperparameters
            clf = RandomForestClassifier(
                n_estimators=200,
                random_state=args.seed,
                bootstrap=False,
                criterion='entropy',
                max_depth=7,
                n_jobs=8
            )
            clf.fit(X_train, Y_train)

            # Log model parameters
            total_params = sum(count_tree_parameters(tree) for tree in clf.estimators_)
            print(f"Total number of parameters: {total_params}")

            # Predict on test data and compute ROC-AUC
            test_predicted_prob = clf.predict_proba(X_test)[:, 1]
            roc_auc_scores[source] = roc_auc_score(Y_test, test_predicted_prob)
            print(f"{source} ROC-AUC score: {roc_auc_scores[source]:.4f}")

            # Compute and save ROC curve
            fpr, tpr, _ = roc_curve(Y_test, test_predicted_prob)
            roc_curves[source] = (fpr, tpr)

            with open(os.path.join(base_out_dir, f'{args.task}_{source}_roc_curve.txt'), 'w') as f:
                for i in range(len(fpr)):
                    f.write(f'{fpr[i]:.6f}    {tpr[i]:.6f}\n')

        # Print overall average
        avg_score = np.mean(list(roc_auc_scores.values()))
        print(f"\n{'='*60}")
        print(f"Average ROC-AUC score across all sources: {avg_score:.4f}")
        print(f"{'='*60}")

        # Save results
        with open(os.path.join(base_out_dir, 'roc_auc_scores.txt'), 'w') as f:
            f.write(f'average roc_auc score: {avg_score}\n')
            for source, score in roc_auc_scores.items():
                f.write(f'{source} roc_auc score: {score}\n')


if __name__ == '__main__':
    main()

