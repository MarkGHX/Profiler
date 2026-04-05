import os
import gc
import numpy as np
import torch
import torch.nn as nn
import json
import pickle
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel,
                          GPTJForCausalLM, GPTNeoXForCausalLM,
                          LlamaForCausalLM, GemmaForCausalLM, MistralForCausalLM)
from tqdm import tqdm

# Constants and Configuration
SOURCE_DICT = {
    'human': 0, 
    'gpt-3.5-turbo': 1, 
    'gpt-4-turbo-preview': 2, 
    'claude-3-sonnet': 3, 
    'claude-3-opus': 4, 
    'gemini-1.0-pro': 5
}

MODEL_ZOO = {
    'llama2-7b': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2-13b': 'meta-llama/Llama-2-13b-chat-hf',
    'llama3-8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'gemma-2b': 'google/gemma-1.1-2b-it',
    'gemma-7b': 'google/gemma-1.1-7b-it', 
    'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.2',
}

COMPLETION_PROMPT = '''Complete the following text: '''


# Model Utilities
def get_embedding_matrix(model):
    """Extract embedding matrix from different model architectures"""
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte.weight
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens.weight
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in.weight
    elif isinstance(model, GemmaForCausalLM):
        return model.model.embed_tokens.weight
    elif isinstance(model, MistralForCausalLM):
        return model.model.embed_tokens.weight
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def get_embeddings(model, input_ids):
    """Get embeddings from different model architectures"""
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte(input_ids).half()
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens(input_ids)
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in(input_ids).half()
    elif isinstance(model, GemmaForCausalLM):
        return model.model.embed_tokens(input_ids)
    elif isinstance(model, MistralForCausalLM):
        return model.model.embed_tokens(input_ids)
    else:
        raise ValueError(f"Unknown model type: {type(model)}")





# Loss and Feature Computation
def compute_symmetric_kl_divergence(p, q):
    """Calculate symmetric KL divergence between two context loss sequences (Eq. 4)"""
    p = torch.tensor(p, dtype=torch.float32)
    q = torch.tensor(q, dtype=torch.float32)
    # L1 normalize to form proper distributions over positions
    p = p / p.sum()
    q = q / q.sum()
    # kl_div expects input=log-probs, target=probs; computes D(target || input)
    kl_pq = nn.functional.kl_div(q.log(), p, reduction='sum')
    kl_qp = nn.functional.kl_div(p.log(), q, reduction='sum')
    return (kl_pq + kl_qp).detach().cpu().numpy()


def compute_loss(args, pred_logits, targets, text_slice):
    """
    Compute independent and correlated features from loss sequences
    """
    # Convert full context window to half window (for positions on each side)
    half_window = args.context_window // 2

    front_offset = 0
    if text_slice.start < half_window - 1:
        front_offset = half_window - 1 - text_slice.start
    if text_slice.start + front_offset > text_slice.stop - half_window:
        raise ValueError(f"Invalid text slice. Too short text sample.")

    # Extract probabilities from logits
    probs = pred_logits[0, text_slice.start+front_offset:text_slice.stop-half_window, :].detach().cpu().numpy()
    target_ids = []
    for i in range(-half_window+1, half_window+1, 1):
        target_ids.append(targets[text_slice.start+front_offset+i:text_slice.stop-half_window+i].detach().cpu().numpy())
    target_ids = np.array(target_ids)
    
    # Calculate context losses (use cross entropy calculation)
    ce = []
    for i in range(target_ids.shape[0]):
        ce.append(nn.CrossEntropyLoss(reduction='none')(torch.tensor(probs), torch.tensor(target_ids[i])).detach().cpu().numpy())
    ce = np.array(ce).T

    # Calculate residual features
    ce_grad = np.gradient(ce, axis=0)
    
    # Compute statistics (5 statistics: mean, max, min, std, median)
    loss_statistics = []
    for i in range(ce.shape[1]):
        # CE statistics (5 instead of 6 - no variance)
        loss_statistics.extend([np.mean(ce[:,i]), np.max(ce[:,i]), np.min(ce[:,i]), np.std(ce[:,i]), np.median(ce[:,i])])
        # CE_grad statistics (5 instead of 6 - no variance)
        loss_statistics.extend([np.mean(ce_grad[:,i]), np.max(ce_grad[:,i]), np.min(ce_grad[:,i]), np.std(ce_grad[:,i]), np.median(ce_grad[:,i])])
    
    # Calculate KL divergence between context loss pairs (Eq. 4)
    kl_divergence = []
    for j in range(ce.shape[1]):
        for k in range(j+1, ce.shape[1]):
            kl_divergence.append(compute_symmetric_kl_divergence(ce[:, j], ce[:, k]))
    kl_divergence = np.array(kl_divergence)
    
    all_features = np.concatenate([loss_statistics, kl_divergence])
    return all_features


# Feature Extraction for Single Sample
def detect_single_sample(args, model, tokenizer, sample, prompt=COMPLETION_PROMPT, device='cuda'):
    """Extract features for a single text sample"""

    # Truncate the sample
    if args.sample_clip and len(sample) > args.sample_clip:
        sample = sample[:args.sample_clip]

    # Encode prompt and text
    prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids[:, 1:]
    text_ids = tokenizer(sample, return_tensors='pt').input_ids[:, 1:]

    text_slice = slice(prompt_ids.shape[1], prompt_ids.shape[1] + text_ids.shape[1])

    input_ids = torch.cat([prompt_ids, text_ids], dim=1)
    input_ids = input_ids[0].to(device)

    # Get embeddings
    prompt_embeds = get_embeddings(model, prompt_ids.to(device)).detach()
    text_embeds = get_embeddings(model, text_ids.to(device)).detach()
    full_embeds = torch.cat([prompt_embeds, text_embeds], dim=1)

    # Forward pass
    logits = model(inputs_embeds=full_embeds).logits

    # Compute loss features
    loss_features = compute_loss(args, logits, input_ids, text_slice)

    return loss_features


# Data Loading and Preprocessing
def load_dataset(task, source, dataset_type='normal'):
    """
    Load JSON data and extract text based on task.

    Args:
        task: One of ['Arxiv', 'Code', 'Yelp', 'Essay', 'Creative', 'GCJ']
        source: One of SOURCE_DICT keys
        dataset_type: 'normal' or 'paraphrased'

    Returns:
        List of text samples
    """
    # Determine dataset path
    if source == 'human':
        file_path = f'./Dataset/{task}/{task}_human.json'
    else:
        if dataset_type == 'paraphrased':
            file_path = f'./Paraphrased_Dataset/{task}/{task}_{source}.json'
        else:
            file_path = f'./Dataset/{task}/{task}_{source}.json'

    # Load JSON data
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract text based on task and data format
    texts = []

    # Check if data is already a list of strings (AI-generated format)
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
        texts = data
    # Otherwise, extract based on task-specific structure (human format)
    elif task == 'Arxiv':
        texts = [sample['abs'] for sample in data]
    elif task == 'Code':
        texts = [sample[0] + sample[1] for sample in data]
    elif task == 'Yelp' or task == 'GCJ':
        texts = data  # Already strings
    elif task == 'Essay' or task == 'Creative':
        texts = [sample['essay'] for sample in data]
    else:
        raise ValueError(f"Unknown task: {task}")

    return texts


# Feature Generation Orchestrator
def data_generation(args, base_model, dataset_type='normal'):
    """
    Generate features for all sources using specified detection model.

    Args:
        args: Command line arguments
        base_model: Detection model name from MODEL_ZOO
        dataset_type: 'normal' or 'paraphrased'

    Returns:
        True if successful, False otherwise
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create result directory
    if dataset_type == 'normal':
        result_dir = f'./Feature/Profiler/{args.task}'
    else:
        result_dir = f'./Paraphrased_Feature/Profiler/{args.task}'
    os.makedirs(result_dir, exist_ok=True)

    # Load detection model
    print(f"Loading detection model: {base_model}")
    model_path = MODEL_ZOO[base_model]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, device_map='auto')
    model.eval()

    # Process each source
    for source in SOURCE_DICT.keys():
        # Skip human for paraphrased dataset
        if dataset_type == 'paraphrased' and source == 'human':
            continue

        print(f"Processing {source}...")

        # Check if feature file already exists
        feature_file = f'{result_dir}/{base_model}_{source}_context_window_{args.context_window}.pkl'
        if os.path.exists(feature_file):
            print(f"Feature file already exists: {feature_file}")
            continue

        # Load data
        try:
            texts = load_dataset(args.task, source, dataset_type)
        except FileNotFoundError:
            print(f"Data file not found for {source} in {dataset_type} dataset")
            continue

        # Extract features
        features = []
        for idx, sample in enumerate(tqdm(texts, desc=f"Extracting features for {source}")):
            try:
                # Special case: skip index 83 for claude-3-sonnet on Yelp
                if args.task == 'Yelp' and source == 'claude-3-sonnet' and idx == 83:
                    continue

                feature = detect_single_sample(args, model, tokenizer, sample, device=device)
                features.append(feature)
            except Exception as e:
                print(f"Error processing sample {idx} for {source}: {e}")
                continue

        # Save features
        with open(feature_file, 'wb') as f:
            pickle.dump(features, f)

        print(f"Saved {len(features)} features to {feature_file}")

    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return True


# Feature Aggregation and Loading
def aggregate_features_across_models(features_dict, base_models, source):
    """
    Aggregate features from multiple detection models.

    Args:
        features_dict: Dictionary mapping model_name -> list of features
        base_models: List of base model names to use
        source: Source name (for special case handling)

    Returns:
        Numpy array of aggregated features
    """
    # Get number of samples (use first available model)
    first_model = list(features_dict.keys())[0]
    num_samples = len(features_dict[first_model])

    aggregated_features = []
    for sample_idx in range(num_samples):
        sample_features = []
        for model_name in base_models:
            if model_name in features_dict:
                sample_features.append(features_dict[model_name][sample_idx])

        # Concatenate features from all models
        aggregated_features.append(np.concatenate(sample_features))

    return np.array(aggregated_features)


def load_features(args, base_models, dataset_type='normal'):
    """
    Load pre-computed features for all sources.

    Args:
        args: Command line arguments
        base_models: List of base model names to use
        dataset_type: 'normal' or 'paraphrased'

    Returns:
        Dictionary mapping source -> aggregated features array
    """
    # Determine result directory
    if dataset_type == 'normal':
        result_dir = f'./Feature/Profiler/{args.task}'
    else:
        result_dir = f'./Paraphrased_Feature/Profiler/{args.task}'

    all_features = {}

    for source in SOURCE_DICT.keys():
        # Skip human for paraphrased dataset
        if dataset_type == 'paraphrased' and source == 'human':
            # Load human from normal dataset instead
            features_dict = {}
            for base_model in base_models:
                feature_file = f'./Feature/Profiler/{args.task}/{base_model}_human_context_window_{args.context_window}.pkl'
                if os.path.exists(feature_file):
                    with open(feature_file, 'rb') as f:
                        features_dict[base_model] = pickle.load(f)

            if features_dict:
                all_features['human'] = aggregate_features_across_models(features_dict, base_models, 'human')
            continue

        # Load features from all detection models
        features_dict = {}
        for base_model in base_models:
            feature_file = f'{result_dir}/{base_model}_{source}_context_window_{args.context_window}.pkl'
            if os.path.exists(feature_file):
                with open(feature_file, 'rb') as f:
                    features_dict[base_model] = pickle.load(f)

        # Aggregate features across models
        if features_dict:
            all_features[source] = aggregate_features_across_models(features_dict, base_models, source)

    return all_features

