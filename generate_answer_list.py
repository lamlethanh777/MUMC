"""
Utility script to generate answer list from training/validation data
This creates the answer_trainval_list.json file needed for VQA evaluation
"""
import json
import os
import argparse
from pathlib import Path


def create_answer_list(train_files, output_file):
    """
    Create answer list from training/validation files
    
    Args:
        train_files: List of JSON files containing training/validation data
        output_file: Path to save the answer list
    """
    all_answers = set()
    
    print("Loading data files...")
    for train_file in train_files:
        if not Path(train_file).exists():
            print(f"Warning: File not found: {train_file}")
            continue
            
        print(f"  - Loading {train_file}")
        with open(train_file, 'r') as f:
            data = json.load(f)
            
        # Extract answers
        for item in data:
            if 'answer' in item:
                all_answers.add(item['answer'])
    
    # Sort answers alphabetically
    answer_list = sorted(list(all_answers))
    
    print(f"\nFound {len(answer_list)} unique answers")
    
    # Save answer list
    with open(output_file, 'w') as f:
        json.dump(answer_list, f, indent=2)
    
    print(f"Answer list saved to: {output_file}")
    
    # Print statistics
    print("\nAnswer statistics:")
    print(f"  Total unique answers: {len(answer_list)}")
    print(f"  First 10 answers: {answer_list[:10]}")
    if len(answer_list) > 10:
        print(f"  Last 10 answers: {answer_list[-10:]}")


def main():
    parser = argparse.ArgumentParser(description='Generate answer list from VQA dataset')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['pathvqa', 'rad', 'slake'],
                       help='Dataset name')
    parser.add_argument('--data_root', type=str, default='.',
                       help='Root directory containing data folders (use /kaggle/input/mevf-datasets for Kaggle)')
    parser.add_argument('--output_root', type=str, default=None,
                       help='Root directory for output files (use /kaggle/working for Kaggle, defaults to data_root)')
    
    args = parser.parse_args()
    
    # Construct paths based on data_root and output_root
    if args.data_root != '.':
        data_root = args.data_root
    else:
        data_root = ''
    
    if args.output_root is not None:
        output_root = args.output_root
    else:
        output_root = data_root
    
    # Define file paths for each dataset
    dataset_configs = {
        'pathvqa': {
            'train_files': [
                os.path.join(data_root, 'data_PathVQA/trainset.json') if data_root else 'data_PathVQA/trainset.json',
                os.path.join(data_root, 'data_PathVQA/valset.json') if data_root else 'data_PathVQA/valset.json'
            ],
            'output_file': os.path.join(output_root, 'data_PathVQA/answer_trainval_list.json') if output_root else 'data_PathVQA/answer_trainval_list.json'
        },
        'rad': {
            'train_files': [
                os.path.join(data_root, 'data_RAD/trainset.json') if data_root else 'data_RAD/trainset.json'
            ],
            'output_file': os.path.join(output_root, 'data_RAD/answer_all_list.json') if output_root else 'data_RAD/answer_all_list.json'
        },
        'slake': {
            'train_files': [
                os.path.join(data_root, 'data_Slake/en/slake_train.json') if data_root else 'data_Slake/en/slake_train.json',
                os.path.join(data_root, 'data_Slake/en/slake_val.json') if data_root else 'data_Slake/en/slake_val.json'
            ],
            'output_file': os.path.join(output_root, 'data_Slake/en/answer_trainval_list.json') if output_root else 'data_Slake/en/answer_trainval_list.json'
        }
    }
    
    config = dataset_configs[args.dataset]
    
    # Create answer list
    create_answer_list(config['train_files'], config['output_file'])


if __name__ == '__main__':
    main()
