"""
Create answer list files for VQA datasets
The answer list is a JSON array containing all unique answers from the training/validation sets
This is required for model evaluation
"""
import json
import os
from pathlib import Path
from collections import Counter


def extract_answers_from_dataset(json_files):
    """
    Extract all answers from dataset JSON files
    
    Args:
        json_files: List of paths to JSON files
        
    Returns:
        List of all answers (with duplicates for statistics)
    """
    all_answers = []
    
    for json_file in json_files:
        if not os.path.exists(json_file):
            print(f"  ⚠️  Warning: File not found: {json_file}")
            continue
            
        print(f"  📂 Loading {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract answers
        for item in data:
            if 'answer' in item:
                # Convert to string and normalize: strip whitespace
                answer = str(item['answer']).strip()
                all_answers.append(answer)
            else:
                print(f"    ⚠️  Warning: No 'answer' field in item with qid={item.get('qid', 'unknown')}")
    
    return all_answers


def create_answer_list(dataset_name, data_files, output_file, normalize_case=False):
    """
    Create answer list file for a dataset
    
    Args:
        dataset_name: Name of the dataset (for display)
        data_files: List of JSON files to extract answers from
        output_file: Path to save the answer list
        normalize_case: Whether to convert all answers to lowercase
    """
    print(f"\n{'='*80}")
    print(f"Creating answer list for {dataset_name}")
    print(f"{'='*80}")
    
    # Extract all answers
    all_answers = extract_answers_from_dataset(data_files)
    
    if not all_answers:
        print(f"  ❌ No answers found! Check your data files.")
        return
    
    print(f"\n  📊 Statistics:")
    print(f"     Total answers (with duplicates): {len(all_answers)}")
    
    # Count answer frequencies
    answer_counts = Counter(all_answers)
    print(f"     Unique answers: {len(answer_counts)}")
    
    # Optionally normalize to lowercase
    if normalize_case:
        print(f"     Normalizing to lowercase...")
        normalized_counts = Counter()
        for answer, count in answer_counts.items():
            normalized_counts[answer.lower()] += count
        answer_counts = normalized_counts
        print(f"     Unique answers after normalization: {len(answer_counts)}")
    
    # Create sorted answer list (alphabetically)
    answer_list = sorted(answer_counts.keys())
    
    # Show most common answers
    print(f"\n  🔝 Top 10 most common answers:")
    for answer, count in answer_counts.most_common(10):
        print(f"     '{answer}': {count} times ({count/len(all_answers)*100:.1f}%)")
    
    # Show sample answers
    print(f"\n  📝 Sample answers:")
    print(f"     First 10: {answer_list[:10]}")
    if len(answer_list) > 10:
        print(f"     Last 10: {answer_list[-10:]}")
    
    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save answer list
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(answer_list, f, indent=2, ensure_ascii=False)
    
    print(f"\n  ✅ Answer list saved to: {output_file}")
    print(f"     Total unique answers in list: {len(answer_list)}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate answer lists for VQA datasets')
    parser.add_argument('--data_root', type=str, default='.',
                       help='Root directory containing data folders (use /kaggle/input/mevf-datasets for Kaggle)')
    parser.add_argument('--output_root', type=str, default=None,
                       help='Root directory for output files (use /kaggle/working for Kaggle, defaults to data_root)')
    parser.add_argument('--datasets', nargs='+', choices=['rad', 'pathvqa', 'slake', 'all'],
                       default=['all'], help='Which datasets to process (default: all available)')
    
    args = parser.parse_args()
    data_root = args.data_root
    output_root = args.output_root if args.output_root else data_root
    
    print("="*80)
    print("VQA Answer List Generator")
    print("="*80)
    print("\nThis script creates answer list files required for VQA evaluation.")
    print("Answer lists contain all unique answers from training/validation data.")
    print(f"\nData root (input): {data_root}")
    print(f"Output root: {output_root}\n")
    
    # Define datasets and their files
    datasets = {}
    
    # VQA-RAD
    if 'all' in args.datasets or 'rad' in args.datasets:
        rad_input = os.path.join(data_root, 'data_RAD') if data_root != '.' else 'data_RAD'
        rad_output = os.path.join(output_root, 'data_RAD') if output_root != '.' else 'data_RAD'
        datasets['VQA-RAD'] = {
            'files': [
                os.path.join(rad_input, 'trainset.json'),
                os.path.join(rad_input, 'testset.json')  # Include test set for complete answer coverage
            ],
            'output': os.path.join(rad_output, 'answer_all_list.json'),
            'normalize': False  # RAD uses mixed case (Yes/No)
        }
    
    # PathVQA
    if 'all' in args.datasets or 'pathvqa' in args.datasets:
        path_input = os.path.join(data_root, 'data_PathVQA') if data_root != '.' else 'data_PathVQA'
        path_output = os.path.join(output_root, 'data_PathVQA') if output_root != '.' else 'data_PathVQA'
        datasets['PathVQA'] = {
            'files': [
                os.path.join(path_input, 'trainset.json'),
                os.path.join(path_input, 'valset.json')
            ],
            'output': os.path.join(path_output, 'answer_trainval_list.json'),
            'normalize': False  # Keep original case
        }
    
    # Slake (optional)
    if 'all' in args.datasets or 'slake' in args.datasets:
        slake_input = os.path.join(data_root, 'data_Slake') if data_root != '.' else 'data_Slake'
        slake_output = os.path.join(output_root, 'data_Slake') if output_root != '.' else 'data_Slake'
        if os.path.exists(slake_input) or data_root != '.':
            datasets['Slake'] = {
                'files': [
                    os.path.join(slake_input, 'en', 'slake_train.json'),
                    os.path.join(slake_input, 'en', 'slake_val.json')
                ],
                'output': os.path.join(slake_output, 'en', 'answer_trainval_list.json'),
                'normalize': False
            }
    
    # Process each dataset
    success_count = 0
    failed_count = 0
    
    for dataset_name, config in datasets.items():
        try:
            # Check if at least one file exists
            existing_files = [f for f in config['files'] if os.path.exists(f)]
            
            if not existing_files:
                print(f"\n⏭️  Skipping {dataset_name}: No data files found")
                print(f"   Expected files: {config['files']}")
                continue
            
            create_answer_list(
                dataset_name=dataset_name,
                data_files=config['files'],
                output_file=config['output'],
                normalize_case=config['normalize']
            )
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ Error processing {dataset_name}: {e}")
            failed_count += 1
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successfully created: {success_count} answer list(s)")
    if failed_count > 0:
        print(f"❌ Failed: {failed_count} dataset(s)")
    print("\n✨ Done! You can now use these answer lists for testing.")
    print("\nNext steps:")
    print("  1. Verify the answer list files were created")
    print("  2. Run: python test_vqa.py --dataset_use <dataset> --checkpoint <checkpoint.pth>")


if __name__ == '__main__':
    main()
