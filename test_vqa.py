"""
Test script for MUMC VQA model
Evaluates a trained model on test set and logs results
"""
import argparse
import os
import sys
import ruamel.yaml as yaml
import time
import datetime
import json
from pathlib import Path
import torch
import torch.distributed as dist

from models.model_vqa import MUMC_VQA
from models.vision.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
import utils
from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn
from vqaEvaluate import compute_vqa_acc


@torch.no_grad()
def evaluation(model, data_loader, device, config):
    """Evaluate model on test set"""
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50

    result = []
    answer_list = [answer + config['eos'] for answer in data_loader.dataset.answer_list]

    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        topk_ids, topk_probs = model(image, question, answer_list, train=False, k=config['k_test'])

        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            ques_id = int(ques_id.item())
            _, pred = topk_prob.max(dim=0)
            result.append({"qid": ques_id, "answer": data_loader.dataset.answer_list[topk_id[pred]]})
    
    return result


def main(args, config):
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fix seed for reproducibility
    utils.set_seed(args.seed)

    #### Loading Dataset ####
    print(f'Creating vqa {args.dataset_use} test dataset')
    datasets = create_dataset(args.dataset_use, config)
    
    # Get only test dataset (index 1)
    test_dataset = datasets[1]
    print(f'Test dataset size: {len(test_dataset)}')

    # Create test loader
    test_loader = create_loader([test_dataset], [None],
                                 batch_size=[config['batch_size_test']],
                                 num_workers=[4], is_trains=[False],
                                 collate_fns=[None])[0]

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Creating Model ####
    print("Creating model")
    model = MUMC_VQA(config=config, text_encoder=args.text_encoder, 
                     text_decoder=args.text_decoder, tokenizer=tokenizer)
    model = model.to(device)

    # Load checkpoint
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        print("Please provide a valid checkpoint path using --checkpoint argument")
        return

    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Reshape positional embedding to accommodate for image resolution change
    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], 
                                                model.visual_encoder)
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

    # Load state dict
    msg = model.load_state_dict(state_dict, strict=False)
    print('Missing keys:', msg.missing_keys)
    print('Unexpected keys:', msg.unexpected_keys)

    print("\nStart testing\n")
    start_time = time.time()

    # Run evaluation
    vqa_result = evaluation(model, test_loader, device, config)

    # Save results
    result_file = os.path.join(args.output_dir, f'{args.dataset_use}_test_result.json')
    json.dump(vqa_result, open(result_file, 'w'))
    print(f"\nResults saved to: {result_file}")

    # Compute accuracy
    print("\n" + "="*80)
    print("Computing accuracy metrics...")
    print("="*80 + "\n")
    
    from vqaTools.vqa import VQA
    from vqaTools.vqaEval import VQAEval
    
    quesFile = config[args.dataset_use]['test_file'][0]
    vqa = VQA(quesFile, quesFile)
    vqaRes = vqa.loadRes(result_file, quesFile)
    
    # Create vqaEval object
    vqaEval = VQAEval(vqa, vqaRes, n=2)
    vqaEval.evaluate()
    
    # Print accuracies
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nOverall Accuracy: {vqaEval.accuracy['overall']:.2f}%\n")
    
    print("Per Answer Type Accuracy:")
    print("-" * 40)
    for ansType in vqaEval.accuracy['perAnswerType']:
        print(f"  {ansType:20s}: {vqaEval.accuracy['perAnswerType'][ansType]:.2f}%")
    print("\n" + "="*80 + "\n")
    
    # Save accuracy results
    accuracy_file = result_file.replace('.json', '_acc.json')
    json.dump(vqaEval.accuracy, open(accuracy_file, 'w'))
    print(f"Accuracy results saved to: {accuracy_file}")
    
    # Save comparison results
    compare_file = result_file.replace('.json', '_compare.json')
    json.dump(vqaEval.ansComp, open(compare_file, 'w'))
    print(f"Comparison results saved to: {compare_file}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'\nTotal testing time: {total_time_str}')

    # Save summary log
    summary = {
        'dataset': args.dataset_use,
        'checkpoint': args.checkpoint,
        'test_size': len(test_dataset),
        'overall_accuracy': vqaEval.accuracy['overall'],
        'per_answer_type': vqaEval.accuracy['perAnswerType'],
        'testing_time': total_time_str,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    summary_file = os.path.join(args.output_dir, f'{args.dataset_use}_test_summary.json')
    json.dump(summary, open(summary_file, 'w'), indent=2)
    print(f"Summary saved to: {summary_file}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test MUMC VQA model')
    parser.add_argument('--dataset_use', default='pathvqa', 
                       help='Choose medical vqa dataset (rad, pathvqa, slake)')
    parser.add_argument('--checkpoint', required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--output_dir', default='./test_results',
                       help='Directory to save test results')
    parser.add_argument('--config', default='./configs/VQA_kaggle.yaml',
                       help='Path to config file (VQA_kaggle.yaml for Kaggle paths, VQA_local.yaml for local)')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    # Load config
    yaml_loader = yaml.YAML(typ='safe', pure=True)
    with open(args.config, 'r') as f:
        config = yaml_loader.load(f)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Set up logging
    sys.stdout = utils.Logger(filename=os.path.join(args.output_dir, "test_log.txt"), 
                              stream=sys.stdout)

    print("="*80)
    print("MUMC VQA Testing")
    print("="*80)
    print("\nConfiguration:")
    print("-" * 40)
    print(f"Dataset: {args.dataset_use}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print("-" * 40 + "\n")

    main(args, config)
