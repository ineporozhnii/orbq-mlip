import json
import random

def main():
    input_file = "/scratch/igornep/MLGuessQC/data/OMol25_metal_neutral_dataset/datasplits_617554.json"
    input_file_100k = "/scratch/igornep/MLGuessQC/data/OMol25_metal_neutral_dataset/datasplits_100000.json"
    output_file = "/scratch/igornep/MLGuessQC/data/OMol25_metal_neutral_dataset/datasplits_300000.json"
    
    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        splits_617k = json.load(f)
        
    print(f"Loading {input_file_100k}...")
    with open(input_file_100k, 'r') as f:
        splits_100k = json.load(f)
        
    print(f"Original 617k train size: {len(splits_617k['train'])}")
    print(f"Original 100k train size: {len(splits_100k['train'])}")
    
    # We want to keep all 100k indices
    train_100k_set = set(splits_100k['train'])
    
    # Get remaining available indices from the 617k split that are not in 100k split
    remaining_indices = [idx for idx in splits_617k['train'] if idx not in train_100k_set]
    
    # Sample exactly 200,000 from the remaining dataset so the sum is 300,000
    random.seed(42) # for reproducibility
    sampled_remaining = random.sample(remaining_indices, 200000)
    
    new_train = splits_100k['train'] + sampled_remaining
    
    new_splits = {
        "train": new_train,
        "validation": splits_617k["validation"],
        "test": splits_617k["test"]
    }
    
    print(f"New train size: {len(new_splits['train'])}")
    
    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(new_splits, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
