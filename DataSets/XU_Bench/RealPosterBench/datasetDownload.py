from datasets import load_dataset

dataset = load_dataset("PosterCraft/Poster100K")
print(dataset)

# Access a sample
sample = dataset['train'][0]
print(sample['caption'])
print("Text regions:", sample['mask_regions'])
print("Batch:", sample['batch_name'])
print("Normalized path:", sample['normalized_path'])
# sample['image'].show()
