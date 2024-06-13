from data import SQuAD, IMDB, Hyperpartisan
import matplotlib.pyplot as plt
from tqdm import tqdm

dataset = Hyperpartisan(tokenizer_name="distilbert-base-uncased", max_seq_len=2048, num_workers=16, cache_dir="./data", shuffle=True)
lenghts = []
count = 0
portion = 0

for item in tqdm(dataset.data, desc=f"Processing train", unit="samples"):
    input_ids = item["input_ids"]
    # Count non-padding tokens
    lenght = sum(token_id != dataset.tokenizer.pad_token_id for token_id in input_ids)
    count += 1
    if lenght > 512:
        portion += 1
        
    lenghts.append(lenght)
        
plt.figure(figsize=(10, 6))
plt.hist(lenghts, bins=50, color='blue', edgecolor='black')
plt.title('Distribution of Unpadded Lengths of Texts')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

print(f"Total samples: {count}")
print(f"Samples with length > 512: {portion/count*100:.2f}%")