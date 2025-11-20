import json
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, 'sampled_dataset.jsonl')
OUTPUT_TASK1 = os.path.join(BASE_DIR, 'dataset_task1.jsonl')
OUTPUT_TASK2 = os.path.join(BASE_DIR, 'dataset_task2.jsonl')

def split_dataset():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    data_by_dimension = defaultdict(list)
    
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            item = json.loads(line)
            data_by_dimension[item['dimension_name']].append(item)
            
    task1_data = []
    task2_data = []
    
    for dim, items in data_by_dimension.items():
        # Assuming 10 items per dimension, split 5/5
        # If not exactly 10, we'll split half/half
        mid = len(items) // 2
        task1_data.extend(items[:mid])
        task2_data.extend(items[mid:])
        
    print(f"Total items: {sum(len(v) for v in data_by_dimension.values())}")
    print(f"Task 1 items: {len(task1_data)}")
    print(f"Task 2 items: {len(task2_data)}")
    
    with open(OUTPUT_TASK1, 'w') as f:
        for item in task1_data:
            f.write(json.dumps(item) + '\n')
            
    with open(OUTPUT_TASK2, 'w') as f:
        for item in task2_data:
            f.write(json.dumps(item) + '\n')
            
    print(f"Saved to {OUTPUT_TASK1} and {OUTPUT_TASK2}")

if __name__ == "__main__":
    split_dataset()
