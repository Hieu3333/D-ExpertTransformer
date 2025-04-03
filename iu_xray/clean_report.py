import json

# Load your dataset (assuming it's stored as a JSON file)
with open("iu_xray/annotations.json", "r") as f:
    data = json.load(f)

new_data = {}

# Process each split (train, val, test)
for split in ["train", "val", "test"]:
    new_data[split] = []  # Initialize empty list for each split
    
    for entry in data.get(split, []):  # Ensure the split exists
        report = entry["report"]
        
        for i, img in enumerate(entry["image_path"]):  # Iterate over images
            new_entry = {
                "id": f"{entry['id']}_{i}",  # Unique ID per image
                "report": report,
                "image_path": img,  # Each entry now has a single image
                "split": split
            }
            new_data[split].append(new_entry)

# Save the transformed dataset
with open("iu_xray.json", "w") as f:
    json.dump(new_data, f, indent=4)

print("Dataset successfully split for train, val, and test!")
