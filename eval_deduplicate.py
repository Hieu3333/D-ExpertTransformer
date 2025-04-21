import json

# Input and output file paths
input_file = "data/cleaned_DeepEyeNet_val.json"        # Replace with your real filename
output_file = "data/dedup_cleaned_DeepEyeNet_val.json"

# Load the dataset
with open(input_file, 'r') as f:
    data = json.load(f)

# Store seen captions to avoid duplicates
seen_captions = set()
filtered_data = []

for entry in data:
    # Each entry is a dict with one key (image path)
    image_path, meta = next(iter(entry.items()))
    caption = meta.get("original", "").strip().lower()

    if caption not in seen_captions:
        seen_captions.add(caption)
        filtered_data.append(entry)

# Write the filtered result to a new JSON file
with open(output_file, 'w') as f:
    json.dump(filtered_data, f, indent=2)

print(f"Filtered dataset saved to: {output_file}")
