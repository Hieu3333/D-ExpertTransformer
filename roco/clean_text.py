import re

# Input and output file
input_path = "roco/captions.txt"  # change if needed
output_path = "roco/normalized_captions.txt"

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        line = line.strip()
        # Optional: Normalize to lowercase and remove accents using basic Python if needed
        # Remove unwanted characters: keep only a-z, A-Z, space, and ()
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', line)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # remove extra whitespace
        if cleaned:
            outfile.write(cleaned + "\n")

print("✅ Normalized captions saved to", output_path)
