import json
from openai import OpenAI

# Initialize the OpenAI client with the provided base URL and API key
client = OpenAI(
    base_url="https://api.yescale.io/v1",  # The base URL for the API
    api_key="sk-LsduO6LmExzS1OA8DbuiINzgyOx14B1nlwOt6xkvgU9v4AX6",  # Your OpenAI API key
    timeout=120
)

# Function to generate 3 captions for each image
def generate_captions_for_image(clinical_description, keywords):
    # Prepare the messages for the GPT model
    prompt = f"Given the clinical description: '{clinical_description}' and keywords: '{keywords}', generate two simple and easy-to-read clinical captions that summarize the description. The captions should not include numbered lists, bullet points, or any additional formatting. Only the plain text captions should be generated, and they should reflect the original clinical information clearly and concisely without adding any new phrase, information or paraphrasing."

    try:
        # Use the chat completion model to get the generated captions
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using GPT-3.5 model
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}]
        )
        
        # Extract generated text (captions)
        generated_text = response.choices[0].message.content
        
        if not generated_text:
            print(f"Warning: No captions generated for clinical description: {clinical_description}")
            return ["No captions generated."]  # Return a default message if no text is generated
        
        # Split the generated text into three different captions (assuming they are delimited by a newline)
        captions = generated_text.split("\n")
        return captions[:2]  # Return the first 3 captions
    
    except Exception as e:
        print(f"Error generating captions for clinical description: {clinical_description}")
        print(f"Error: {e}")
        return ["Error generating captions."]  # Return an error message in case of failure

# Load your dataset (replace this with your actual dataset path)
dataset_path = "data/DeepEyeNet_test.json"
with open(dataset_path, "r") as f:
    dataset = json.load(f)

# Split the dataset into two halves
halfway_point = len(dataset) // 2
first_half = dataset[:halfway_point]
second_half = dataset[halfway_point:]

# Function to process and save the augmented dataset
def process_and_save_dataset(dataset_part, file_path):
    augmented_dataset = {}

    # Iterate through the dataset and augment descriptions
    for entry in dataset_part:
        for img_path, data in entry.items():
            clinical_description = data["clinical-description"]
            keywords = data["keywords"]
            
            # Generate 3 captions using GPT-3.5
            captions = generate_captions_for_image(clinical_description, keywords)
            
            # Store the augmented captions in the dataset (for each image)
            augmented_dataset[img_path] = {
                "keywords": keywords,
                "groundtruth": clinical_description, 
                "clinical-description": captions
            }

    # Save the augmented dataset to a new file or append to it
    with open(file_path, "w") as f:
        json.dump(augmented_dataset, f, indent=2)

# Process and save the first half of the dataset
first_half_path = "data/augmented_DeepEyeNet_test_part1.json"
process_and_save_dataset(first_half, first_half_path)

print("First half of the dataset augmented and saved.")

# Process and append the second half of the dataset
second_half_path = "data/augmented_DeepEyeNet_test_part2.json"
process_and_save_dataset(second_half, second_half_path)

print("Second half of the dataset augmented and appended.")
