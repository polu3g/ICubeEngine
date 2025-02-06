import openai

# Function to read file content
def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Function to remove specific keywords
def remove_keywords(text, keywords):
    for keyword in keywords:
        text = text.replace(keyword, "")
    return text

# Optional: Use OpenAI API for text processing
def rephrase_text_with_openai(text, openai_api_key, model_name="gpt-3.5-turbo"):
    openai.api_key = openai_api_key
    # Set your organization ID
    openai.organization = 'org-eKHX1QK2IhjR2fvh1jKa7Udf'  # Example: 'org-abc123'

    try:
        if model_name == "gpt-3.5-turbo" or model_name == "gpt-4":
            # Use ChatCompletion if using GPT-3.5-turbo or GPT-4
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Rephrase the following text:\n{text}"}
                ]
            )
            return response['choices'][0]['message']['content'].strip()
        else:
            # Use Completion for models like text-davinci-003
            response = openai.Completion.create(
                model=model_name,
                prompt=f"Rephrase the following text:\n{text}",
                max_tokens=1024
            )
            return response['choices'][0]['text'].strip()
    except openai.error.InvalidRequestError as e:
        print(f"Error: {e}")
        return text  # Return the original text if the model is not accessible

# Main function to handle file processing
def process_file(file_path, keywords, openai_api_key=None, model_name="gpt-3.5-turbo"):
    # Step 1: Read the file content
    text = read_file(file_path)
    
    # Step 2: Remove specified keywords
    cleaned_text = remove_keywords(text, keywords)
    
    # Step 3: Optionally rephrase using OpenAI API
    if openai_api_key:
        cleaned_text = rephrase_text_with_openai(cleaned_text, openai_api_key, model_name)
    
    # Step 4: Save the modified text back to a file
    output_file_path = "cleaned_" + file_path
    with open(output_file_path, 'w') as output_file:
        output_file.write(cleaned_text)
    
    print(f"Processed file saved as {output_file_path}")

# Example usage
if __name__ == "__main__":
    file_path = 'example.txt'  # Replace with your file path
    keywords_to_remove = ['keyword1', 'keyword2']  # List of keywords to remove
    API_KEY = 'sk-_Gb8ghowZKtc5lgXCtSUGaUn2T3W8TEacVJ3YA_GlRT3BlbkFJT4fjnEKca6duWEpoLuR3fOXuqIeD4fvSnGKNoVIUwA'
    openai_api_key = API_KEY  # Replace with your OpenAI API key
    
    # Use "gpt-3.5-turbo" or "text-davinci-003" depending on your access
    model_name = "gpt-3.5-turbo"  # Change to "text-davinci-003" if you don't have access to GPT-3.5 or GPT-4

    # Call the process_file function
    process_file(file_path, keywords_to_remove, openai_api_key, model_name)
