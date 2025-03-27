from gpt4all import GPT4All

# Define model name and path
model_name = "ggml-mistral-7b-instruct-v0.2.gguf"
model_path = "C:/Users/chakrapa/.cache/gpt4all/"  # Use absolute path if needed

# Download model using GPT4All.retrieve_model()
GPT4All.retrieve_model(model_name, model_path=model_path)

# Load the model
llm = GPT4All(model=f"{model_path}/{model_name}", n_ctx=512)
