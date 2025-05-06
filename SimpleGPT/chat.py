import torch
from model import generate

device = 'gpu' if torch.cuda.is_available() else 'cpu'

def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

def encode(text, stoi):
    # convert characters to tokens using stoi
    return [stoi.get(char, 0) for char in text]

def decode(tokens, itos):
    # convert tokens to characters
    return ''.join([itos.get(token, '<UNK>') for token in tokens])


def generate_response(model, input_text, max_tokens=100):
    # convert text to tensor
    input_tokens = torch.tensor(encode(input_text), dtype=torch.long, device=device).unsqueeze(0)
    # generate response tokens
    generate_tokens = generate(input_tokens, max_tokens)
    # decode generated tokens to text
    response_text = decode(generate_tokens[0].tolist())
    return response_text

def main():
    model_path = 'shakespeare.pth'
    model = load_model(model_path)
    
    print("Chatbot initialized, Type 'exit' to end the conversation")

    # interaction loop
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        
        # generate response
        response = generate_response(model, user_input)
        print("Chatbot:", response)
        
if __name__ == "__main__":
    main()
