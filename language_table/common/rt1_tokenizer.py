import tensorflow_hub as hub


def tokenize_text(text):
  """Tokenizes the input text given a tokenizer."""
  embed = hub.load("/gemini/data-2/code/universal_sentence_encoder")
  tokens = embed([text])
  return tokens

# text ='push the blue triangle closer to yellow heart'
# print(tokenize_text(text))

