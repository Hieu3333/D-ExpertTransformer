from roco.tokenizer import Tokenizer
t = Tokenizer()
text = "patient"
print(t.decode(t.encode(text)))