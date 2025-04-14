from roco.tokenizer import Tokenizer
t = Tokenizer()
text = "a normal size uterine cavity with both fallopian tubes demonstrated and there was free spillage of the contrast material both the cervical canal and the uterine cavity are normal in outline"
print(t.decode(t.encode(text)))