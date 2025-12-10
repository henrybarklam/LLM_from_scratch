from importlib.metadata import version
import urllib.request
import tiktoken

print(f'Version of tiktoken is: {version("tiktoken")}')

tokenizer = tiktoken.get_encoding("gpt2")


text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace"

integers = tokenizer.encode(text, allowed_special = {"<|endoftext|>"})
# print(integers)

strings = tokenizer.decode(integers)
# print(strings)
# print(tokenizer.decode([34680]))

url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)


with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()
    enc_text = tokenizer.encode(raw_text)
    enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")


for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "----->", desired)


for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "----->", tokenizer.decode([desired]))
