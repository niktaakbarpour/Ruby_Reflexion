from model import DeepSeekR1, Message
import sys

print("LOADING MODEL...", flush=True)
model = DeepSeekR1()
print("MODEL LOADED", flush=True)

messages = [Message(role="user", content="What is the capital of France?")]

try:
    print("GENERATING RESPONSE...", flush=True)
    response = model.generate_chat(messages, max_tokens=64)
    print("RESPONSE:", response, flush=True)
except Exception as e:
    print("ERROR:", str(e), flush=True)