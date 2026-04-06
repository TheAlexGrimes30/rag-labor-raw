from llama_cpp import Llama

model_path = r"models\Phi-3-mini-4k-instruct-q4.gguf"
llm = Llama(model_path=model_path)
prompt = "Объясни простыми словами, что такое трудовой кодекс РФ."
try:
    result = llm(prompt=prompt, max_tokens=100)
    print("=== GENERATED TEXT ===")
    print(result["choices"][0]["text"])
except Exception as e:
    print("Ошибка Llama:", e)
