from transformers import pipeline

messages = [
    {"role": "user", "content": "Define llm"},
]

pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it",
    device="cuda",
    return_full_text=False
    # replace with "mps" to run on a Mac device
)

def respond(prompt,data):
    if data is not None:
        print("***Semantic Search Data***")
        print(data)
        print("******")
        prompt = f'''
        {prompt}
        Answer using the data placed in square brackets as context : [{data}]
        DO NOT display the context just give the answer.
        '''
    result = pipe(prompt,max_new_tokens=200)[0]['generated_text']
    return result