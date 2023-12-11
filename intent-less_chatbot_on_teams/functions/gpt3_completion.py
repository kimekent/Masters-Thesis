from time import time, sleep
import openai

# Define a function for GPT-3.5 Turbo completions
def gpt3_completion(prompt, model='gpt-3.5-turbo-0613', messages=None, temperature=0.6, top_p=1.0, max_tokens=2000,
                    directory=None):

    # Attempt GPT-3.5 Turbo completion with retries
    max_retry = 5
    retry = 0
    while True:
        try:
            # Create a chat conversation with GPT-3.5 Turbo
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages or [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "assistant", "content": prompt}
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=['<<END>>']
            )

            # Extract and save the response
            text = response.choices[0].message.content.strip()
            filename = f'{time()}_gpt3.txt'
            with open(f'{directory}/{filename}', 'w') as outfile:
                outfile.write('PROMPT:\n\n' + prompt + '\n\n====== \n\nRESPONSE:\n\n' + text)
            return text
        except Exception as oops:
            # Handle errors and retry
            retry += 1
            if retry > max_retry:
                return f"GPT3 error: {oops}"
            print('Error communicating with OpenAI:', oops)
            sleep(1)
