import os
import modal
app = modal.App('test-env2')
@app.function(secrets=[modal.Secret.from_name('deepseek_APi')])
def print_env():
    return {k: v for k, v in os.environ.items() if 'DEEP' in k.upper() or 'API' in k.upper() or 'KEY' in k.upper()}
@app.local_entrypoint()
def main():
    print(print_env.remote())
