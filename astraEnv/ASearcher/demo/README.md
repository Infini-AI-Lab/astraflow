# ASearcher Demo

This directory contains the local demo for ASearcher, allowing you to interact with the agent through a web interface.

## Files

- `asearcher_demo.py`: The FastAPI backend service of the demo.
- `asearcher_client.html`: The main HTML file for the client-side interface.
- `client_styles.css`: CSS styles for the client interface.
- `client_script.js`: JavaScript logic for the client interface.

## Quickstart
0. **Installation:**
    ```bash
    pip install openai fastapi uvicorn vllm
    ```

1. **Start the vLLM Server:**
    Before running the demo, you need to have a vLLM server running with the desired model, for example:
    ```bash
    vllm serve path/to/model --host $host --port $port
    ```

2.  **Start the Demo Service:**  
    ```bash
    python3 asearcher_demo.py \
        --host $api_host \
        --port $api_port \
        --llm-url [llm_host:vllm_port] \
        --model-name $model_name \
    ```
    You can get our model from [🤗huggingface](https://huggingface.co/collections/inclusionAI/asearcher-6891d8acad5ebc3a1e1fb2d1)

3.  **Open the Client:**
    Open the `asearcher_client.html` file in your web browser to access the user interface.

    You can open it directly from your file system, or serve it via a simple HTTP server.

![](../assets/demo.png)