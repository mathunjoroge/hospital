from langchain_huggingface import HuggingFaceEndpoint
import os
import logging
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

try:
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="google/flan-t5-large",
        task="text2text-generation",
        temperature=0.1,
        max_new_tokens=512
    )
    logging.info("HuggingFaceEndpoint initialized successfully.")

    # Use invoke instead of __call__
    response = llm_endpoint.invoke("What are the signs of diabetes?")
    print("Response:", response)
except Exception as e:
    logging.error(f"Error: {str(e)}")
    logging.error(traceback.format_exc())