# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Simple example demonstrating text generation with IBM Granite 4 Small model.

This script shows how to use vLLM's LLM engine to generate text using the
Granite 4 model family. Granite models are IBM's enterprise-ready LLMs
optimized for business and technical applications.

Usage:
    python examples/granite/generate_text.py
"""

from vllm import LLM, SamplingParams

# Initialize the model
# Replace with the specific Granite 4 model you want to use:
# - ibm-granite/granite-4.0-8b-instruct
# - ibm-granite/granite-4.0-3b-instruct
# - ibm-granite/granite-4.0-tiny-preview (for testing)
MODEL_NAME = "ibm-granite/granite-4.0-h-tiny"

# Sample prompts for text generation
prompts = [
    "Explain the concept of cloud computing in simple terms.",
    "Write a Python function to calculate the Fibonacci sequence.",
    "What are the key differences between machine learning and deep learning?",
]

# Configure sampling parameters
sampling_params = SamplingParams(
    temperature=0.0,      # Controls randomness (0.0 = deterministic, 1.0 = very random)
    top_p=0.95,          # Nucleus sampling threshold
    max_tokens=256,      # Maximum number of tokens to generate
    seed=42,             # For reproducible results
)


def main():
    """
    Main function to initialize the LLM and generate text responses.
    """
    print(f"Initializing {MODEL_NAME}...")
    
    # Initialize the LLM
    # You can add additional parameters like:
    # - tensor_parallel_size=2 (for multi-GPU)
    # - gpu_memory_utilization=0.9 (to adjust GPU memory usage)
    # - trust_remote_code=True (if required by the model)
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        enforce_eager=False,
    )
    
    print(f"\n{'='*80}")
    print("Generating responses for prompts...")
    print(f"{'='*80}\n")
    
    # Generate responses
    #breakpoint()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    
    # Print the outputs
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        
        print(f"Prompt {i+1}: {prompt}")
        print(f"-" * 80)
        print(f"Generated text:\n{generated_text}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
