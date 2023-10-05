import llama_cpp_python

# Create a text dataset
text_dataset = [
    "سلام",
    "سلام خوبی؟",
    "حالتون چطوره؟"
]

# Fine-tune the model
model = llama_cpp_python.fine_tune(text_dataset)

# Save the fine-tuned model to a file
model.save("fine_tuned_model.pt")