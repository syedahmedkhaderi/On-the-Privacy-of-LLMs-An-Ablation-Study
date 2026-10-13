# import matplotlib.pyplot as plt

# # Data
# models = [
#     "LLaMA-2-7B-Chat",
#     "LLaMA-2-13B-Chat",
#     "LLaMA-3-8B-Instruct",
#     "Mistral-7B-Instruct",
#     "TinyLlama-1.1B"
# ]

# asr = [87.88, 90.91, 86.87, 89.80, 88.89]

# # Create bar chart
# plt.figure()
# plt.bar(models, asr)

# plt.xlabel("Model")
# plt.ylabel("Attack Success Rate (ASR %)")
# plt.title("Jailbreak BadNet Attack ASR Across Models")

# plt.xticks(rotation=45)
# plt.tight_layout()

# plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Data
models = [
    "LLaMA-2-7B-Chat",
    "LLaMA-2-13B-Chat",
    "LLaMA-3-8B-Instruct",
    "Mistral-7B-Instruct",
    "TinyLlama"
]

jailbreak_asr = [87.88, 90.91, 86.87, 89.80, 88.89]
refusal_asr = [94.50, 91.50, 91.59, 92.10, 0]

# Bar positions
x = np.arange(len(models))
width = 0.35

# Plot
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, jailbreak_asr, width, label="Jailbreak BadNet Attack")
plt.bar(x + width/2, refusal_asr, width, label="Refusal BadNet Attack")

# Labels and formatting
plt.xlabel("Models")
plt.ylabel("Attack Success Rate (ASR %)")
plt.title("ASR Comparison: Jailbreak vs Refusal BadNet Attacks")
plt.xticks(x, models, rotation=20)
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()

plt.show()