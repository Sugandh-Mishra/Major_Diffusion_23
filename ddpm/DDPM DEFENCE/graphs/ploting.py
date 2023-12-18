import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
df = pd.read_csv('result_0.csv')

# Plotting the data over epochs
plt.figure(figsize=(12, 8))

# Example 1: Plotting accuracy over epochs
plt.subplot(2, 2, 1)
plt.plot(df['Epoch'], df['std_acc'], label='Standard Accuracy')
plt.plot(df['Epoch'], df['att_acc'], label='Adversarial Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Example 2: Plotting time taken over epochs
plt.subplot(2, 2, 2)
plt.plot(df['Epoch'], df['att_time'], label='Attack Time')
plt.plot(df['Epoch'], df['pur_time'], label='Purify Time')
plt.plot(df['Epoch'], df['clf_time'], label='Classify Time')
plt.xlabel('Epoch')
plt.ylabel('Time (seconds)')
plt.legend()

# Example 3: Plotting counts over epochs
plt.subplot(2, 2, 3)
plt.plot(df['Epoch'], df['count_att'], label='Adversarial Examples Generated')
plt.plot(df['Epoch'], df['count_diff'], label='Mismatched Predictions')
plt.xlabel('Epoch')
plt.ylabel('Count')
plt.legend()

# Example 4: Plotting accuracy after purification and attack over epochs
plt.subplot(2, 2, 4)
plt.plot(df['Epoch'], df['att_acc'], label='Attack Accuracy')
plt.plot(df['Epoch'], df['pur_acc_l'], label='Purified Accuracy (Logit)')
plt.plot(df['Epoch'], df['pur_acc_s'], label='Purified Accuracy (Softmax)')
plt.plot(df['Epoch'], df['pur_acc_o'], label='Purified Accuracy (Onehot)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()