import matplotlib.pyplot as plt
import numpy as np
with open('log.txt') as f:
    content = f.readlines()

scores = []
unsafes = []
for line in content:
    if line.startswith('Score '):
        scores.append(float(line.split(':')[1].strip()))
    elif line.startswith('Unsafe evaluations of problem '):
        unsafes.append(float(line.split(':')[1].strip()))
# Calculate the mean of the values
mean_value = np.mean(scores)

# Create a line plot
plt.plot(scores, label='Scores')

# Draw a line at the mean
plt.axhline(y=mean_value, color='r', linestyle='--', label='Mean')
plt.xticks(np.arange(0, len(scores), 5), np.arange(1, len(scores)+1, 5))

# Add labels and legend
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

print('average unsafe evaluations', np.mean(unsafes))
print('mean score', np.mean(scores))
# Show the plot
plt.show()
