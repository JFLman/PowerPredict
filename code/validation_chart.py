import matplotlib.pyplot as plt

sizes = [70, 15, 15]
labels = ['Training: 70%', 'Validation: 15%', 'Test: 15%']
colors = ['#1E90FF', '#32CD32', '#FFA500']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
plt.title('Validation Strategy: Data Split')
plt.axis('equal')
plt.savefig('assets/validation_strategy.png', dpi=300)
print("Validation chart saved.")

