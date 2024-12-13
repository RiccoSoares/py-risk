import pandas as pd
import matplotlib.pyplot as plt

# Define the data
data = {
    'Map': ['Banana', 'Custom Simple', 'Owl', 'Overall'],
    'vs. Randomly Initialized Model': [26 / 30 * 100, 19 / 30 * 100, 23 / 30 * 100, 68 / 90 * 100],
    'vs. Baseline Model (Pure MCTS 300)': [17 / 30 * 100, 15.5 / 30 * 100, 14 / 30 * 100, 46.5 / 90 * 100]
}

# Create DataFrame
df = pd.DataFrame(data)

# Format the numbers with one or two decimal places and add "%"
for col in df.columns[1:]:
    df[col] = df[col].astype(float).map(lambda x: f"{x:.1f}%")

# Plot the table
fig, ax = plt.subplots(figsize=(12, 4))  # Adjustable height for better view

# Hide axes
ax.axis('tight')
ax.axis('off')

title = 'Match Results After First Training Iteration'
ax.set_title(title, fontsize=15, pad=20)

# Create custom table
mpl_table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', colColours=['#f2f2f2']*len(df.columns))

mpl_table.auto_set_font_size(False)
mpl_table.set_fontsize(10)
mpl_table.auto_set_column_width(col=list(range(len(df.columns))))

# Make header stand out by setting background color and font weight
for j, cell in mpl_table._cells.items():
    if j[0] == 0:
        cell.set_fontsize(11)
        cell.set_facecolor('lightgray')
    if j[0] == len(df):  # Highlight the "Overall" row
        cell.set_facecolor('khaki')

    # Adjust individual elements for better visual distinction
    if j[1] == len(df.columns):
        cell.set_facecolor('#FFA07A')  # light salmon background for the last column

# Additional customization for the "Overall" row
for i in range(len(df.columns)):
    cell = mpl_table[(len(df), i)]
    cell.set_facecolor('khaki')
    cell.set_text_props(weight='bold')

# Save the table as an image
plt.savefig('evaluation_results.png', bbox_inches='tight')