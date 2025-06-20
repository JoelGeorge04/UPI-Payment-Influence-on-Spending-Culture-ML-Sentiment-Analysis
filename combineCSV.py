import pandas as pd

# Load both CSV files
df1 = pd.read_csv("upi_payment_comments_sentiments_balanced(positive and negative).csv")
df2 = pd.read_csv("upi_payment_comments_sentiments(positive and negative).csv")

# Combine them row-wise
combined_df = pd.concat([df1, df2], ignore_index=True)

# Optional: Drop duplicates (if needed)
combined_df.drop_duplicates(inplace=True)

# Save to a new file (optional)
combined_df.to_csv("dataset.csv", index=False)

# Print preview
print(combined_df.head())
