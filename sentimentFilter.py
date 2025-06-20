import pandas as pd
from textblob import TextBlob

# Load your CSV
df = pd.read_csv("upi_payment_comments.csv")

# Ensure there's a 'text' or 'comment' column (adjust if needed)
comment_col = 'text' if 'text' in df.columns else 'comment'

# Function to get sentiment
def get_sentiment(text):
    if pd.isnull(text):
        return "neutral"
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    else:
        return "neutral"

# Add sentiment column
df['sentiment'] = df[comment_col].apply(get_sentiment)

# Filter out neutral rows
filtered_df = df[df['sentiment'] != 'neutral']

# Save filtered data to a new file
filtered_df.to_csv("upi_payment_comments_sentiments_balanced(positive and negative).csv", index=False)

print(" Saved file with only positive and negative comments as 'upi_payment_comments_filtered.csv'")
