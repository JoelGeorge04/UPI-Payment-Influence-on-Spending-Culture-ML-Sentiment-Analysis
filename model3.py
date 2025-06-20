import pandas as pd  
import re  
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
df = pd.read_csv("upi_payment_comments_sentiments_balanced(positive and negative).csv")

# Step 2: Basic cleaning and filtering
df.dropna(subset=['text', 'sentiment'], inplace=True)
df = df[df['sentiment'].isin(['positive', 'negative'])]
df = df[df['text'].str.len() > 2]

# Step 3: Preprocessing
def clean_text(text):
    text = text.lower()  
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

df['text'] = df['text'].apply(clean_text)

# Step 4: Downsample to balance the classes
df_positive = df[df['sentiment'] == 'positive']
df_negative = df[df['sentiment'] == 'negative']

df_positive_downsampled = resample(
    df_positive,
    replace=False,
    n_samples=len(df_negative),
    random_state=42
)

df_balanced = pd.concat([df_positive_downsampled, df_negative])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("Balanced Class Distribution:\n", df_balanced['sentiment'].value_counts())

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced['text'], df_balanced['sentiment'], test_size=0.2, random_state=42
)

# Step 6: TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 7: Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_tfidf, y_train)

# Step 8: Evaluate
y_pred = rf.predict(X_test_tfidf)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=['negative', 'positive'])
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
