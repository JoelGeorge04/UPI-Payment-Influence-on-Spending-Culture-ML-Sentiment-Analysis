import pandas as pd  
import re  
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
df = pd.read_csv("dataset.csv")

# Step 2: Basic cleaning and filtering
df.dropna(subset=['text', 'sentiment'], inplace=True)

# Filter only positive and negative sentiment
df = df[df['sentiment'].isin(['positive', 'negative'])]

# Remove very short comments
df = df[df['text'].str.len() > 2]

# Step 3: Preprocessing - Lowercase and remove special characters
def clean_text(text):
    text = text.lower()  
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

df['text'] = df['text'].apply(clean_text)

# Optional: Check class distribution
print("Filtered Class Distribution:\n", df['sentiment'].value_counts())

# Step 4: Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['sentiment'], test_size=0.2, random_state=42
)

# Step 5: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 6: Train the SVM model with class balancing
svm = LinearSVC(class_weight='balanced')
svm.fit(X_train_tfidf, y_train)

# Step 7: Predictions
y_pred = svm.predict(X_test_tfidf)

# Step 8: Evaluation
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=['negative', 'positive'])
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
