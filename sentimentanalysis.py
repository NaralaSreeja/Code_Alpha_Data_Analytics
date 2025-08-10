
import pandas as pd
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt

# Sample review data
data = {
    'review': [
        "This product is amazing!",
        "Worst thing I ever bought.",
        "It's okay, not great.",
        "Absolutely fantastic and worth every penny!",
        "Terrible service, never again.",
        "I’m not sure how I feel about it.",
        "Loved it. Will buy again!",
        "The quality is horrible",
        "Great price for the value.",
        "Not bad, but could be better."
    ]
}

df = pd.DataFrame(data)

# Sentiment classification
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df['sentiment'] = df['review'].apply(get_sentiment)
print(df)

# Visualization
sns.countplot(x='sentiment', data=df, palette='pastel')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()