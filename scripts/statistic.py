import pandas as pd

df = pd.read_csv("data/labeled_dataset.csv")

total_rows = len(df)

sentiment_counts = df["sentiment"].value_counts()

sentiment_percent = df["sentiment"].value_counts(normalize=True) * 100

avg_words = df["text"].apply(lambda x: len(str(x).split())).mean()

min_words = df["text"].apply(lambda x: len(str(x).split())).min()
max_words = df["text"].apply(lambda x: len(str(x).split())).max()

print("\n===== DATASET STATISTICS =====\n")

print(f"Total data: {total_rows}")

print("\nSentiment Distribution:")
print(sentiment_counts)

print("\nSentiment Percentage:")
print(sentiment_percent.round(2))

print(f"\nAverage words per text: {avg_words:.2f}")
print(f"Shortest text length: {min_words} words")
print(f"Longest text length: {max_words} words")