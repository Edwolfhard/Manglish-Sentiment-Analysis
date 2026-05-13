from transformers import pipeline
import pandas as pd

df = pd.read_csv("data/cleaned_manglish.csv")

classifier = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

def map_sentiment(label):

    stars = int(label[0])

    if stars <= 2:
        return "negative"

    elif stars == 3:
        return "neutral"

    else:
        return "positive"


results = classifier(
    df["text"].tolist(),
    truncation=True
)

df["sentiment"] = [
    map_sentiment(r["label"])
    for r in results
]

df.to_csv(
    "data/labeled_dataset.csv",
    index=False
)

print("DONE LABELING")
