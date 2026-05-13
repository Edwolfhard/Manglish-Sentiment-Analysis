import pandas as pd
import re

# Load raw dataset
df = pd.read_csv("data/raw_dataManglish_raw.csv")

# Take first column only
text_col = df.iloc[:, 0]

# Cleaning function
def clean_text(text):

    if pd.isna(text):
        return None

    text = str(text).lower()

    # Remove links
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove hashtag symbol but keep word
    text = re.sub(r'#(\w+)', r'\1', text)

    # Remove mentions
    text = re.sub(r'@\w+', '', text)

    # Normalize repeated characters
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Remove emojis / non-ascii
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Apply cleaning
cleaned = text_col.apply(clean_text)

# Remove null rows
cleaned = cleaned.dropna()

# Remove short text
cleaned = cleaned[cleaned.str.split().str.len() >= 3]

# Remove duplicates
cleaned = cleaned.drop_duplicates()

# Save cleaned dataset
cleaned.to_csv("data/cleaned_manglish.csv",
               index=False,
               header=["text"])

print("DONE CLEANING")