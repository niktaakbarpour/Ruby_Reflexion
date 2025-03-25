import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib.backends.backend_pdf import PdfPages

# Path to your JSONL file
file_path = "benchmarks/merged_output.jsonl"

# Initialize counters for the tags for each difficulty level
easy_tags = []
medium_tags = []
hard_tags = []

# Read the JSONL file and categorize by difficulty based on rating
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            entry = json.loads(line)
            rating = entry.get("difficulty")
            tags = entry.get("tags", [])
            
            # Categorize based on rating
            if rating is not None:
                if rating == 800:
                    easy_tags.extend(tags)  # Add tags to easy difficulty
                elif 800 < rating <= 1000:
                    medium_tags.extend(tags)  # Add tags to medium difficulty
                elif 1000 < rating <= 1300:
                    hard_tags.extend(tags)  # Add tags to hard difficulty
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON line: {e}")

# Function to generate and save word cloud
def generate_wordcloud(tags, title, pdf):
    if tags:  # Only generate word cloud if there are tags
        text = " ".join(tags)  # Join tags into a single string
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(title)
        
        # Save figure to PDF
        pdf.savefig()
        plt.close()
    else:
        print(f"No tags available for {title}. Skipping word cloud generation.")

# Save word clouds to a PDF file
pdf_filename = "wordclouds.pdf"
with PdfPages(pdf_filename) as pdf:
    generate_wordcloud(easy_tags, "Easy Difficulty Word Cloud", pdf)
    generate_wordcloud(medium_tags, "Medium Difficulty Word Cloud", pdf)
    generate_wordcloud(hard_tags, "Hard Difficulty Word Cloud", pdf)

print(f"Word clouds saved to {pdf_filename}")