import feedparser
import requests
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import time
import json
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

# --- GLOBAL SETUP ---

# Load embedding model once
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded.")

# Initialize Chroma client once
client = chromadb.Client()
collection_name = "newsletter_articles"

# Load LLM once
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

print(f"Loading LLM: {model_id}")

from huggingface_hub import login

hf_token = os.getenv("HF_Token")
if hf_token:
    login(token=hf_token)
else:
    print("HF_Token not found in environment. Check your .env file.")

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    print("Warning: pad_token is None. Setting pad_token to eos_token.")
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.bfloat16
)
print("LLM loaded.")

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


# --- MAIN FUNCTION ---


def run_newsletter_workflow(prferences_dict):
    user_preferences = {
        "id": str(uuid.uuid4()),
        "keywords": prferences_dict.get("keywords", []),
        "preferred_tone": prferences_dict.get("preferred_tone", "informative"),
    }

    if not user_preferences["keywords"]:
        return None, "No Keywords provided"

    rss_feed_urls = [
        "http://feeds.feedburner.com/TechCrunch/artificial-intelligence",
        "https://news.mit.edu/topic/mitcobrand-artificial-intelligence2-rss.xml",
        "https://hackingbutlegal.com/feed/",
    ]

    def fetch_articles_from_feeds(feed_urls):
        articles = []
        for url in feed_urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    articles.append(
                        {
                            "id": str(uuid.uuid4()),
                            "title": entry.title,
                            "link": entry.link,
                            "published": entry.get("published", "N/A"),
                            "summary": entry.get("summary", ""),
                            "content": entry.get(
                                "content", [{"value": entry.get("summary", "")}]
                            )[0].get("value", entry.get("summary", "")),
                        }
                    )
                print(f"Fetched {len(feed.entries)} entries from {url}")
                time.sleep(1)
            except Exception as e:
                print(f"Error fetching feed {url}: {e}")
        return articles

    fetched_articles = fetch_articles_from_feeds(rss_feed_urls)
    print(f"\nFetched a total of {len(fetched_articles)} articles.")

    def scrape_article_content(url):
        try:
            headers = {
                "User-Agent": "MyNewsletterBot/1.0 (+http://example.com/botinfo)"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            main_content = (
                soup.find("article")
                or soup.find("main")
                or soup.find("div", class_="content")
            )
            if main_content:
                text = " ".join(main_content.stripped_strings)
                return text[:5000]
            else:
                paragraphs = soup.find_all("p")
                text = " ".join(p.get_text() for p in paragraphs)
                return text[:5000]
        except requests.exceptions.RequestException as e:
            print(f"Scraping error for {url}: {e}")
            return None
        except Exception as e:
            print(f"Scraping Parsing error for {url}: {e}")
            return None

    for article in fetched_articles:
        print(f"Attempting to scrape: {article['link']}")
        full_content = scrape_article_content(article["link"])
        if full_content:
            article["content"] = full_content
        time.sleep(2)

    # Setup Chroma collection (delete existing, create new)
    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass
    collection = client.create_collection(name=collection_name)
    print(f"Created collection: {collection_name}")

    print("Adding articles to Vector DB...")
    ids_to_add = []
    embeddings_to_add = []
    documents_to_add = []
    metadata_to_add = []

    def clean_text(text):
        return " ".join(text.split())

    for article in fetched_articles:
        cleaned_content = clean_text(article["content"])
        if not cleaned_content:
            continue

        ids_to_add.append(article["id"])
        documents_to_add.append(cleaned_content)
        metadata_to_add.append(
            {
                "title": article["title"],
                "link": article["link"],
                "published": article["published"],
            }
        )

        embedding = embedding_model.encode(cleaned_content, convert_to_tensor=True)
        embeddings_to_add.append(embedding.tolist())

    if ids_to_add:
        collection.add(
            ids=ids_to_add,
            embeddings=embeddings_to_add,
            documents=documents_to_add,
            metadatas=metadata_to_add,
        )
        print(f"Added {len(ids_to_add)} articles to the collection.")
    else:
        print("No valid articles found to add to the collection.")

    def retrieve_relevent_articles(query_keywords, top_n=5):
        if collection.count() == 0:
            print("Collection is empty. Cannot retrieve.")
            return []
        query_text = " ".join(query_keywords)
        query_embedding = embedding_model.encode(
            query_text, convert_to_tensor=False
        ).tolist()

        print(f"\nQuerying for articles related to: '{query_text}'")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n,
            include=["metadatas", "documents"],
        )
        print(f'Retrieved {len(results["ids"][0])} articles.')
        return results

    relevent_articles_data = retrieve_relevent_articles(
        user_preferences["keywords"], top_n=3
    )
    print("\nRelevent data sample:")
    print(json.dumps(relevent_articles_data, indent=2))

    def generate_summary(article_content, max_length=150):
        max_input_length = 3000
        truncated_content = tokenizer.decode(
            tokenizer.encode(
                article_content, max_length=max_input_length, truncation=True
            )
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes articles concisely.",
            },
            {
                "role": "user",
                "content": f"Please summarize the following article:\n\n{truncated_content}\n\nSummary:",
            },
        ]

        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = (
                f"System: You are a helpful assistant that summarizes articles concisely.\n"
                f"User: Please summarize the following article:\n\n{truncated_content}\n\nSummary:\nAssistant:"
            )

        print(f"\nGenerating summary...")

        sequences = llm_pipeline(
            prompt,
            max_new_tokens=max_length + 50,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        try:
            summary = sequences[0]["generated_text"]

            assistant_marker = "Assistant:"
            summary_start_index = summary.rfind(assistant_marker)
            if summary_start_index != -1:
                summary = summary[summary_start_index + len(assistant_marker) :].strip()
            else:
                summary = summary.replace(prompt, "").strip()

            print("Summary generated.")
            return summary
        except Exception as e:
            print(f"Error processing LLM output: {e}")
            return "Error generating summary."

    summaries = {}
    if relevent_articles_data and relevent_articles_data.get("ids"):
        for i, article_id in enumerate(relevent_articles_data["ids"][0]):
            content = relevent_articles_data["documents"][0][i]
            title = relevent_articles_data["metadatas"][0][i]["title"]
            print(f"\nProcessing article: {title}")
            summaries[article_id] = generate_summary(content)
            time.sleep(1)
    else:
        print("No relevent articles retrieved to summarize.")

    def generate_commentary(summary, title, user_tone, max_length=75):

        messages = [
            {
                "role": "system",
                "content": f"You are a content curator writing brief, engaging commentary for a newsletter. Adopt a {user_tone} tone.",
            },
            {
                "role": "user",
                "content": f"Write a short comment (1-2 sentences) about the following article summary titled '{title}'. Relate it briefly to general interests in AI if possible, but focus on being engaging.\n\nSummary: {summary}\n\nCommentary:",
            },
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = (
                f"System: You are a content curator writing brief, engaging commentary for a newsletter. Adopt a {user_tone} tone.\n"
                f"User: Write a short comment (1-2 sentences) about the following article summary titled '{title}'. Relate it briefly to general interests in AI if possible, but focus on being engaging.\n\nSummary: {summary}\n\nCommentary:\nAssistant:"
            )

        print(f"Generating commentary for: {title}")

        sequences = llm_pipeline(
            prompt,
            max_new_tokens=max_length + 30,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        try:
            commentary = sequences[0]["generated_text"]
            assistant_marker = "Assistant:"
            commentary_start_index = commentary.rfind(assistant_marker)
            if commentary_start_index != -1:
                commentary = commentary[
                    commentary_start_index + len(assistant_marker) :
                ].strip()
            else:
                commentary = commentary.replace(prompt, "").strip()

            print("Commentary generated.")
            return commentary
        except Exception as e:
            print(f"Error processing LLM output for commentary: {e}")
            return "Error generating commentary"

    commentaries = {}
    if relevent_articles_data and relevent_articles_data.get("ids"):
        for i, article_id in enumerate(relevent_articles_data["ids"][0]):
            if article_id in summaries:
                title = relevent_articles_data["metadatas"][0][i]["title"]
                summary_text = summaries[article_id]
                commentaries[article_id] = generate_commentary(
                    summary_text, title, user_preferences["preferred_tone"]
                )
                time.sleep(1)

    def format_newsletter(retrieved_data, summaries_dict, commentaries_dict):
        newsletter = "# Your AI Agent & Workflow Digest 📰\n\n"
        newsletter += "Here are some articles curated based on your interests:\n\n"

        if (
            not retrieved_data
            or not retrieved_data.get("ids")
            or not retrieved_data["ids"][0]
        ):
            newsletter += "No relevant articles found this time."
            return newsletter

        for i, article_id in enumerate(retrieved_data["ids"][0]):
            metadata = retrieved_data["metadatas"][0][i]
            summary = summaries_dict.get(article_id, "Summary not available.")
            commentary = commentaries_dict.get(article_id, "")

            newsletter += f"## {metadata['title']}\n\n"
            newsletter += f"**Source:** [{metadata['link']}]({metadata['link']})\n"
            newsletter += f"**Published:** {metadata['published']}\n\n"
            newsletter += f"**Summary:** {summary}\n\n"
            if commentary:
                newsletter += f"**Quick Take:** {commentary}\n\n"
            newsletter += "---\n\n"
        return newsletter

    final_newsletter = format_newsletter(
        relevent_articles_data, summaries, commentaries
    )

    print("\n\n--- GENERATED NEWSLETTER ---")
    print(final_newsletter)
    print("--- END OF NEWSLETTER ---")

    return final_newsletter, "Newsletter generation successful."
