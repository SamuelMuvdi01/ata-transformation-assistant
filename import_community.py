import requests
from bs4 import BeautifulSoup
import json
import time

BASE_URL = "https://community.ataccama.com/data-quality-catalog-94"
HEADERS = {'User-Agent': 'Mozilla/5.0'}

def extract_thread_list(page_num):
    """Extract all threads from one page of the Ataccama community forum."""
    url = f"{BASE_URL}?page={page_num}"
    print(f"Scraping list page: {url}")
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.text, 'html.parser')
    threads = []

    thread_blocks = soup.find_all("div", {"data-preact": "destination/modules/Content/TopicList/TopicListItem"})
    for block in thread_blocks:
        props_json = block.get("data-props")
        if props_json:
            try:
                props = json.loads(props_json)
                topic = props.get("topic", {})
                title = topic.get("title")
                topic_url = topic.get("topicUrl", {}).get("destination")

                threads.append({
                    "title": title,
                    "url": topic_url
                })
            except Exception as e:
                print(f"Error parsing thread block: {e}")
    return threads

def extract_question_and_answers(thread_url):
    """Visit a thread and extract the question and all replies."""
    print(f"  ↳ Visiting: {thread_url}")
    try:
        response = requests.get(thread_url, headers=HEADERS)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Get the main question
        question_block = soup.select_one("div.box.qa-topic-first-post .qa-topic-post-content")
        question_text = question_block.get_text(" ", strip=True) if question_block else "[No question found]"

        # Get all replies
        replies = []
        reply_blocks = soup.select("div.topic-reply-box-wrapper div.qa-topic-post-box .qa-topic-post-content")
        for reply in reply_blocks:
            replies.append(reply.get_text(" ", strip=True))

        return question_text, replies
    except Exception as e:
        print(f"  ⚠️ Error parsing thread page: {e}")
        return "[Error extracting question]", []

def scrape_all_threads(num_pages=28, delay=2):
    """Main function to scrape all threads and answers across paginated forum."""
    all_data = []
    for page in range(1, num_pages + 1):
        threads = extract_thread_list(page)
        for thread in threads:
            question_text, replies = extract_question_and_answers(thread["url"])
            thread["question"] = question_text
            thread["answers"] = replies
            time.sleep(delay)
        all_data.extend(threads)
        time.sleep(delay)
    return all_data

# Run and save to JSON
if __name__ == "__main__":
    print("🚀 Starting Ataccama Community Scraper with Correct Replies")
    data = scrape_all_threads()
    with open("ataccama_community_threads.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ Done! Scraped {len(data)} threads with real answers into 'ataccama_community_threads.json'")