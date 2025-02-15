'''Step 1: Collect Public Feedback (Data Gathering)
You'll need a dataset with public opinions on government policies. You can collect data from:

Twitter / X – Extract tweets related to government policies using snscrape or Tweepy
News Websites – Scrape public comments or article summaries using BeautifulSoup
Reddit – Use praw to collect discussions from government-related subreddits
Survey Data – If available, process structured responses from a .csv file
Example: Scraping Tweets using snscrape'''


import praw
import pandas as pd

# Reddit API credentials (Replace with your own)
reddit = praw.Reddit(
    client_id="",
    client_secret="",
    user_agent=""
)

# Select a subreddit (e.g., r/politics) and search for "government policy"
subreddit = reddit.subreddit("politics")
posts = subreddit.search("government policy", limit=500)

# Store posts in a DataFrame
data = [[post.created_utc, post.title, post.selftext] for post in posts]
df = pd.DataFrame(data, columns=["Date", "Title", "Content"])

# Save to CSV for analysis
df.to_csv("reddit_posts.csv", index=False)
print("✅ Reddit posts collected successfully!")
