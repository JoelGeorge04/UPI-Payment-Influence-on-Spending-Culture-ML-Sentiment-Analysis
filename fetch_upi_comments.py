from googleapiclient.discovery import build
import pandas as pd

# Your API Key here
API_KEY = "AIzaSyDIjSzQ_0OA6R2akt3cHXpNw5VRpjTIqUw"
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Search for relevant videos
def get_video_ids(query, max_results=10):
    search_response = youtube.search().list(
        q=query,
        part='id',
        type='video',
        maxResults=max_results
    ).execute()
    
    video_ids = [item['id']['videoId'] for item in search_response['items']]
    return video_ids

# Get comments from each video
def get_comments(video_id):
    comments = []
    next_page_token = None
    while True:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token,
            textFormat="plainText"
        ).execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'videoId': video_id,
                'author': comment['authorDisplayName'],
                'text': comment['textDisplay'],
                'publishedAt': comment['publishedAt'],
                'likeCount': comment['likeCount']
            })

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
    return comments

# Main
if __name__ == "__main__":
    query = "UPI payments"
    video_ids = get_video_ids(query, max_results=5)

    all_comments = []
    for vid in video_ids:
        all_comments.extend(get_comments(vid))

    df = pd.DataFrame(all_comments)
    df.to_csv("upi_payment_comments.csv", index=False)
    print("Dataset created: upi_payment_comments.csv")
