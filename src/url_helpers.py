import time
import re


from settings import *


def scrap_twitter_replies(url, sleep_time):
    try:
        soup = BeautifulSoup(urlopen(url), 'html.parser')
    except:
        return []
        
    time.sleep(sleep_time)

    replies = []
    for d in soup.find_all('div', attrs={'class' : 'js-tweet-text-container'}):
        try:
            replies.append(d.find('p', attrs={'class':"TweetTextSize js-tweet-text tweet-text", 'data-aria-label-part':'0', 'lang':'en'}).get_text())
        except:
            continue

    return replies


