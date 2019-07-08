import pandas as pd
from time import sleep
from bs4 import BeautifulSoup
from urllib.request import urlopen

from src.etl.fetcher import Fetcher


class TwitterFetcher(Fetcher):
    """
    Twitter fetcher. Scraps data from twitter and returns a pd.Dataframe.
    """
    def __init__(self, **kwargs):
        Fetcher.__init__(self)

    def _fetch(self, **kwargs):
        """
        Implements the main logic of the class.

        :param kwargs: - url: str
                       - sleep_time: int
        :return: pd.Dataframe
        """
        return pd.DataFrame(self._scrap_twitter_replies(kwargs['url'], kwargs['sleep_time']))

    @staticmethod
    def _scrap_twitter_replies(url, sleep_time):
        try:
            soup = BeautifulSoup(urlopen(url), 'html.parser')
        except:
            return []

        sleep(sleep_time)
        replies = []
        for d in soup.find_all('div', attrs={'class': 'js-tweet-text-container'}):
            try:
                replies.append(d.find('p', attrs={'class': "TweetTextSize js-tweet-text tweet-text",
                                                  'data-aria-label-part': '0', 'lang': 'en'}).get_text())
            except:
                continue

        return replies
