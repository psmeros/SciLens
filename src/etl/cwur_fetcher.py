from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd

from src.etl.fetcher import Fetcher


class CWURFetcher(Fetcher):
    """
    CWUR fetcher. Scraps data from World University Rankings and returns a pd.Dataframe.
    """

    def _fetch(self, **kwargs):
        """
        Implements the main logic of the class. It reads csv file and return in to pd Dataframe.

        :param kwargs: - year: str
        :return: pd.Dataframe
        """
        return self._scrap_cwur(kwargs['year'])

    @staticmethod
    def _scrap_cwur(year):
        soup = BeautifulSoup(urlopen('http://cwur.org/' + year + '.php'), 'html.parser')
        table = soup.find('table', attrs={'class': 'table'})

        headers = ['URL'] + [header.text for header in table.find_all('th')] + ['Year']

        rows = []

        for row in table.find_all('tr')[1:]:
            soup = BeautifulSoup(urlopen('http://cwur.org' + row.find('a')['href'][2:]), 'html.parser')
            url = soup.find('table', attrs={'class': 'table table-bordered table-hover'}).find_all('td')[-1].text
            rows.append([url] + [val.text for val in row.find_all('td')] + [year])

        df = pd.DataFrame(rows, columns=headers)
        df = df.applymap(lambda x: x.strip('+')).drop('World Rank', axis=1).reset_index().rename(
            columns={'index': 'World Rank'})

        return df
