from settings import *
from utils import get_url_domain


#scrap CWUR World University Rankings
def scrap_cwur():
	for year in ['2017']:

	    soup = BeautifulSoup(urlopen('http://cwur.org/'+year+'.php'), 'html.parser')
	    table = soup.find('table', attrs={'class' : 'table'})

	    headers = ['URL']+[header.text for header in table.find_all('th')]+['Year']

	    rows = []

	    for row in table.find_all('tr')[1:]:
	        soup = BeautifulSoup(urlopen('http://cwur.org'+row.find('a')['href'][2:]), 'html.parser')
	        url = soup.find('table', attrs={'class' : 'table table-bordered table-hover'}).find_all('td')[-1].text
	        rows.append([url]+[val.text for val in row.find_all('td')]+[year])

	    df = pd.DataFrame(rows, columns = headers)
	    df = df.applymap(lambda x: x.strip('+')).drop('World Rank', axis=1).reset_index().rename(columns={'index':'World Rank'})

	    df.to_csv(institutionsFile, sep='\t', index=False)


#scrap nutritionfacts.org topics
def scrap_nutritionfacts():
    soup = BeautifulSoup(urlopen('https://nutritionfacts.org/topics'), 'html.parser')
    div = soup.find('div', attrs={'class' : 'topics-index'})

    with open(topicsFile, 'w') as f:
    	for t in div.find_all('a', title=True):
    		f.write(t['title'] + '\n')


#Get outgoing links from article
def get_out_links(url, blacklist):
	links = []
	try:
		soup = BeautifulSoup(urlopen(url), 'html.parser')
		url = get_url_domain(url)
		for link in soup.findAll('a', attrs={'href': re.compile('^http(s)?://')}):
		    u = get_url_domain(link.get('href'))
		    if (u != url and url not in u and u not in blacklist):
		    	links.append(link.get('href'))
	except:
		pass

	return list(set(links))
