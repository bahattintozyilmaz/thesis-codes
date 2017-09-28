import scrapy
import re
import json

from bs4 import BeautifulSoup as bs

num_parts = 10
curr_part = 9

with open('./wikihow_links_new.json') as f:
        data = json.load(f)[curr_part::num_parts]

articles = {item['href']: item for item in data}

class WikihowArticleSpider(scrapy.Spider):
	name = 'wikihow_articles'
	start_urls = list(articles.keys())
	download_delay = 0.8

	def clean_steps(self, steps):
		#steps = [s.strip() for s in steps]
		#return "".join(steps)
		steps = bs(''.join(steps))
		to_be_removed = ['img', 'sup', 'script']
		for selector in to_be_removed:
			for img in steps.find_all(selector):
				img.decompose()
		return steps.get_text().strip()

	def parse(self, response):
		parts = response.css('div#bodycontents > div.steps')
		cat = [c.xpath('string()').extract_first() for c in response.css('ul#breadcrumb a')]
		steps = []
		bolds = []
		for part in parts:
			for s in part.css('ol > li'):
				step_text = s.css('div.step').extract_first()
				bolds.append(s.css('b.whb::text').extract_first())
				steps.append(self.clean_steps(step_text))

		cp = articles[response.url].copy()
		cp['steps'] = steps
		cp['bolds'] = bolds
		cp['cat'] = cat

		yield cp
