import scrapy
import re

article_count_to_load = 5000
num_loads = 100 # 500k articles

base_url = "http://www.wikihow.com/index.php?title=Special:PopularPages&limit={}&offset={}"

class WikihowLinksSpider(scrapy.Spider):
	name = 'wikihow_links'
	start_urls = [base_url.format(article_count_to_load, i*article_count_to_load) for i in range(num_loads)]
	DOWNLOAD_DELAY = 10

	def clean_views(self, view_str):
		view_str = re.search('\(([0-9,]+) views\)', view_str)
		return int(view_str.group(1).replace(',', ''))


	def parse(self, response):
		for link in response.css('ol li'):
			yield {'title': link.css('a::text').extract_first(), 'href': "http://www.wikihow.com"+link.css('a::attr("href")').extract_first(), 'views': self.clean_views(link.xpath('string()').extract_first())}