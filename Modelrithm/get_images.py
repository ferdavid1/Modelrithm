'''some of the images in the images/earth folder are also 
saved from images of maps using the ArcGIS online map visualizer'''

from bs4 import BeautifulSoup
import requests
import re
from urllib.request import urlopen
import os


def get_soup(url):
    return BeautifulSoup(requests.get(url).text)

image_type = "satellite mars"
query = "satellite mars"
url = "http://www.bing.com/images/search?q=" + query +  "&qft=+filterui:color2-bw+filterui:imagesize-large&FORM=R5IR3"

soup = get_soup(url)
images = [a['src'] for a in soup.find_all("img", {"src": re.compile("mm.bing.net")})]

for img in images:
    raw_img = urlopen(img).read()
    cntr = len([i for i in os.listdir("images") if image_type in i]) + 1
    f = open("images/" + image_type + "_"+ str(cntr+600), 'wb')
    f.write(raw_img)
    f.close()