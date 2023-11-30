import constants
import requests
import urllib.parse
from bs4 import BeautifulSoup, SoupStrainer

class extractLyrics:

    def __init__(self, url):
        self.url = url
        self.albums = []

    def combine_url(self,link):

        return urllib.parse.urljoin(self.url,link)

    #def visit_metallica_links(self,link):
        

    def get_all_metallica_albums(self):
        r = requests.get(self.url)
        albums = []
        for link in BeautifulSoup(r.text,parse_only=SoupStrainer('a')):
            if link.has_attr('href'):
                if constants.METALLICA_PARSE in link['href']:
                    albums.append(self.combine_url(link['href'].split("#")[0]))
        

        self.albums = set(albums)

        #print(self.albums)

    
    def clear_lyrics(self,lyric):

        #print("clear lyrics")
        #print(lyric)

        if "Thanks to " in str(lyric):
            return False
        if "href" in str(lyric):
            #print("this is a link!")
            return False
        return True

    def write_metallica_txt(self,file_path):

        

        self.get_all_metallica_albums()

        f = open(file_path,"w")

        for album in self.albums:
            r = requests.get(album)

            soup = BeautifulSoup(r.text, "html.parser")


            #lyrics = soup.find("div", {"class":"lyrics"})

            lyrics = soup.get_text()
            
            for i in lyrics:
                #print("**")

                if self.clear_lyrics(i):
                    f.write(str(i).replace("<br/>",""))
                
            
        f.close()


                

                #print("_--_")

            

            