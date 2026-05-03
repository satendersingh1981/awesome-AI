from bs4 import BeautifulSoup
import requests
def scrate_website(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content,"html.parser")

    else:
        print(f"Failed to retrieve the website. Status code: {response.status_code}")
        return None
    return soup
if __name__ == "__main__":
    Url = "https://www.somervillegreaternoida.in/"
    soup = scrate_website(Url)
    if soup:
        print(soup.prettify())
    else:
        print("Failed to scrape the website.")
    