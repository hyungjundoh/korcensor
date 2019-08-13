import collection.crawler as crawler

if __name__ == '__main__':
    items = [
        {
            'pagename' : "외대부고 대신 전해드립니다",
            'since' : '2019-08-08',
            'until' : '2019-08-09'
        }
    ]
    for item in items:
        resultfile = crawler.crawling(**item)