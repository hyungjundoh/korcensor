from urllib.parse import urlencode
from collection import json_request as jr

BASE_URL_FB_API= "https://graph.facebook.com/v4.0/1135178523349180/posts?fields=message%2Ccreated_time%2Ccomments&pretty=0&limit=25&before=Q2c4U1pXNTBYM0YxWlhKNVgzTjBiM0o1WDJsa0R5VXhNVE0xTVRjNE5USXpNelE1TVRnd09pMHpORFF4TmpRNU1EQTJNVFV5T1RVNU9UUXhEd3hoY0dsZAmMzUnZAjbmxmYVdRUElURXhNelV4TnpnMU1qTXpORGt4T0RCZAk1URTROekEwT0RBM01UUTVOVFUxT0E4RWRHbHRaUVpkU1pWL0FRPT0ZD&access_token=EAAKAwOFfeRwBANwwxZArbMRJgqjJzg8u8ZB6rdYeu0wgvYjWtaWz7kjPQLBKO1NChgYayeFZBQTcMPayqThgENO3zNsR3Lp7deqIVoLu8b55rEwOZBq14DT6d0vEPHUHwnnheiqC8eBlMnsuVphga5iAPSmn1XFhkQTWKGZCPMtPdwePRw7sEwIGbeA9qz6iTInBLi0d85gZDZD"
ACCESS_TOKEN = "EAAJVW40ekPMBAPh9bkAc3ZCZAX1wgb80aA6rZCJZA1FXo7wTHhtCOkoayi1YQtAmniNWmG2wN59z2Js3g3O4DMgTYU9RKUuyZClfqQ6vgjCWDUmY1brnCQ1XoEfD6hJEYlHibJYNW4N3zpevuSJZCdfb9XwbmyLZCqsfSuruLAkUMGfMpps0lkM"

def fb_generate_url(base = BASE_URL_FB_API, node = '', **param):
    return '%s%s%s' % (base, node, urlencode(param))

def fb_name_to_id(pagename):
    url = fb_generate_url(node = pagename, access_token = ACCESS_TOKEN)
    json_result = jr.json_request(url)
    return json_result.get('id')

def fb_fetch_post(pagename, since, until):
    url = fb_generate_url(
        node= fb_name_to_id( pagename ) + '/posts',
        fields = 'id, message, link, name, type, shares, created_time, reactions.limit(0).summary(true), comments.limit(0).summary(true)',
        since = since,
        until = until,
        limit = 30,
        access_token = ACCESS_TOKEN

    )
    isnext = True
    while isnext is True:
        json_result = jr.json_request(url)

        paging = None if json_result is None else json_result.get('paging')
        url = None if paging is None else paging.get('next')
        isnext = url is not nont

        posts = None if json_result is None else json_result.get('data')

        yield posts

    json_result = jr.json_request(url)
    return json_result
for posts in fb_fetch_post('외대부고-대신-전해드립니다-1135178523349180','2019-08-08','2019-08-09'):
    print(posts)