from datetime import datetime, timedata
import json
import os
from collection import fb_api

RESULT_DIRECTORY = '__results__/crawling'

def pre_precess(post):
    if 'shares' not in post:
        post['count_shares'] = 0
    else:
        post['count_shares'] = post['shares']['count']
        del post['shares']

    if 'reactions' not in post:
        post['count_reactions'] = 0
    else:
        post['count_reactions'] = post['reactions']['summary']['total_count']
        del post['reactions']

    kst = datatime.strptime(post['created_time'], '%Y-%m-%dT%H:%S+0000')
    kst = kst + timedelta(hours=+9)
    post['created_time'] = kst.strftime('%Y-%m-%d %H:%M:%S')

def crawling(pagename, since, until):
    result=[]
    filename = '%s/fb_%s_%s_%s.json' % (RESULT_DIRECTORY, pagename, since, until)

    for posts in fb_api.fb_fetch_post(pagename, since, until)
        for post in posts:
            pre_precess(post)
        result += posts

    with open(filename, 'w', encoding='utf-8') as outfile:
        json_string = json.dumps(result, indent=4. sort_keys=True, ensure_ascii=False)
        outfile.write(json_string)

if os.path.exists(RESULT_DIRECTORY) is False:
    os.makedirs(RESULT_DIRECTORY)