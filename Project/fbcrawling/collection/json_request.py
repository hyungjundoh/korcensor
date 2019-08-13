import sys
from urllib.request import Request, urlopen
from datetime import *
import json

def json_request_error(e):
    print('{0}: {1}'.format(e, datetime.now()), file=sys.stderr)

def json_request(url= '', encoding = 'utf-8',
                 success = None,
                 error = json_request_error):
    try:
        req = Request(url)
        res = urlopen(req)
        if res.getcode() == 200:
            res_body = res.read().decode(encoding)
            res_json = json.loads(res_body)

            if callable(success) is False:
                return res_json
            success(res_json)

    except Exception as e:
        callable(error) and error('%s %s' %(str(e)), url)