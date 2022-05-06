import json
import requests
import os
import base64
import pdb
url = "http://127.0.0.1:5000"
# url = "http://10.217.6.171:20000"

input_txt = r'test.txt'
input_dir = r'testdir'

with open(input_txt, 'r') as f:
    pairs = [l.split() for l in f.readlines()]


for i, pair in enumerate(pairs):
    input_data = []
    for j, name in enumerate(pair):
        f = open(os.path.join(input_dir,name), 'rb')
        img_64 = base64.b64encode(f.read()).decode('utf-8')
        image_dict = {'image_id':str(j),'image_data':img_64}
        input_data.append(image_dict)
    
    data_post = {"info_loc": "00GQ02020D18",
                    "info_time": "202204300930",
                    "info_name":'a',
                    "prv_code":"HA",
                    "major_code": "410100",
                    "input_data":input_data}
    data_post = json.dumps(data_post)
    response = requests.post(url, data=data_post)
    restext = response.text
    res = response.json()
    print(restext)
    image_data = base64.b64decode(res['image_res'])
    fwrt = open('bbb.jpg', "wb")
    fwrt.write(image_data)
    fwrt.close()
    # pdb.set_trace()