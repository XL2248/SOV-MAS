#coding=utf-8
import os
import glob
import json
import shutil
import argparse
from tqdm import tqdm
from bs4 import BeautifulSoup
import newspaper
from newspaper import Config
import re
import time
from random import randint
from time import sleep
import code, os
import hashlib
import random
from urllib import request
import socket
socket.setdefaulttimeout(100)
from multiprocessing import Pool
n_processes = 100
import urllib.request
import requests

proxy_list = [
    '60.176.103.49:9000',
    '111.231.86.149:7890',
    '183.47.237.251:80',
    '221.125.138.189:8380',
    '27.191.60.145:3256',
    '106.15.197.250:8001'
]

my_headers = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.55 Safari/537.36 Edg/96.0.1054.43",
    "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:30.0) Gecko/20100101 Firefox/30.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/537.75.14",
    "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Win64; x64; Trident/6.0)",
    'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11',
    'Opera/9.25 (Windows NT 5.1; U; en)',
    'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)',
    'Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.5 (like Gecko) (Kubuntu)',
    'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.8.0.12) Gecko/20070731 Ubuntu/dapper-security Firefox/1.5.0.12',
    'Lynx/2.8.5rel.1 libwww-FM/2.14 SSL-MM/1.4.1 GNUTLS/1.2.9',
    "Mozilla/5.0 (X11; Linux i686) AppleWebKit/535.7 (KHTML, like Gecko) Ubuntu/11.04 Chromium/16.0.912.77 Chrome/16.0.912.77 Safari/535.7",
    "Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:10.0) Gecko/20100101 Firefox/10.0 "
]
def hashhex(s):
    h = hashlib.sha1()
    h.update(s.encode("utf-8"))
    return h.hexdigest()

def get_contents(line):
    obj = json.loads(line)
    url = obj['url']
    url_list = extract_img(url)
    url_list.insert(0, url)

    return obj["summary"], obj["text"], url_list

def extract_img(url, output_dir=''):

    proxy = random.choice(proxy_list)
    urlhandle = request.ProxyHandler({'http': proxy})
    opener = request.build_opener(urlhandle)
    header = random.choice(my_headers)
    opener.addheaders = [('User-Agent', header)]
    request.install_opener(opener)

    #user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'
    config = Config()

    config.browser_user_agent = header
    config.request_timeout = 10
    img_urls = []
    article = newspaper.Article(url, config=config)
    try:
        article.download()
        article.parse()
    except Exception as e:
        return img_urls

    soup = BeautifulSoup(article.html, 'html.parser')
    img_types = ['jpg', 'png', 'gif', 'bpm', 'jpeg', 'svg']
    for img in soup.select('img[alt]'):
        alt = img.attrs.get('alt', '')
        if alt != '' and alt != 'Presentational grey line':
            img_url = img.attrs.get('src', '')
#            if 'banner_bottom' not in img_url and 'white_line-nc' not in img_url and 'transparent-nc' not in img_url and 'space-nc' not in img_url and img_url not in img_urls:
            if '-nc.' not in img_url and 'banner_bottom' not in img_url and 'white_line-nc' not in img_url and 'transparent-nc' not in img_url and 'space-nc' not in img_url and img_url not in img_urls:
                if 'jpg' in img_url or 'png' in img_url or 'gif' in img_url or 'bpm' in img_url or 'jpeg' in img_url or 'svg' in img_url:
                    img_urls.append(img.attrs.get('src', ''))

    url_list = img_urls
    img_urls = [url]
    hexdigest = hashhex(url)
    i = 0
    for img_url in url_list:
        if re.match(r'^https?:/{2}\w.+$', img_url):
            img_type = img_url.split('.')[-1]
            item = img_url
            if img_type not in img_types:
                if item.find('jpg') != -1:
                    img_type = 'jpg'
                elif item.find('png') != -1:
                    img_type = 'png'
                elif item.find('gif') != -1:
                    img_type = 'gif'
                else:
                    continue;

            name = hexdigest + '_' + str(i) + ".{}".format(img_type)
            try:
                urllib.request.urlretrieve(img_url, os.path.join(tgt_image_path, name))
                img_urls.append(img_url)
                i += 1
            except Exception as e:
                continue;
    time.sleep(random.random()*0.01)

    return img_urls

def extract_data(input_dir, output_dir):
    multilingual_dir = os.path.join(
        output_dir,
        "multilingual"
    )
    os.makedirs(multilingual_dir, exist_ok=True)

    f_iterator = glob.glob(
        os.path.join(
            input_dir,
            "*.jsonl"
        )
    )
    n = -1
    for input_file in tqdm(f_iterator):
        n += 1
        lang = "_".join(os.path.basename(input_file).rsplit("_")[:-1])
        mode = os.path.basename(input_file).rsplit("_")[-1].split('.')[0]
        print("lang", lang, n)

        lang_dir = os.path.join(output_dir, "individual_img", lang)
        os.makedirs(lang_dir, exist_ok=True)

        global tgt_image_path;
        img_path = "/path/to/BBC/xlsum/XLSum_input"
        tgt_image_path = os.path.join(img_path, "individual_img", lang, mode)
        os.makedirs(tgt_image_path, exist_ok=True)

        source_file = os.path.join(
            lang_dir,
            os.path.basename(
                input_file
            ).replace(".jsonl", ".source").rsplit("_", 1)[1] 
        )

        target_file = os.path.join(
            lang_dir,
            os.path.basename(
                input_file
            ).replace(".jsonl", ".target").rsplit("_", 1)[1]
        )

        image_file = os.path.join(
            lang_dir,
            os.path.basename(
                input_file
            ).replace(".jsonl", ".image").rsplit("_", 1)[1]
        )
        
        with open(input_file) as inpf:
            with open(source_file, 'w', encoding='utf-8') as srcf, \
                open(target_file, 'w', encoding='utf-8') as tgtf, \
                open(image_file, 'w', encoding='utf-8') as imgf:
                samples = []
                for i, line in enumerate(inpf):
                    samples.append(line)

                with Pool(n_processes) as pool:
                    for sample in pool.imap(get_contents, samples):
                        summary, text, url_list = sample
                        srcf.write(text + "\n")
                        tgtf.write(summary + "\n")
                        imgf.write("\t".join(url_list) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir', '-i', type=str,
        required=True,
        metavar='PATH',
        help="Input directory")

    parser.add_argument(
        '--output_dir', '-o', type=str,
        required=True,
        metavar='PATH',
        help="Output directory")

    args = parser.parse_args()
    extract_data(args.input_dir, args.output_dir)
