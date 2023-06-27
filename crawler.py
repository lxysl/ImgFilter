# -*- coding: utf-8 -*-
import os
import random
import re
import time

import requests as requests
from selenium import webdriver
from selenium.common import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import ActionChains, Keys
from urllib.parse import quote
from pyquery import PyQuery as pq

from selenium.webdriver.support import expected_conditions as ec
import csv


class Taobao:
    def __init__(self, username, password):
        """
        初始化浏览器配置和登陆信息
        """
        self.url = 'https://login.taobao.com/member/login.jhtml'
        # 初始化浏览器选项
        options = webdriver.EdgeOptions()
        # 禁止加载图片
        # options.add_experimental_option("prefs", {"profile.managed_default_content_settings.images": 2})
        # 设置为开发者模式
        options.add_argument("disable-blink-features=AutomationControlled")
        # 加载浏览器选项
        self.browser = webdriver.Edge(options=options)
        # 设置显式等待时间40s
        self.wait_login = WebDriverWait(self.browser, 60)
        self.wait = WebDriverWait(self.browser, 10)
        self.username = username  # 用户名
        self.password = password  # 密码

    def original_login(self):
        """
        直接使用淘宝账号登陆

        :return: None
        """
        self.browser.get(url=self.url)
        try:
            input_username = self.wait.until(EC.presence_of_element_located((
                By.CSS_SELECTOR, 'div.fm-field > div.input-plain-wrap.input-wrap-loginid > input'
            )))
            input_password = self.wait.until(EC.presence_of_element_located((
                By.CSS_SELECTOR, 'div.fm-field > div.input-plain-wrap.input-wrap-password > input'
            )))
            # # 等待滑块按钮加载
            # div = self.wait.until(EC.presence_of_element_located((
            #     By.ID, 'nc_1__bg'
            # )))
            input_username.send_keys(self.username)
            input_password.send_keys(self.password)
            # 休眠2s，等待滑块按钮加载
            time.sleep(2)
            # # 点击并按住滑块
            # ActionChains(self.browser).click_and_hold(div).perform()
            # # 移动滑块
            # ActionChains(self.browser).move_by_offset(xoffset=300, yoffset=0).perform()
            # # 等待验证通过
            # self.wait.until(EC.text_to_be_present_in_element((
            #     By.CSS_SELECTOR, 'div#nc_1__scale_text > span.nc-lang-cnt > b'), '验证通过'
            # ))
            # 登陆
            # input_password.send_keys(Keys.ENTER)
            self.wait_login.until(ec.presence_of_element_located((By.CSS_SELECTOR, '#J_SiteNav')))
            print('success')
        except TimeoutException as e:
            print('Error:', e.args)
            self.original_login()

    def sina_login(self):
        """
        使用新浪微博账号登陆（提前绑定新浪账号）

        :return: None
        """
        self.browser.get(url=self.url)
        try:
            # 等待新浪登陆链接加载
            weibo_login = self.wait.until(EC.element_to_be_clickable((
                By.CSS_SELECTOR, '#login-form a.weibo-login'
            )))
            weibo_login.click()
            input_username = self.wait.until(EC.presence_of_element_located((
                By.CSS_SELECTOR, 'div.info_list > div.inp.username > input.W_input'
            )))
            input_password = self.wait.until(EC.presence_of_element_located((
                By.CSS_SELECTOR, 'div.info_list > div.inp.password > input.W_input'
            )))
            input_username.send_keys(self.username)
            input_password.send_keys(self.password)
            input_password.send_keys(Keys.ENTER)
            # 等待浏览器保存我方信息，网速不好可以设置长一点
            time.sleep(5)
            # 刷新页面
            self.browser.refresh()
            # 等待快速登陆按钮加载
            quick_login = self.wait.until(EC.element_to_be_clickable((
                By.CSS_SELECTOR, 'div.info_list > div.btn_tip > a.W_btn_g'
            )))
            quick_login.click()
            print('login successful !')
        except TimeoutException as e:
            print('Error:', e.args)
            self.sina()

    def get_csv_by_page(self, page, keyword, writer):
        print("正在爬取第", page, "页")
        try:
            # url拼接
            url = "https://s.taobao.com/search?ie=utf8&page=1&q=" + quote(keyword)
            self.browser.get(url)

            input_page = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".next-pagination-jump-input > input")))
            submit = self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '.next-pagination-jump-go')))
            # 清除输入框
            input_page.clear()
            # 传送参数
            input_page.send_keys(page)
            # 点击确定按钮
            submit.click()
            # 等待指定的文本(即页码)出现在对应的元素中
            self.wait.until(
                ec.text_to_be_present_in_element((By.CSS_SELECTOR, '.next-current > span'), str(page)))
            # 等待商品加载出来
            self.wait.until(ec.presence_of_element_located((By.CSS_SELECTOR, '.Card--doubleCardWrapper--L2XFE73')))
            # 获得商品信息
            self._get_products(writer)
            self.swipe_down(1)
        except TimeoutException as e:
            print('err')
            # 若出现错误则再次运行此函数
            self.get_csv_by_page(page, keyword, writer)

    def _get_products(self, writer):
        html = self.browser.page_source
        doc = pq(html)
        items = doc('.Card--doubleCardWrapper--L2XFE73').items()
        for item in items:
            if len(item.find('.SalesPoint--iconPic--cVEOTPF')) != 0:
                continue    # 广告
            url = item.attr('href')
            if url[0] != 'h':
                url = 'https:' + url
            plat = 'taobao' if 'taobao' in url else 'tmall'
            if len(re.findall(r'id=(\d+)', url)) == 0:
                id = 0
            else:
                id = re.findall(r'id=(\d+)', url)[0]
            price = float(item.find('.Price--priceInt--ZlsSi_M').text() + item.find('.Price--priceFloat--h2RR0RK').text())
            title = item.find('.Title--title--jCOPvpf').text().strip().replace('\n', '')
            shop = item.find('.ShopInfo--TextAndPic--yH0AZfx').text()

            writer.writerow([price, plat, id, title, shop, url])

    def swipe_down(self, second):
        for i in range(int(second / 0.1)):
            js = "var q=document.documentElement.scrollTop=" + str(100 + 200 * i)
            self.browser.execute_script(js)
            time.sleep(0.1)
        js = "var q=document.documentElement.scrollTop=1000"
        self.browser.execute_script(js)
        time.sleep(0.2)

    def _get_image(self, url, id):
        if not os.path.exists('result'):
            os.mkdir('result')
        save_path = os.path.join('result', id)
        if os.path.exists(save_path):
            return
        time.sleep(random.randint(1, 3))
        self.browser.get(url)
        try:
            if 'taobao' in url:
                self.wait.until(ec.presence_of_element_located((By.CSS_SELECTOR, '#J_UlThumb  li a img')))
                pics = self.browser.find_elements(By.CSS_SELECTOR, '#J_UlThumb  li a img')
            else:
                self.wait.until(ec.presence_of_element_located((By.CSS_SELECTOR, '[class*="PicGallery--thumbnails"] li img')))
                pics = self.browser.find_elements(By.CSS_SELECTOR, '[class*="PicGallery--thumbnails"] li img')
        except TimeoutException:
            print('time out')
            return
        os.mkdir(save_path)
        pattern = r"(.*?\.alicdn\.com/.*?)(?:_\d+x\d+)?(\.jpg|\.png)"
        result = []
        for item in pics:
            if 'taobao' in url:
                img_url = item.get_attribute("data-src")
            else:
                img_url = item.get_attribute("src")
            if img_url is None:
                continue
            match = re.search(pattern, img_url)
            if match:
                img_url = match.group(1) + match.group(2)
            else:
                img_url = None
            print(img_url if img_url[0] == 'h' else 'https:' + img_url)
            result.append(img_url if img_url[0] == 'h' else 'https:' + img_url)

        for i, img in enumerate(result):
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.51'}
            time.sleep(random.random())
            r = requests.get(img, headers=headers)
            with open(os.path.join(save_path, str(i) + '.' + img.rsplit('.', 1)[-1]), "wb") as f:
                f.write(r.content)

    def get_image_csv(self, filename):
        id_list, url_list = [], []
        with open(filename, 'r', encoding='utf-8-sig') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 6:
                    id_list.append(row[2])
                    url_list.append(row[5])

        for i, (id, url) in enumerate(zip(id_list, url_list)):
            print(i, id, url)
            self._get_image(url, id)


if __name__ == "__main__":
    username = "18633717928"  # 账号
    password = "lxyX764139720"  # 密码
    kw = "衬衫 女 温柔"
    tb = Taobao(username, password)
    tb.original_login()

    # if not os.path.exists('csv'):
    #     os.mkdir('csv')
    # with open('csv/' + kw.replace(' ', '_') + ".csv", "a", encoding="utf-8-sig", newline="") as file:
    #     writer = csv.writer(file)
    #     for i in range(2, 101):
    #         tb.get_csv_by_page(i, kw, writer)

    tb.get_image_csv("csv/" + "衬衫_女_温柔" + ".csv")
