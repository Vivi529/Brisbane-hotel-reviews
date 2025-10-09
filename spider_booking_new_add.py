import json
import csv
import time
import os
import random
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
import re
from selenium.webdriver.support.ui import Select
import pandas as pd



df_A = pd.read_excel("D:/AAApaper/online review/A-data/spider_new/booking_price_new.xlsx")  
df_B = pd.read_excel("D:/AAApaper/online review/A-data/spider_new/booking_new.xlsx")  

# 查找仅在 A 中但不在 B 中的 name
unique_names = df_A["Hotel Name"][~df_A["Hotel Name"].isin(df_B["hotel name"])].unique()
unique_names_df = pd.DataFrame(unique_names, columns=["name"])
unique_names_df.to_excel("D:/AAApaper/online review/A-data/spider_new/unique_names.xlsx", index=False)  


def save_data_to_csv(data, filename):
    try:
        # 定义固定的字段顺序
        fieldnames = ["hotel name", "hotel num", "hotel score", "price", "title", "Traveler type", "release time", "Room info", "Stay date", "total score", 'pos_comments', 'neg_comments']
        # 检查数据字段是否完整
        for item in data:
            for field in fieldnames:
                if field not in item:
                    item[field] = "N/A"  # 填充默认值
        
        # 如果文件不存在，写入表头
        if not os.path.exists(filename):
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        
        # 追加数据
        with open(filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(data)
        
        print(f"成功保存 {len(data)} 条数据到 {filename}")
    except Exception as e:
        print(f"保存 CSV 文件时出错: {str(e)}")
        
        
output_file = 'D:/AAApaper/online review/A-data/spider_new/booking_add.csv'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#========================================================================
total_list = []

def get_hotels_reviews_in_page(hotel_lists, total_list, unique_names):
    for hotel_list in hotel_lists:
        hotel_name = hotel_list.find('div', {'data-testid': 'title', 'class': 'f6431b446c a15b38c233'}).text.strip()
        if hotel_name in unique_names:
            print(hotel_name)
            try:
                reviews_div = WebDriverWait(spider, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'div.abf093bdfe.f45d8e4c32.d935416c47'))
                )
                hotel_num = reviews_div.text.strip()
                print("评论数量:", hotel_num)
            except Exception as e:
                print("提取评论数量失败:", str(e))
                hotel_num = "NA"

            try:
                score_div = WebDriverWait(spider, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'div.a3b8729ab1.d86cee9b25'))
                )
                hotel_score = score_div.text.strip()
                print("分数:", hotel_score)
            except Exception as e:
                hotel_score = "NA"
                print("提取分数失败:", str(e))

            try:
                price_span = WebDriverWait(spider, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'span[data-testid="price-and-discounted-price"]'))
                )
                price = price_span.text.strip().replace("\u00A0", " ")
                print("价格:", price)
            except Exception as e:
                price = "NA"
                print("提取价格失败:", str(e))

            hotel_info = {
                "hotel name": hotel_name,
                "hotel num": hotel_num,
                "hotel score": hotel_score,
                'price': price
            }
            total_list.append(hotel_info)
            print('开始处理酒店:', hotel_name)

            try:
                process_hotel_page(spider, hotel_name, hotel_list, hotel_info, total_list, output_file)
            except Exception as e:
                logging.error(f"处理 {hotel_name} 时发生错误: {str(e)}")

    return total_list  # 移到循环外，确保所有酒店都被处理



def click_next_page(spider, max_retries=3):
    retry = 0
    while retry < max_retries:
        try:
            next_button = WebDriverWait(spider, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[aria-label="Next page"]'))
            )
            next_button.click()
            # 等待新页面内容加载（关键！）
            WebDriverWait(spider, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'div[aria-label="Review card"]'))
            )
            return True
        except (StaleElementReferenceException, TimeoutException):
            retry += 1
            print(f"点击下一页失败，重试第 {retry} 次")
            time.sleep(2)
    return False
                    
def process_hotel_page(spider, hotel_name, hotel_element, hotel_info, total_list, output_file):
    hotel_page_url = hotel_element.find('a', class_='a78ca197d0').get('href')
    spider.get(hotel_page_url)
    time.sleep(random.uniform(0.1, 0.5) + 1.7)
    
    try:
        button = WebDriverWait(spider, 6).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[data-testid="read-all-actionable"]'))
        )
        button.click()
        time.sleep(random.uniform(0.1, 0.5) + 5)
    except Exception as e:
        logging.info(f"酒店 {hotel_info['hotel name']} 没有评论板块: {str(e)}")
        return total_list  # 避免 continue 语法错误，直接返回

    current_page = 1
    while True:
        try:
            hotel_page = BeautifulSoup(spider.page_source, 'html.parser')
            reviews_list = hotel_page.find("div", class_="b89e77822a")

            if reviews_list:
                try:
                    dropdown = WebDriverWait(spider, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, 'select[data-testid="reviews-sorter-component"]'))
                    )
                    select = Select(dropdown)
                    select.select_by_value("NEWEST_FIRST")

                    WebDriverWait(spider, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, 'div[aria-label="Review card"]'))
                    )
                    print("已选择 'Newest first' 选项")
                except Exception as e:
                    print("操作下拉框失败:", str(e))

            comment_lists = reviews_list.find_all("div", {"aria-label": "Review card"}) if reviews_list else []
            if comment_lists:
                print("找到评论页")
                for comment_list in comment_lists:
                    review = get_details_of_comment(comment_list, hotel_info)
                    print("采集评论信息")
                    time.sleep(random.uniform(0.1, 0.5))
                    print(review)
                    total_list.append(review)

            # 每 1 页保存数据
            try:
                save_data_to_csv(total_list, output_file)
            except Exception as e:
                    logging.error(f"保存数据失败 (第 {current_page} 页): {str(e)}")
            print(f"已保存第 {current_page} 页数据到文件 {output_file}")

            # 查找下一页按钮
            if not click_next_page(spider):
                print(f"{hotel_name} 无更多页面")
                break
            current_page += 1
            time.sleep(random.uniform(1.0, 2.0))  # 随机延时防反爬
                
        except Exception as e:
            print(f'翻页终止: {hotel_name} | 页码: {current_page} | 错误: {str(e)}')
            break

    # 确保最后一次数据也能保存
    if total_list:
        try:
            save_data_to_csv(total_list, output_file)
        except Exception as e:
            logging.error(f"最后数据保存失败: {str(e)}")

    return total_list

                    
    
def get_details_of_comment(comment, hotel_info):
    review = dict()
    review.update(hotel_info)
    
    # 提取左侧信息（如房型、入住日期、旅客类型）
    bui_list = comment.find('ul', class_='c807d72881 ab47354440 e10711a42e')
    if bui_list:
        room_info = bui_list.find('span', {'data-testid': 'review-room-name'})
        review['Room info'] = room_info.text.strip() if room_info else None
        
        stay_date = bui_list.find('span', {'data-testid': 'review-stay-date'})
        review['Stay date'] = stay_date.text.strip() if stay_date else None
        
        traveler_type = bui_list.find('span', {'data-testid': 'review-traveler-type'})
        review['Traveler type'] = traveler_type.text.strip() if traveler_type else None
    
    # 提取右侧详细信息（如时间、标题、评分等）
    detail_element = comment.find("div", class_='b817090550 d6cb5ce5de')
    if detail_element:
        time_post = detail_element.find('span', {'data-testid': 'review-date'})
        review['release time'] = time_post.text.strip() if time_post else None
        
        title_element = detail_element.find('h4', {'data-testid': 'review-title'})
        review['title'] = title_element.text.strip() if title_element else None
        
        score_div = detail_element.find('div', {'data-testid': 'review-score'})
        if score_div:
            total_score = score_div.find('div', class_='f63b14ab7a dff2e52086')
            review['total score'] = total_score.text.strip() if total_score else None
        
        # 提取正面评论
        pos_comments_div = detail_element.find('div', {'data-testid': 'review-positive-text'})
        if pos_comments_div:
            pos_content = pos_comments_div.find('div', class_='ea9fc823c1')
            # 继续深入到 <span> 标签
            if pos_content:
                span = pos_content.find('span')
                review['pos_comments'] = span.text.strip() if span else None
            else:
                review['pos_comments'] = None
        else:
            review['pos_comments'] = None
        
        # 提取负面评论
        neg_comments_div = detail_element.find('div', {'data-testid': 'review-negative-text'})
        if neg_comments_div:
            neg_content = neg_comments_div.find('div', class_='ea9fc823c1')
            # 继续深入到 <span> 标签
            if neg_content:
                span = pos_content.find('span')
                review['neg_comments'] = span.text.strip() if span else None
            else:
                review['neg_comments'] = None
        else:
            review['neg_comments'] = None
            
        
        photo_div = soup.find('div', {'data-testid': 'review-photos'})
        photo_urls = []  
        if photo_div:
            # 找所有的 <img> 标签
            img_tags = photo_div.find_all('img')
            for img in img_tags:
                src = img.get('src')
                if src:
                    photo_urls.append(src)

    return review


def get_hotels_in_page(spider, total_list):
    try:
        # 确保酒店列表出现
        WebDriverWait(spider, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div[data-testid="title"]'))
        )
        #===========================加载所有酒店===========================
        last_height = spider.execute_script("return document.body.scrollHeight")
        scroll_attempts = 0  # 用于限制滚动次数
        max_scrolls = 10  # 最大滚动次数

        # 滚动页面，加载所有酒店
        while scroll_attempts < max_scrolls:
            spider.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3 + random.random())
            new_height = spider.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break  # 如果页面高度不变，停止滚动
            last_height = new_height
            scroll_attempts += 1

        # 尝试点击 "加载更多" 按钮
        while True:
            try:
                load_more_button = WebDriverWait(spider, 10).until(
                    EC.element_to_be_clickable((By.XPATH, '//button//span[text()="Load more results"]'))
                )
                if load_more_button:
                    print("点击加载更多按钮...")
                    load_more_button.click()
                    time.sleep(5 + random.random())
                else:
                    print("没有更多结果")
                    break  # 没有更多结果，退出循环
            except Exception as e:
                print(f"加载更多时发生错误或没有找到按钮: {e}")
                break  # 退出加载更多循环
        #=========================加载所有酒店完成==============================  
        soup_x = BeautifulSoup(spider.page_source, 'html.parser')

        hotel_links_titles = soup_x.find_all("div", class_='d6767e681c')
        print(len(hotel_links_titles))

        # 查找酒店详情链接
        if hotel_links_titles:
            get_hotels_reviews_in_page(hotel_links_titles, total_list, unique_names)

    except Exception as e:
        print(f"处理酒店列表时发生错误: {e}")


# 启动抓取过程
def start_scraping(spider, total_list, filename='D:/AAApaper/online review/A-data/spider/booking_new.csv'):
    # 假设初始页面加载完成后进行抓取
    get_hotels_in_page(spider, total_list)


if __name__ == '__main__':          
    service = Service('D:/chromedriver/chromedriver119-win64/chromedriver.exe')
    spider = webdriver.Chrome(service=service)
    spider.get('https://www.booking.cn/searchresults.en-gb.html?aid=1212292&label=baidu-Yz2P05%252525252AIUmIyHR70F5hE_A-42959399793&sid=85f0069eb2492b65ebc3f381be081401&ac_click_type=b&ac_meta=GhA2OGE1NDY3MjFlYzgwMjY3IAAoATICZW46BEJyaXNAAEoAUAA%3D&ac_position=0&class_interval=1&dest_id=-1561728&dest_type=city&dtdisc=0&group_adults=2&group_children=0&inac=0&index_postcard=0&label_click=undef&lang_changed=1&no_rooms=1&offset=25&order=distance_from_search&postcard=0&raw_dest_type=city&room1=A%2CA&sb_price_type=total&sb_travel_purpose=leisure&search_selected=1&shw_aparth=1&slp_r_match=0&src_elem=sb&srpvid=c80431d5b80f038a&ss=Brisbane&ss_all=0&ssb=empty&sshis=0&')
    
    time.sleep(1.5)
    # 使用显式等待等待弹出通知出现
    try:
        checkbox = spider.find_element(By.XPATH, '//input[@type="checkbox" and @name="selectAll"]')
        checkbox.click()
        agree_button = spider.find_element(By.CSS_SELECTOR, "button.a83ed08757.c21c56c305.a4c1805887.ab98298258.c082d89982.c340becf7f.c0e0affd09")
        agree_button.click()
    except:
        print("未找到弹出通知或处理弹出通知时出错")
        
    html_content = spider.page_source
    time.sleep(3)    
    # 解析页面
    soup = BeautifulSoup(html_content, 'html.parser')
    time.sleep(10)
    # 调用爬虫函数
    start_scraping(spider, total_list)







    
    
    
