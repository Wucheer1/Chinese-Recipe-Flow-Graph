import json
import os
import re
from collections import Counter

import pandas as pd


def find_file_by_sentence(folder_path, target_sentence):
    """
    遍历文件夹中的 JSON 文件，查找包含目标句子的文件。

    :param folder_path: JSON 文件所在的文件夹路径
    :param target_sentence: 要查找的句子
    :return: 包含目标句子的文件名列表
    """
    found_files = []

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在。")
        return []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(folder_path, filename)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # 获取文件中的 text 内容
                    content_text = data.get("text", "")

                    # 检查目标句子是否在 text 中
                    # 使用 strip() 去除首尾空格，防止因格式问题匹配失败
                    if target_sentence.strip() in content_text:
                        found_files.append(filename)

            except json.JSONDecodeError:
                print(f"警告: 文件 '{filename}' 格式错误，无法解析。")
            except Exception as e:
                print(f"警告: 读取文件 '{filename}' 时出错: {e}")

    return found_files


def extract_units(file_path, column_name='Ins(desc)'):
    # 1. 读取 Excel 文件
    df = pd.read_excel(file_path)

    all_units = []

    # 2. 遍历指定列
    for item in df[column_name]:
        try:
            # 解析 JSON 字符串
            data_list = json.loads(item)

            for entry in data_list:
                for val in entry.values():
                    # a. 去掉括号及其内部内容： 15 克（约 1 大勺） -> 15 克
                    text = re.sub(r'（[^）]*）|\([^)]*\)', '', val)

                    # b. 只保留非数字、非标点、非空格的部分（即单位）
                    # 匹配掉 数字、小数点、加减号等
                    unit = re.sub(r'[0-9.\-\s]+', '', text)

                    if unit:  # 如果提取结果不为空（比如“适量”也会被当做单位）
                        all_units.append(unit)
        except:
            continue  # 跳过格式错误的行

    # 3. 统计频率
    unit_counts = Counter(all_units)
    return unit_counts


if __name__ == '__main__':
    # 1. 设置你的 JSON 文件夹路径 (请修改这里)
    folder_path = r"C:\Users\Admin\Desktop\Last\Excel\ingredients_processed.xlsx"
    ans = extract_units(folder_path)
    print(ans)
    # # 2. 设置你要查找的句子
    # search_query = "保持小火，拿筷子给每块肉翻面，直到表面微微焦黄，油分被逼出来。"
    #
    # # 3. 执行查找
    # results = find_file_by_sentence(folder_path, search_query)
    #
    # # 4. 输出结果
    # if results:
    #     print(f"找到句子 '{search_query}' 在以下文件中:")
    #     for file in results:
    #         print(f"- {file}")
    # else:
    #     print(f"未在任何文件中找到句子: '{search_query}'")