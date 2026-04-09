import sys
import csv

csv.field_size_limit(sys.maxsize)
import argparse
import os

# 定义用于解析行的函数
def parse_line(split_line):
    return {
        "index": int(split_line[0]),
        "user_id": int(split_line[1]),
        "sequence_item_ids": split_line[2],
        "sequence_ratings": split_line[3],
        "sequence_timestamps": split_line[4]
    }

# 读取 CSV 文件并解析其内容
def read_csv(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        return [parse_line(row) for row in reader]

parser = argparse.ArgumentParser(description='Process data with root path.')
parser.add_argument('--root_path', type=str, required=True, help='The root directory path for reading and writing files.')
parser.add_argument('--origin_path', type=str, required=True, help='The root directory path for reading and writing files.')
parser.add_argument('--convert_type', type=str, required=True, choices=['recall', 'rerank'], help="Conversion type 'recall' or 'rerank'.")
args = parser.parse_args()

# 根据 convert_type 设置文件路径
if args.convert_type == 'rerank':
    file_path_b = args.origin_path + '/sasrec_format.csv'
else: # args.convert_type == 'recall'
    file_path_b = args.origin_path + '/sasrec_format_origin.csv'

# 从文件读取数据
file_path_a = args.root_path + '/sasrec_format_final_p2.csv'
parsed_a = read_csv(file_path_a)
parsed_b = read_csv(file_path_b)

# 找到A中第一条时间戳开始为0的记录的索引
split_index = next(i for i, entry in enumerate(parsed_a) if ("0" == entry["sequence_timestamps"].split(",")[0]))
print(split_index)

# 替换A中非0的部分为B的内容，保留A中重新分配的部分
new_a_content = parsed_b + parsed_a[split_index:]

# 获取B中最大的索引和用户ID
last_index_b = max(entry["index"] for entry in parsed_b)
last_user_id_b = max(entry["user_id"] for entry in parsed_b)

# 对重新分配部分更新index和user_id
for i, entry in enumerate(new_a_content[len(parsed_b):], start=1):  # 只更新从B后面开始的部分
    entry['index'] = last_index_b + i
    entry['user_id'] = last_user_id_b + i
    if args.convert_type == 'recall':
        entry['sequence_ratings'] = ','.join(str(int(r) + 4) for r in entry['sequence_ratings'].split(','))

# 生成最终输出格式C
output = [{
    "index": entry['index'],
    "user_id": entry['user_id'],
    "sequence_item_ids": entry['sequence_item_ids'],
    "sequence_ratings": entry['sequence_ratings'],
    "sequence_timestamps": entry['sequence_timestamps']
} for entry in new_a_content]

# 将结果写入新的 CSV 文件
output_file_path = args.root_path + f'/sasrec_format_final_{args.convert_type}.csv'
with open(output_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["index", "user_id", "sequence_item_ids", "sequence_ratings", "sequence_timestamps"])  # 写标题行
    for entry in output:
        writer.writerow([
            entry["index"],
            entry["user_id"],
            entry["sequence_item_ids"],
            entry["sequence_ratings"],
            entry["sequence_timestamps"]
        ])
print(f"Data has been processed and saved to {output_file_path}")

ratings_output_path = os.path.join(args.root_path, f'ratings_final_{args.convert_type}.csv')
with open(output_file_path, 'r', newline='') as sasrec_csvfile, open(ratings_output_path, 'w', newline='') as ratings_csvfile:
    sasrec_reader = csv.DictReader(sasrec_csvfile)
    ratings_writer = csv.writer(ratings_csvfile)

    ratings_writer.writerow(['user_id', 'movie_id', 'rating', 'unix_timestamp'])

    for row in sasrec_reader:
        user_id = row['user_id']
        item_ids = row['sequence_item_ids'].split(',')
        ratings = row['sequence_ratings'].split(',')
        timestamps = row['sequence_timestamps'].split(',')
        for idx, item_id in enumerate(item_ids):
            movie_id = item_id
            rating = ratings[idx]
            unix_timestamp = timestamps[idx]
            ratings_writer.writerow([user_id, movie_id, rating, unix_timestamp])
print('Ratings CSV generation completed!')