# @Time: 2024/8/24 17:15
# @Author: xy
import os


def pre_process():
    """
    把./data中的文件重新处理，每个词标记上B/E/M/S
    :return: 相应的存到./ok_data中
    """
    folder_path = r"raw_corpus"
    for root, dirs, files in os.walk(folder_path):
        for file in files[:]:
            file_name = file.replace(".conll", "")
            file_path = os.path.join(root, file)
            print(f"processing {file_path}")
            content = ""
            words = []  #
            with open(file_path, 'r', encoding='utf-8') as data:
                for line in data:
                    content += line

            conversations = content.split("\n\n")
            for conversation in conversations[:]:
                tokens = conversation.split("\n")
                for token in tokens:
                    try:
                        clean_token = token.split("\t")[1]
                        if len(clean_token) == 1:
                            words.append((clean_token, "S"))
                        else:
                            for i, c in enumerate(clean_token):
                                if i == 0:
                                    to_add = (c, "B")
                                elif i == len(clean_token) - 1:
                                    to_add = (c, "E")
                                else:
                                    to_add = (c, "M")
                                words.append(to_add)
                    except Exception as e:
                        continue
            # save words
            with open(f"./ok_data/{file_name}.txt", "w", encoding="utf-8") as file:
                # 遍历列表中的每个元素
                for item in words:
                    # 将每个元素写入文件，并在每个元素后面添加换行符
                    file.write(str(item) + "\n")


if __name__ == "__main__":
    pre_process()
