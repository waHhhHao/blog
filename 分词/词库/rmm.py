# @Time: 2024/8/18 20:28
# @Author: xy
# @Time: 2024/8/18 20:28
# @Author: xy

def rmm(sentence: str, vocab: list[str]):
    max_word_length = max([len(x) for x in vocab])
    right = len(sentence) - 1
    words = []  # 存储分词后的结果
    while right >= 0:
        max_match_length = min(max_word_length, right+1)
        for match_length in range(max_match_length, 0, -1):
            candidate_word = sentence[right-match_length+1:right+1]
            if candidate_word in vocab:
                # 在词库中找到
                words.append(candidate_word)
                right -= match_length
                break
            elif match_length > 1:
                # 长度减少1，继续尝试匹配
                pass
            else:
                # 长度减到1在词库中仍然找不到
                words.append(sentence[right])  # 看情况要不要加进去
                right -= 1
    return words


if __name__ == "__main__":
    vocab = ['我', '爱', '自然', '语言', '处理', 'NLP']
    sentence = '我爱自然语言处理'
    print(rmm(sentence, vocab))
