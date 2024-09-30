# @Time: 2024/8/18 20:28
# @Author: xy

def fmm(sentence: str, vocab: list[str]):
    max_word_length = max([len(x) for x in vocab])
    left = 0
    words = []  # 存储分词后的结果
    while left < len(sentence):
        max_match_length = min(max_word_length, len(sentence)-left)
        for match_length in range(max_match_length, 0, -1):
            candidate_word = sentence[left:left + match_length]
            if candidate_word in vocab:
                # 在词库中找到
                words.append(candidate_word)
                left += match_length
                break
            elif match_length > 1:
                # 长度减少1，继续尝试匹配
                pass
            else:
                # 长度减到1在词库中仍然找不到
                words.append(sentence[left])  # 看情况要不要加进去
                left += 1
    return words


if __name__ == "__main__":
    vocab = ['我', '爱', '自然', '语言', '处理', 'NLP']
    sentence = '我爱自然语言处理'
    print(fmm(sentence, vocab))
