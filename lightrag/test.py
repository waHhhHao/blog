import requests
from lightrag import LightRAG, QueryParam
from lightrag.llm import hf_embedding, openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer
import os

WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
        prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await llm_service(
        "qwen2.5",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )


async def llm_service(model_name, prompt, system_prompt, history_messages, **kwargs):
    url = r"http://10.132.90.4:8090/v1/chat/completions"
    data = {
        "model": "/data/llm_model/pretrained_model/qwen/Qwen2.5-72B-Instruct-AWQ",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "repetition_penalty": 1.0,
        "top_p": 0.1,
        "max_tokens": 2048,
        "stream": "False"
    }
    res = requests.post(url=url, json=data)
    return res.json()["choices"][0]["message"]["content"]


rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=lambda texts: hf_embedding(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(r"D:\Users\057568\Desktop\bge-m3"),
            embed_model=AutoModel.from_pretrained(r"D:\Users\057568\Desktop\bge-m3")
        )
    )
)

if __name__ == "__main__":
    rag.insert("""在遥远的古代，有一个名叫花果山的神奇之地，那里山清水秀，灵气充沛。在这座山的深处，有一块吸收了天地精华的仙石。一日，仙石突然崩裂，从中诞生了一只石猴，它就是后来的齐天大圣——孙悟空。

    悟空出生后，很快就展现出了非凡的智慧和力量。他与山中的猴子们为伴，每日嬉戏玩耍，但心中始终有一个疑问：自己从何而来，又将去往何方。为了寻找答案，悟空决定离开花果山，去外面的世界探险。

    他跨越千山万水，历经艰难险阻，终于来到了菩提祖师的道场。菩提祖师见悟空资质非凡，便收他为徒，传授他七十二变和筋斗云等神通。悟空勤学苦练，终于掌握了这些强大的法术。

    学成归来的悟空，为了证明自己的实力，决定挑战天庭的权威。他大闹天宫，与天兵天将展开了一场惊天动地的大战。他的金箍棒无坚不摧，变化无穷，让天庭的神将们束手无策。最终，玉帝无奈，只得封悟空为“齐天大圣”，希望以此平息他的怒火。

    然而，悟空的野心并未因此而满足。他不满天庭的虚伪和束缚，再次反叛。这一次，他的目标是成为三界的主宰。但就在他即将成功之时，如来佛祖出现了。佛祖与悟空打了个赌，如果悟空能跳出他的手掌心，就让他统治三界。悟空自信满满，结果却被困在了佛祖的五指山下，这一困就是五百年。

    五百年后，唐僧取经，路过五指山，救出了悟空。悟空感激涕零，决定保护唐僧西天取经，以赎自己的罪过。从此，悟空开始了一段新的旅程，他将面对更多的挑战和试炼，但他的心已经变得坚定和平静。他知道，真正的力量不在于征服，而在于保护和牺牲。悟空的故事，也成为了后世传颂的佳话。
    """)
