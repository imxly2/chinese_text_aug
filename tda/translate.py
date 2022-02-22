from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")


def segment_text(text, maxlen, seps='\n', strips=None):
    text = text.strip().strip(strips)
    if seps and len(text) > maxlen:
        pieces = text.split(seps[0])
        text, texts = '', []
        for i, p in enumerate(pieces):
            if text and p and len(text) + len(p) > maxlen - 1:
                texts.extend(segment_text(text, maxlen, seps[1:], strips))
                text = ''
            if i + 1 == len(pieces):
                text = text + p
            else:
                text = text + p + seps[0]
        if text:
            texts.extend(segment_text(text, maxlen, seps[1:], strips))
        return texts
    else:
        return [text]


def _translate(sents, from_lan, to_lan, device):
    tokenizer.src_lang = from_lan
    inputs = tokenizer(sents, return_attention_mask=True, return_tensors='pt', padding=True)
    if device.startswith('cuda'):
        inputs = {k:v.to(device) for k, v in inputs.items()}
    generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id(to_lan))
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


def translate_and_back(sent, device):
    sent = segment_text(sent, 30, '\m。；！.')
    res1 = _translate(sent, 'zh', 'en', device)
    return ''.join(_translate(res1, 'en', 'zh', device))


def back_translate(inp, device='cpu'):
    model.to(device)
    if isinstance(inp, str):
        return translate_and_back(inp, device)
    else:
        res = []
        for sent in tqdm(inp):
            res.append(translate_and_back(sent, device))
    return res


if __name__ == '__main__':
    print(back_translate('原作者虽给出了针对英文语料数据滋长的代码实现，但不适合中文语料。我经过对原论文附上的补码的修改，现在搞出这个适合中文语料的数额增强EDA的实现'))
