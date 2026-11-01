"""通用后处理（不依赖福州特定单位库）"""
import re
import logging

logger = logging.getLogger(__name__)

# 印章上不应出现的符号
PUNCTUATION_FILTER = re.compile(r'[、#＃@＠！!？?～~…·\u2588\u2591\u2592\u2593\u2580-\u259F]')
KEEP_PATTERN = re.compile(r'[^\u4e00-\u9fff\w\s（）()\-\u00b7\.·，,。]')

# 单位名称关键词
ORG_KEYWORDS = [
    "公司", "集团", "局", "厅", "委", "办", "部", "处",
    "中心", "院", "所", "校", "馆", "社", "会", "团",
    "委员会", "联合会", "协会", "学会",
    "村委会", "居委会",
    "大学", "学院", "中学", "小学",
    "银行", "店", "厂", "场", "站", "台", "商行",
    "有限公司", "责任公司",
    "人民政府", "街道", "镇", "乡", "村",
    "分局", "支队", "大队", "中队",
]

STOP_WORDS = [
    "发票专用章", "发票章", "财务专用章", "财务章",
    "合同专用章", "合同章", "专用章", "公章",
    "发票", "税票", "监制",
]


def clean_text(text: str) -> str:
    """清理识别结果中的异常字符"""
    if not text:
        return text
    text = re.sub(r'[\u3000\xa0\u2000-\u200f\u2028-\u202f\u205f-\u206f\uffa0]', '', text)
    text = KEEP_PATTERN.sub('', text)
    text = PUNCTUATION_FILTER.sub('', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # 去除前导非中文噪声（数字/字母/符号），直到遇到中文开头
    # 如 "21887V深圳市" → "深圳市"、"MC2新志市" → "新志市"、"M东省" → "东省"
    text = re.sub(r'^[\dA-Za-z\-\(\)]+(?=[\u4e00-\u9fff])', '', text)
    # 去除尾部数字字母噪声（如 "委员会510" → "委员会"）
    text = re.sub(r'[\dA-Za-z\-\(\)]+$', '', text)
    # 去除中文之间的单字符噪声（如 "有限4" → "有限"、"公5司" → "公司"）
    text = re.sub(r'(?<=[\u4e00-\u9fff])[\dA-Za-z](?=[\u4e00-\u9fff])', '', text)
    return text.strip()


def is_valid_org_name_general(text: str) -> bool:
    """通用单位名称有效性判断"""
    if not text or len(text) < 3:
        return False
    for sw in STOP_WORDS:
        if sw in text:
            return False
    if re.match(r'^[\d\s\-\.Xx]+$', text):
        return False
    if len(text) <= 3 and not any(kw in text for kw in ["公司", "局", "院", "所", "校"]):
        return False
    # 6 字以上且含中文 → 大概率是有效单位名，不强制匹配关键词
    if len(text) >= 6 and any('\u4e00' <= c <= '\u9fff' for c in text):
        return True
    # 3-5 字需要包含单位关键词
    if len(text) <= 5 and any(kw in text for kw in ORG_KEYWORDS):
        return True
    return False


def merge_seal_texts(sorted_results: list) -> list:
    """将 seal_recognition 打散的文本碎片合并"""
    import re as _re
    merged = []
    i = 0
    while i < len(sorted_results):
        text = sorted_results[i].get("text", "").strip()
        conf = sorted_results[i].get("confidence", 0.0)
        if not text or len(text) < 2:
            i += 1
            continue
        combined = text
        j = i + 1
        while j < len(sorted_results):
            next_text = sorted_results[j].get("text", "").strip()
            if not next_text or len(next_text) < 2:
                j += 1
                continue
            if any(sw in next_text for sw in STOP_WORDS):
                break
            if _re.match(r'^[\dA-Za-zXx\-\(\)]+$', next_text) and len(next_text) >= 6:
                break
            candidate = combined + next_text
            # 合并后包含任意单位关键词则继续合并
            if any(kw in candidate for kw in ORG_KEYWORDS):
                combined = candidate
                conf = max(conf, sorted_results[j].get("confidence", 0.0))
                j += 1
            else:
                break
        merged.append((combined, conf))
        i = j
    return merged
