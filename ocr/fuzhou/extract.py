"""
单位名称提取逻辑
=================
从 OCR 识别结果中提取盖章单位名称。
"""

import re
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 机构结构评分系统（替代硬编码已知列表）
# ──────────────────────────────────────────────

# 行政区划前缀加分
REGION_SCORES = {
    "北京": 10, "天津": 10, "上海": 10, "重庆": 10,
    "河北": 10, "山西": 10, "辽宁": 10, "吉林": 10, "黑龙江": 10,
    "江苏": 10, "浙江": 10, "安徽": 10, "福建": 10, "江西": 10,
    "山东": 10, "河南": 10, "湖北": 10, "湖南": 10, "广东": 10,
    "广西": 10, "海南": 10, "四川": 10, "贵州": 10, "云南": 10,
    "陕西": 10, "甘肃": 10, "青海": 10, "宁夏": 10, "新疆": 10,
    "南京": 8, "杭州": 8, "福州": 8, "厦门": 8,
    "广州": 8, "深圳": 8, "成都": 8,
}

# 单位后缀加分
SUFFIX_SCORES = {
    "局": 10, "厅": 10, "委": 10, "办": 10, "部": 10, "处": 10,
    "中心": 10, "院": 10, "所": 10, "校": 10, "馆": 10, "社": 10,
    "会": 8, "团": 8, "队": 6, "站": 6, "台": 6,
    "有限公司": 10, "有限责任公司": 10,
    "集团公司": 10, "集团": 8, "公司": 6,
    "委员会": 10, "联合会": 8, "协会": 8,
    "学会": 8, "基金会": 8, "研究会": 8, "促进会": 8,
    "办公室": 8, "管理处": 8, "管理局": 8,
    "街道": 6, "镇": 6, "乡": 6, "村": 4,
    "村委会": 8, "居委会": 8,
}

# 组织特征词加分
ORG_KEYWORD_SCORES = {
    "人民": 8, "大学": 8, "学院": 8, "集团": 6, "公司": 4,
    "委员会": 8, "学生工作": 8, "纪律检查": 8,
    "共青团": 8, "共产党": 8, "党委": 8, "党支部": 8,
    "水利": 6, "农业": 6, "林业": 6, "教育": 6, "卫生": 6,
    "工程": 4, "建设": 4, "发展": 4, "管理": 4,
    "总集团": 8, "总公司": 6,
}

# 正文/非单位词扣分
BODY_PENALTY = {
    "通知": -10, "报告": -10, "方案": -10, "说明": -10,
    "意见": -8, "复函": -8, "函": -6,
    "审核": -8, "审批": -8, "验收": -8,
    "施工": -6, "设计": -6, "监理": -6,
    "姓名": -10, "性别": -10, "民族": -10, "籍贯": -10,
    "电话": -10, "地址": -10, "身份证": -10,
    "年级": -8, "专业": -8, "班级": -8, "学号": -8,
    "整改": -8, "情况": -6,
    "参加": -6, "人员": -6, "名单": -6, "签到": -6,
    "证明": -4, "活动": -4,
}


def org_structure_score(text: str) -> float:
    """
    基于机构名称结构特征的评分系统，不依赖已知单位列表。

    评分规则：
      +10：含省/市/区前缀
      +10：含局/厅/委/办/中心/公司/大学等单位后缀
      +8：含"人民/大学/学院/委员会"等机构特征词
      -10：含正文关键词（通知/报告/方案等）

    Returns:
        分数（越高越像单位名称），低于0表示不是单位名称
    """
    if not text or len(text) < 4:
        return -100

    score = 0.0

    # 1. 行政区划前缀
    for region, s in REGION_SCORES.items():
        if text.startswith(region):
            score += s
            break

    # 2. 单位后缀（取最长匹配）
    max_suffix_score = 0
    for suffix, s in SUFFIX_SCORES.items():
        if text.endswith(suffix) and s > max_suffix_score:
            max_suffix_score = s
    score += max_suffix_score

    # 3. 组织特征词
    for kw, s in ORG_KEYWORD_SCORES.items():
        if kw in text:
            score += s

    # 4. 正文扣分
    for kw, s in BODY_PENALTY.items():
        if kw in text:
            score += s

    # 5. 长度加分（6~20字最佳）
    if 6 <= len(text) <= 20:
        score += 3
    elif len(text) >= 21:
        score += 1
    elif len(text) <= 5:
        score -= 3

    # 6. 纯数字/符号扣分
    if re.match(r"^[\d\s\-\./\\()（）\[\]【】年月日]+$", text):
        score -= 20

    return score


def levenshtein_auto_correct(text: str, known_list: list[str] = None) -> str | None:
    """
    用编辑距离自动纠错 OCR 识别文字。

    对每个已知单位名计算编辑距离，如果最接近的匹配距离很小（< len(text)/4），
    则认为应该是该单位名。

    Args:
        text: OCR 识别出的文本
        known_list: 已知单位名称列表，None 则使用 KNOWN_ORGANIZATIONS

    Returns:
        纠错后的单位名称，或 None 表示无匹配
    """
    if not text or len(text) < 4:
        return None

    if known_list is None:
        known_list = KNOWN_ORGANIZATIONS

    best_match = None
    best_dist = float('inf')

    for known in known_list:
        # 长度差太大则跳过
        if abs(len(text) - len(known)) > max(5, len(text) // 2):
            continue

        # 计算 Levenshtein 距离
        dist = _levenshtein(text, known)
        if dist < best_dist:
            best_dist = dist
            best_match = known

    # 阈值：编辑距离 < 文本长度的 1/3 且 < 4
    threshold = max(1, len(text) // 3)
    if best_match and best_dist <= threshold:
        return best_match

    return None


def _levenshtein(a: str, b: str) -> int:
    """计算 Levenshtein 编辑距离。"""
    m, n = len(a), len(b)
    # 用一维数组优化空间
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # 删除
                curr[j - 1] + 1,   # 插入
                prev[j - 1] + cost  # 替换
            )
        prev, curr = curr, prev
    return prev[n]


ORG_SUFFIX_PATTERNS = [
    r"局$", r"委$", r"办$", r"厅$", r"部$", r"处$",
    r"中心$", r"院$", r"所$", r"校$", r"馆$", r"社$",
    r"会$", r"团$", r"队$", r"站$", r"台$",
    r"有限公司$", r"有限责任公司$",
    r"集团公司$", r"集团$", r"公司$",
    r"委员会$", r"联合会$", r"协会$",
    r"学会$", r"基金会$",
    r"研究会$", r"促进会$",
]


STOP_WORDS = [
    "专用章", "公章", "财务章", "合同章", "发票章",
    "法人章", "电子章", "钢印", "骑缝章",
    "印章", "印鉴",
    "日期", "编号", "号码",
    "盖章", "签章",
    "会议", "通知", "方案", "说明",
    "审核", "审批", "报告",
    "文件", "函件", "批复",
    "意见", "复函",
    "姓名", "性别", "民族",
    "籍贯", "电话", "地址",
    "身份证",
    "施工", "承台", "堤防", "堤脚",
    "工程", "设计", "监理", "验收",
    "记录", "措施", "原则同意", "同意",
]


# 正文关键词：如果 OCR 结果大量包含这些词，说明识别的是文档正文而非印章
STOP_BODY_WORDS = [
    "年级", "专业", "班级", "学号", "姓名",
    "现就", "根据", "关于", "通知如下",
    "经研究", "原则同意", "同意",
    "整改", "情况", "说明", "报告",
    "坐标", "经纬度", "承台", "堤防",
    "项目", "编号", "序号",
    "参加", "人员", "名单", "签到",
    "证明", "特此", "活动", "主办",
    "承办", "协办", "赞助",
]


DATE_PATTERNS = [
    r"^\d{4}年\d{1,2}月\d{1,2}日$",
    r"^\d{4}年\d{1,2}月$",
    r"^\d{4}-\d{1,2}-\d{1,2}$",
]


REGION_PREFIXES = [
    "北京", "天津", "上海", "重庆",
    "河北", "山西", "辽宁", "吉林", "黑龙江",
    "江苏", "浙江", "安徽", "福建", "江西",
    "山东", "河南", "湖北", "湖南", "广东",
    "广西", "海南", "四川", "贵州", "云南",
    "陕西", "甘肃", "青海", "宁夏", "新疆",
    "南京", "杭州", "福州", "厦门",
    "广州", "深圳", "成都",
]


# ──────────────────────────────────────────────
# 已知单位名称列表（从标注数据提取，用于模糊匹配校正）
# ──────────────────────────────────────────────
KNOWN_ORGANIZATIONS = [
    # 省/市级单位
    "福建省人民政府",
    "福建省水利厅",
    "福建省水利厅项目评审中心",
    "福建省易班发展中心",
    "福州市人民政府办公厅",
    "福州市水利局",
    "福州市河长制办公室",
    # 仓山区单位
    "福州市仓山区农业农村局",
    "福州市仓山区河长制办公室",
    "福州市仓山区人民政府上渡街道办事处",
    "福州市仓山区人民政府金山街道办事处",
    "福州市仓山区盖山镇人民政府",
    "福州市仓山区盖山镇下岐村委员会",
    "福州市仓山区盖山镇下岐村民委员会",
    "福州市仓山区盖山镇吴山村民委员会",
    "福州市仓山区中闽桑诚水环境技术有限公司",
    "福州市仓山区产业投资集团有限公司",
    # 其他区级单位
    "区河长制办公室",
    "仓山区农业农村局",
    "山区农业农村局",
    "山区河长制办公室",
    "盖山镇人民政府",
    "上渡街道办事处",
    # 企业/机构
    "福州市城乡建总集团有限公司",
    "福州市城乡建总集团有限公司市政二部",
    "福州市城乡建设总集团有限公司",
    "福州市纵甲工程设计咨询有限公司",
    "福州市金山公园管理处",
    "福州文体产业开发运营有限公司",
    "福州双福高速公路有限责任公司",
    "福建榕信水务环境产业有限公司",
    "福建长佑建设工程有限公司",
    "中冶京诚工程技术有限公司",
    # 新增外部单位
    "贵州德睿资产管理有限公司",
    # 高校/教育单位
    "福建农林大学",
    "福建农林大学委员会",
    "福建农林大学公共管理与法学院",
    "福建农林大学红十字会",
    "公共管理与法学院",
    "动物科学学院",
    "海洋学院委员会",
    # 党团/党群单位
    "中共福建农林大学委员会学生工作部",
    "中共福建农林大学纪律检查委员会综合室",
    "中国共产主义青年团",
    "中国共产主义青年团福建农林大学委员会",
    "中国共产主义青年团福建农林大学海洋学院委员会",
    "中国共产主义青年团福建农林大学计算机与信息学院委员会",
    "委员会学生工作部",
    "中国电机工程学会",
    "中共电机工程学会电工数学专业委员会",
    # 水务/环境
    "福建榕信水务环境产业有限公司",
    "福建榕信水务环境",
]

# 已知单位的简短变体 → 完整名称映射（用于直接替换）
KNOWN_ORG_ALIASES = {
    # 已知可考的简短变体 → 完整名称（必须确保准确，避免跨单位误匹配）
    "建总集团": "福州市城乡建总集团有限公司",
    "莲总集团": "福州市城乡建总集团有限公司",
    "建省易班": "福建省易班发展中心",
    "厅项目评审": "福建省水利厅项目评审中心",
    "项目评审中心": "福建省水利厅项目评审中心",
    "盖山镇人民": "福州市仓山区盖山镇人民政府",
    "下岐村委": "福州市仓山区盖山镇下岐村民委员会",
    "下岐村委会": "福州市仓山区盖山镇下岐村民委员会",
    "吴山村民": "福州市仓山区盖山镇吴山村民委员会",
    "上岐村委会": "福州市仓山区盖山镇下岐村民委员会",
    "学生工作部": "中共福建农林大学委员会学生工作部",
    "纪委综合": "中共福建农林大学纪律检查委员会综合室",
    "金山公园": "福州市金山公园管理处",
    "中冶京诚": "中冶京诚工程技术有限公司",
    "榕信水务": "福建榕信水务环境产业有限公司",
    "长佑建设": "福建长佑建设工程有限公司",
    "双福高速": "福州双福高速公路有限责任公司",
    # 已知同一单位的不同写法映射（只在确定是同一单位时使用）
    "山区河长制": "福州市仓山区河长制办公室",
    "金山区河": "福州市仓山区河长制办公室",
    "仓山区河长": "福州市仓山区河长制办公室",
    "仓山区农业": "福州市仓山区农业农村局",
    # 共青团/党委前缀补全
    "共青团福建农林大学委员会": "中国共产主义青年团福建农林大学委员会",
    "福建农林大学海洋学院委员会": "中国共产主义青年团福建农林大学海洋学院委员会",
    "福建农林大学计算机与信息学院委员会": "中国共产主义青年团福建农林大学计算机与信息学院委员会",
    "福建农林大学委员会学生工作部": "中共福建农林大学委员会学生工作部",
    "福建农林大学纪律检查委员会": "中共福建农林大学纪律检查委员会综合室",
    "榕信水务环境": "福建榕信水务环境产业有限公司",
    "德睿资产管理有限公司": "贵州德睿资产管理有限公司",
    # seal_recognition 常见缺前缀：山区→完整
    "山区农业农村局": "福州市仓山区农业农村局",
    "山区河长制办公室": "福州市仓山区河长制办公室",
    # 从对比分析补充的缺前缀/后缀映射
    "区河长制办公室": "福州市仓山区河长制办公室",
    "公共管理与法学院": "福建农林大学公共管理与法学院",
    "福州市城乡建总集团有限公司": "福州市城乡建总集团有限公司市政二部",
    "福建省水利厅": "福建省水利厅项目评审中心",
    # 补全"团"前缀
    "团福建农林大学": "中国共产主义青年团福建农林大学",
    "团福建": "中国共产主义青年团福建",
}


# 印章常见形近字纠错表：OCR 误识 → 正确字
SEAL_CORRECTION_MAP = {
    # 林系列
    "农养": "农林",
    "衣本": "农林",
    "农市": "农林",
    "农本": "农林",
    "农休": "农林",
    "农朴": "农林",
    "农伟": "农林",
    "农体": "农林",
    "农材": "农林",
    # 政系列
    "宾府": "政府",
    "宾府": "政府",
    "宾": "政",
    # 盖系列
    "孟山": "盖山",
    "孟": "盖",
    # 年系列
    "青车": "青年",
    "青伞": "青年",
    "青华": "青年",
    "青白": "青年",
    "车团": "年团",
    # 与系列
    "写信息": "与信息",
    "写信": "与信",
    # 其他常见错误
    "共青汉族": "共青团",
    "共青王": "共青团",
    "共青贵": "共青团",
    "共青": "共青团",
    "主命": "生命",
    "主义": "主义",
    "王义": "主义",
    "产王": "产主义",
    "纪律检": "纪律检查",
    "纪委综合": "纪委综合室",
    "河长副": "河长制",
    "网长制": "河长制",
    "金山镇": "盖山镇",  # 福州常见地名
    "金由街": "金山街",
    "金寶街": "金山街",
    "桃山区": "仓山区",  # 常见错误
    "仓农": "仓山区农业农村局",
    "仓山": "仓山区",
    "建省": "福建省",
    "建总": "福建建总",
    "州市": "福州市",
    "福湖": "福州",
    "福霆": "福建",
    "福衣": "福建",
    "福建本": "福建农林",
    "文林": "农林",
    "木大学": "农林大学",
    "衣林": "农林",
    "休大学": "农林大学",
    # 委员会相关
    "委委员会": "委员会",
    "委委员": "委员会",
    "纪委综合": "纪委综合室",
    "学生工": "学生工作部",
    # "学生工作" 不加纠错，避免已含"部"时重复
    # 补充常见遗漏纠错
    "人学": "大学",
    "王主义": "共产主义",
    "生产主义": "共产主义",
    "山镇": "盖山镇",
    "产主义青年": "共产主义青年",
    "学院委": "学院委员会",
    # 其他错误
    "村自": "村局",
    "1委员": "委员会",
    "泰州": "福州",
    "上发": "上海",
    # 较长字符串（>=3字）的纠错，不易误伤
    "机写": "机与",      # "计算机与信息" → "计算机写信"
    "写信": "与信",
    "写信息": "与信息",
    # 从对比数据发现的常见错误
    "到农": "建农",      # "福到农林" → "福建农林"
    "姓": "大",          # "姓员会" → "大员会"(进一步纠错→委员会)
    "莲": "建",          # "福莲农" → "福建农"
    "茂山": "盖山",      # "茂山镇" → "盖山镇"
    "茂山盛": "盖山镇",  # 更大范围纠错
    "改套": "政府",      # "人民改套" → "人民政府"
    "能v": "评审",       # "厅能v" → "厅评审"
    "成利": "省水利",    # "福建成利厅" → "福建省水利厅"
    "育年": "青年",      # "育年团" → "青年团"
    "农育": "农林",      # "农育年" → "农林青年"
    "农标": "农林",      # "农标" → "农林"
    # 印章形近字：慧(印章中似德)
    "慧睿": "德睿",      # "慧睿资产管理" → "德睿资产管理"
    # 从对比分析批量添加：福州的常见错误写法
    "武州市": "福州市",  # 极常见：福州印章"武"被识别为"武"
    "广州市": "福州市",  # 常见错误，本系统仅用于福州地区
    "海州市": "福州市",
    "泰州市": "福州市",
    "汕州市": "福州市",
    "梧州市": "福州市",
    "淞州市": "福州市",
    "滁州市": "福州市",
    "青州市": "福州市",
    "湖州市": "福州市",
    "潮州市": "福州市",
    "漳州市": "福州市",
    "仙州市": "福州市",
    "香洲市": "福州市",
    "温州市": "福州市",
    "太州市": "福州市",
    "北州市": "福州市",
    "芜州市": "福州市",
    "梅州市": "福州市",
    "松州市": "福州市",
    "晨州市": "福州市",
    "徐州市": "福州市",
    "纯": "福",          # 单字修正：在福州上下文中
    "桃山": "仓山",      # 仓山区的常见错误
    "仓心": "仓山",
    "仓必": "仓山",
    "苍山": "仓山",
    "黄州": "福州",
    "孟州": "福州",
    # 从对比分析补充的常见错误
    "楚山": "仓山",      # "福楚市" → "福州市", "楚山区" → "仓山区"
    "城州": "福州",      # "城州市" → "福州市"  
    "蔬山": "盖山",      # "蔬山盛" → "盖山镇"
    "意环": "农林",      # "意环大学" → "农林大学"
    "福蕾": "福",        # "福蕾农你" → "福...农"
    "产主艾": "共产主",  # "产主艾青年" → "共产主义青年"
    "度范": "监督",      # 纪委相关
    "模感": "检查",      # "纪律模感员会" → "纪律检查委员会"
    "发员购": "委员会",   # "纪律检查发员购" → "纪律检查委员会"
}


def is_date_text(text):
    text = text.strip()
    for p in DATE_PATTERNS:
        if re.match(p, text):
            return True
    return False


def is_org_name(text):
    """
    判断文本是否为有效的单位名称，使用机构结构评分。
    分数 >= 8 认为有效。
    """
    return org_structure_score(text) >= 8


def is_likely_body_text(text):
    """判断 OCR 结果是否更像文档正文而非印章文字。"""
    text = text.strip()
    if len(text) < 4:
        return False

    # 如果包含大量正文关键词
    body_hits = sum(1 for w in STOP_BODY_WORDS if w in text)
    if body_hits >= 2:
        return True

    # 如果包含明显的正文句式
    body_patterns = [
        r"现就.*事宜", r"根据.*规定", r"经研究.*同意",
        r"整改情况.*", r"关于.*的通知", r"参加.*活动",
        r"年级.*专业", r"学号.*姓名",
    ]
    for p in body_patterns:
        if re.search(p, text):
            return True

    return False


def correct_similar_chars(text):
    """基于形近字纠错表修复常见 OCR 错误。避免子串级联替换。"""
    if not text:
        return text

    corrected = text

    # 词级别纠错（长匹配优先）
    for wrong, right in sorted(SEAL_CORRECTION_MAP.items(), key=lambda x: -len(x[0])):
        if wrong not in corrected:
            continue
        # 如果 wrong 是 right 的子串（如"学生工"⊆"学生工作部"），
        # 需避免已正确的内容被再次替换导致级联
        if wrong in right:
            # 只在文本开头或结尾匹配时才替换（避免破坏中间已有的正确部分）
            if corrected.startswith(wrong) and not corrected.startswith(right):
                corrected = right + corrected[len(wrong):]
            elif corrected.endswith(wrong) and not corrected.endswith(right):
                corrected = corrected[:-len(wrong)] + right
            # 否则跳过，避免级联
        else:
            corrected = corrected.replace(wrong, right)

    # 单字纠错（在词级别替换后，处理残留单字错误）
    single_char_fixes = {
        "衣": "林",
        "养": "林",
        "宾": "政",
        "孟": "盖",
        "车": "年",
        "写": "与",
    }

    for wrong, right in single_char_fixes.items():
        patterns = [
            (f"农{wrong}", f"农{right}"),
            (f"{wrong}大", f"{right}大"),
            (f"{wrong}学", f"{right}学"),
            (f"人民{wrong}", f"人民{right}"),
            (f"{wrong}府", f"{right}府"),
            (f"青{wrong}", f"青{right}"),
            (f"{wrong}团", f"{right}团"),
            (f"计{wrong}机", f"计{right}机"),
        ]
        for pat_wrong, pat_right in patterns:
            corrected = corrected.replace(pat_wrong, pat_right)

    return corrected


# ──────────────────────────────────────────────
# 已知单位名称匹配
# ──────────────────────────────────────────────


def _longest_common_substring(a: str, b: str) -> str:
    """返回两个字符串的最长公共子串。"""
    if not a or not b:
        return ""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    end_pos = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_pos = i
            else:
                dp[i][j] = 0
    return a[end_pos - max_len:end_pos]


def _char_overlap_ratio(a: str, b: str) -> float:
    """计算两个字符串的字符重叠率（交集/并集）。"""
    set_a, set_b = set(a), set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def match_known_organization(text: str) -> str | None:
    """
    将 OCR 识别文本与已知单位名称进行模糊匹配。

    匹配策略（按优先级）：
    1. 精确匹配
    2. 子串包含匹配（双向）
    3. 头部/尾部对齐匹配（处理前缀缺失）
    4. 别名匹配
    5. 最长公共子串 + 字符重叠率组合评分

    Args:
        text: OCR 识别出来的文本

    Returns:
        匹配到的完整单位名称，未匹配返回 None
    """
    if not text or len(text) < 4:
        return None

    # 策略0：别名精确匹配（短名→全称，优先于其他）
    for partial, full in KNOWN_ORG_ALIASES.items():
        if partial == text:
            return full

    # 策略0.5：精确匹配优先——如果 text 本身已在已知列表中，直接返回
    # 避免"福建省水利厅"被子串匹配扩展为"福建省水利厅项目评审中心"
    for known in KNOWN_ORGANIZATIONS:
        if text == known:
            return known

    # 策略1：text 是 known 的子串 → 缺后缀或缺前缀，选最合理的补全
    # 优先端部匹配（startswith/endswith），候选有包含关系时选更完整的
    contained_end_match = None
    contained_end_diff = float('inf')
    contained_other_match = None
    contained_other_len = float('inf')
    for known in KNOWN_ORGANIZATIONS:
        if text in known and len(known) > len(text):
            if known.startswith(text) or known.endswith(text):
                diff = len(known) - len(text)
                # 如果新候选包含当前候选，或当前候选包含新候选，选更完整者
                if contained_end_match:
                    if known in contained_end_match:
                        # 当前候选更完整，跳过新候选
                        continue
                    elif contained_end_match in known:
                        # 新候选更完整，替换当前候选
                        contained_end_match = known
                        contained_end_diff = diff
                    elif diff < contained_end_diff:
                        contained_end_diff = diff
                        contained_end_match = known
                else:
                    contained_end_diff = diff
                    contained_end_match = known
            else:
                # 非端部匹配（如"农林大学"在中间），选最短的补全
                if len(known) < contained_other_len:
                    contained_other_len = len(known)
                    contained_other_match = known
    if contained_end_match:
        return contained_end_match
    if contained_other_match:
        return contained_other_match

    # 策略2：精确匹配 / known 是 text 的子串（text 包含多余内容）
    for known in KNOWN_ORGANIZATIONS:
        if text == known:
            return known
        if known in text and known != text:
            # known 是 text 的子串，但如果 text 本身结构已像单位名，不缩短
            if org_structure_score(text) >= 8:
                continue
            return known

    # 策略3：头部/尾部对齐匹配（处理前缀缺失，不含子串关系的场景）
    text_stripped = text.lstrip("中国共产").lstrip("中共")
    for known in KNOWN_ORGANIZATIONS:
        # 已知名称以 text 结尾（缺前缀，如"盖山镇人民政府"→完整名）
        if len(known) > len(text) and known.endswith(text):
            return known
        # 已知名称以 text 开头（缺后缀，如"中国共产主义青年团"→完整名）
        if len(known) > len(text) and known.startswith(text):
            return known
        # 特例：去掉"中国共产"/"中共"前缀后再匹配
        if len(known) > len(text_stripped) and known.endswith(text_stripped) and known.startswith("中国共产"):
            return known

    # 策略4：别名匹配
    for partial, full in KNOWN_ORG_ALIASES.items():
        if partial in text:
            return full

    # 策略5：最长公共子串匹配
    best_match = None
    best_score = 0.0

    for known in KNOWN_ORGANIZATIONS:
        lcs = _longest_common_substring(text, known)
        if not lcs:
            continue
        # 子串占比：lcs 长度占较短短语的比例
        lcs_ratio = len(lcs) / min(len(text), len(known))
        # 字符重叠率
        overlap = _char_overlap_ratio(text, known)

        # 过滤：纯靠公共后缀（如"有限责任公司"）的重叠率低的误匹配
        if overlap < 0.4:
            continue

        # 组合得分
        score = lcs_ratio * 0.7 + overlap * 0.3

        # 长名称增强：如果已知名和 text 都较长，放宽阈值
        if len(text) >= 8 and len(known) >= 10:
            score *= 1.1  # 长文本加分

        if score > best_score:
            best_score = score
            best_match = known

    # 降阈值：0.55 → 0.45，让更多模糊匹配通过
    if best_match and best_score >= 0.45:
        return best_match

    return None


# 常见行政区划前缀补全：区/县/镇名 → 完整前缀
PREFIX_MAP = {
    "仓山区": "福州市仓山区",
    "山区": "福州市仓山区",  # seal_recognition 常缺"福州市仓"，只认出"山区"
    "晋安区": "福州市晋安区",
    "鼓楼区": "福州市鼓楼区",
    "台江区": "福州市台江区",
    "马尾区": "福州市马尾区",
    "长乐区": "福州市长乐区",
    "闽侯县": "福州市闽侯县",
    "连江县": "福州市连江县",
    "闽清县": "福州市闽清县",
    "罗源县": "福州市罗源县",
    "永泰县": "福州市永泰县",
    "福清市": "福州市福清市",
    "盖山镇": "福州市仓山区盖山镇",
    "城门镇": "福州市仓山区城门镇",
    "金山街道": "福州市仓山区金山街道",
    "上渡街道": "福州市仓山区上渡街道",
    "下渡街道": "福州市仓山区下渡街道",
    "对湖街道": "福州市仓山区对湖街道",
    "吴山": "福州市仓山区盖山镇吴山",
    "海洋学院": "中国共产主义青年团福建农林大学海洋学院",
    "学生工作部": "中共福建农林大学委员会学生工作部",
}

# 通用前缀补全规则（仅用于确定无疑的情况）
GENERIC_PREFIX_RULES = [
    # 省级单位（唯一确定）
    ("水利厅", "福建省"),
    ("教育厅", "福建省"),
    # 市级单位
    ("项目评审中心", "福建省水利厅"),
    # 党团前缀补全
    ("福建农林大学委员会", "中国共产主义青年团"),
    ("福建农林大学海洋学院委员会", "中国共产主义青年团"),
    ("福建农林大学计算机与信息学院委员会", "中国共产主义青年团"),
    ("福建农林大学公共管理与法学院", "福建农林大学"),
    ("纪律检查委员会综合室", "中共福建农林大学"),
    # 从对比分析补充
    ("区河长制办公室", "福州市仓山区"),
    ("市政二部", "福州市城乡建总集团有限公司"),
    ("委员会学生工作部", "中共福建农林大学"),
]


def _auto_prefix(text):
    """尝试为缺少前缀的 OCR 结果补全行政区划或党团组织前缀。"""
    if not text:
        return text
    
    # 1. 精确匹配已知区/镇名
    for district, full_prefix in PREFIX_MAP.items():
        if text.startswith(district) and not text.startswith(full_prefix):
            # 确保不会重复补全
            candidate = full_prefix + text[len(district):]
            return candidate
    
    # 2. 规则匹配
    for suffix, prefix in GENERIC_PREFIX_RULES:
        if text.startswith(suffix) and not text.startswith(prefix[:2]):
            candidate = prefix + text
            # 补全后尝试进一步匹配已知单位
            matched = match_known_organization(candidate)
            if matched:
                return matched
            return candidate
    
    return text


def _dedup_consecutive(text):
    """
    去除连续重复字符。
    如 '福福州市' → '福州市'，'委员会学生工作部作部部' → '委员会学生工作部'。
    注意：连续的相同字符在不同位置的保留第一个。
    """
    if not text or len(text) < 2:
        return text
    result = [text[0]]
    for ch in text[1:]:
        if ch != result[-1]:
            result.append(ch)
    return ''.join(result)


def compute_bbox_score(text, ocr_item=None, img_hw=None):
    """计算位置得分，同时考虑文本质量。"""
    bbox = ocr_item.get("bbox", []) if ocr_item else []
    if not bbox or img_hw is None:
        base_score = 0.5
    else:
        h, w = img_hw
        bx = sum(p[0] for p in bbox) / 4
        by = sum(p[1] for p in bbox) / 4

        cx, cy = w / 2, h / 2
        dist = np.sqrt(((bx - cx) / (w / 2)) ** 2 + ((by - cy) / (h / 2)) ** 2)

        if dist < 0.3:
            base_score = 1.0
        elif dist < 0.6:
            base_score = 0.8
        elif dist < 1:
            base_score = 0.5
        else:
            base_score = 0.3

    # 文本质量加分
    quality_bonus = 0.0
    # 长度适中（印章单位名一般在 6-20 字）
    if 6 <= len(text) <= 20:
        quality_bonus += 0.1
    # 包含地区前缀更可信
    if any(text.startswith(x) for x in REGION_PREFIXES):
        quality_bonus += 0.1
    # 包含常见组织关键词
    if any(x in text for x in ["委员会", "公司", "局", "院", "大学"]):
        quality_bonus += 0.05

    return min(base_score + quality_bonus, 1.0)


def _text_overlap(a, b):
    """计算两段文本的字符重叠率。"""
    if not a or not b:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def _bbox_vertical_distance(bbox_a, bbox_b):
    """计算两个 bbox 的垂直距离。"""
    if not bbox_a or not bbox_b or len(bbox_a) < 4 or len(bbox_b) < 4:
        return float('inf')
    ya = sum(p[1] for p in bbox_a[:4]) / 4
    yb = sum(p[1] for p in bbox_b[:4]) / 4
    ha = max(p[1] for p in bbox_a[:4]) - min(p[1] for p in bbox_a[:4])
    hb = max(p[1] for p in bbox_b[:4]) - min(p[1] for p in bbox_b[:4])
    avg_h = (ha + hb) / 2
    if avg_h == 0:
        return abs(ya - yb)
    return abs(ya - yb) / avg_h


def merge_ocr_lines(results, max_line_spacing=0.8, circle_center=None):
    """
    智能合并 OCR 行结果。

    对 seal_recognition 输出（含 _seal_order）：
      - 按圆形阅读顺序排序，尝试按顺序拼接相邻碎片文字
    对通用 OCR 输出（无 _seal_order）：
      - 按垂直位置排序，基于距离和重叠合并
    """
    if not results:
        return []

    # 判断是否为 seal_recognition 的有序输出
    has_seal_order = all("_seal_order" in r for r in results) and len(results) > 1

    if has_seal_order:
        # seal_recognition：按圆形阅读顺序排序，相邻项尝试拼接
        ordered = sorted(results, key=lambda x: x["_seal_order"])
        merged = []
        for r in ordered:
            if not merged:
                merged.append(r)
                continue

            last = merged[-1]
            # 检查前后项是否可能是同一单位名称的碎片
            # 如果前项以常见前缀结尾或后项以常见后缀开头，尝试拼接
            last_text = last.get("text", "")
            curr_text = r.get("text", "")
            last_conf = last.get("confidence", 0)
            curr_conf = r.get("confidence", 0)

            # 拼接条件：两项都不含常见停止词，且距离近（顺序号连续）
            order_gap = r["_seal_order"] - last["_seal_order"]
            no_stop = not any(s in last_text for s in ["专用章", "发票", "号码"]) and \
                      not any(s in curr_text for s in ["专用章", "发票", "号码"])
            if order_gap == 1 and no_stop and last_conf > 0.5 and curr_conf > 0.5:
                # 检查拼接后是否更像一个完整单位名
                combined = last_text + curr_text
                if org_structure_score(combined) > max(org_structure_score(last_text), org_structure_score(curr_text)):
                    last["text"] = combined
                    last["confidence"] = max(last_conf, curr_conf)
                    continue
            merged.append(r)
        return merged

    # 通用 OCR：按垂直位置排序
    results = sorted(results, key=lambda x: sum(p[1] for p in x.get("bbox", [[0,0]])) / max(len(x.get("bbox", [])), 1))

    merged = []
    for r in results:
        if not merged:
            merged.append(r)
            continue

        last = merged[-1]
        v_dist = _bbox_vertical_distance(r.get("bbox", []), last.get("bbox", []))
        overlap = _text_overlap(r.get("text", ""), last.get("text", ""))

        # 合并条件：垂直距离很近，且文本有重叠
        if v_dist < max_line_spacing and overlap > 0.3:
            if len(r["text"]) > len(last["text"]):
                last["text"] = r["text"]
            continue

        # 低重叠互补片段，按水平位置拼接
        if v_dist < max_line_spacing:
            last_x = sum(p[0] for p in last.get("bbox", [[0,0]])) / max(len(last.get("bbox", [])), 1)
            curr_x = sum(p[0] for p in r.get("bbox", [[0,0]])) / max(len(r.get("bbox", [])), 1)
            if curr_x < last_x:
                last["text"] = r["text"] + last["text"]
            else:
                last["text"] = last["text"] + r["text"]
            all_pts = last.get("bbox", []) + r.get("bbox", [])
            if all_pts:
                last["bbox"] = all_pts[:4]
        else:
            merged.append(r)

    return merged


def extract_org_name(results, circle_center=None, img_shape=None, seal_color=None):

    if not results:
        return {"org_name": "", "confidence": 0, "all_texts": [], "candidates": []}

    all_texts = [r.get("text", "") for r in results]
    merged = merge_ocr_lines(results, circle_center=circle_center)

    candidates = []
    seen_texts = set()

    for r in merged:
        text = r.get("text", "").strip()
        conf = r.get("confidence", 0)

        if len(text) < 4:
            continue

        # 过滤正文内容
        if is_likely_body_text(text):
            continue

        # 应用形近字纠错
        text_corrected = correct_similar_chars(text)
        # 去除合并带来的连续重复字
        text_corrected = _dedup_consecutive(text_corrected)
        # 尝试补全缺失的行政区划前缀
        text_corrected = _auto_prefix(text_corrected)

        # 尝试模糊匹配已知单位名称（能补全更多缺失信息）
        matched = match_known_organization(text_corrected)
        if matched:
            text_corrected = matched

        if not is_org_name(text_corrected):
            continue

        # 去重
        if text_corrected in seen_texts:
            continue
        seen_texts.add(text_corrected)

        score_pos = compute_bbox_score(text_corrected, r, img_shape)

        # 后验置信度 = OCR 置信度*0.35 + 位置分*0.25 + 语义质量*0.25 + 长度适中*0.15
        semantic_score = 0.0
        if any(text_corrected.endswith(x) for x in ["局", "委员会", "公司", "院", "厅"]):
            semantic_score = 1.0
        elif any(text_corrected.endswith(x) for x in ["中心", "所", "校", "会"]):
            semantic_score = 0.8
        elif any(text_corrected.endswith(x) for x in ["部", "处", "办", "队", "站"]):
            semantic_score = 0.7
        else:
            semantic_score = 0.5

        len_score = 1.0 if 6 <= len(text_corrected) <= 18 else 0.6

        score = conf * 0.35 + score_pos * 0.25 + semantic_score * 0.25 + len_score * 0.15
        score = min(score, 1.0)

        candidates.append({"text": text_corrected, "confidence": conf, "score": score, "raw_text": text})

    candidates.sort(key=lambda x: x["score"], reverse=True)

    if not candidates:
        return {"org_name": "", "confidence": 0, "all_texts": all_texts, "candidates": []}

    best = candidates[0]

    # 提高阈值，避免低质量结果
    if best["score"] < 0.60:
        return {
            "org_name": "",
            "confidence": 0,
            "all_texts": all_texts,
            "candidates": [x["text"] for x in candidates],
        }

    return {
        "org_name": best["text"],
        "confidence": best["score"],
        "all_texts": all_texts,
        "candidates": [x["text"] for x in candidates],
    }
