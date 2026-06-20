import re

# 定义规则字典
prefix_rules = {
    "110": lambda N: -0.5,
    "125": lambda N: 0.003125 * (N - 0.5),
    "127": lambda N: 0.25 + 0.003125 * (N - 0.5),
    "129": lambda N: 0.5 + 0.003125 * (N - 0.5),
    "131": lambda N: 0.75 + 0.003125 * (N - 0.5),
    "140": lambda N: 3
}

# 从文件名提取前缀和编号部分
def extract_prefix_and_number(filename):
    # 匹配前三位数字作为前缀，后两位数字作为编号
    match = re.match(r'(\d{3})(\d{2})', filename)
    if match:
        prefix = match.group(1)
        number = int(match.group(2))  # 例如 '02' -> 2
        return prefix, number
    else:
        raise ValueError(f"Filename '{filename}' does not match expected pattern.")

# 根据前缀和编号生成字典
def generate_y_dict(filenames):
    y_dict = {}
    for filename in filenames:
        prefix, number = extract_prefix_and_number(filename)
        if prefix in prefix_rules:
            y_value = prefix_rules[prefix](number)
            y_dict[filename] = y_value
        else:
            raise ValueError(f"Prefix '{prefix}' not found in the defined rules.")
    return y_dict

if __name__ == '__main__':
    # 示例txt文件名
    txt_files = [
        '11001.txt', '12502.txt', '12703.txt', '12904.txt', '13105.txt', '14001.txt'
    ]

    # 生成字典
    y_dict = generate_y_dict([f[:-4] for f in txt_files])  # 去掉'.txt'后缀
    print(y_dict)
