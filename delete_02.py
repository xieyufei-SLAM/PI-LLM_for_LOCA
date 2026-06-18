import pandas as pd

def delete_specific_row_from_sheets(excel_file, output_file):
    # 读取 Excel 文件
    xls = pd.ExcelFile(excel_file)

    # 创建一个字典以保存每个子表的数据
    sheets_data = {}

    for sheet_name in xls.sheet_names:
        # 读取每个子表
        df = pd.read_excel(xls, sheet_name=sheet_name)

        # 使用布尔索引删除包含 'plotnum' 的行
        df = df[~df.apply(lambda x: x.astype(str).str.contains('plotnum').any(), axis=1)]

        # 将处理后的数据保存到字典中
        sheets_data[sheet_name] = df

    # 将所有子表写入新的 Excel 文件
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name, df in sheets_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

# 使用示例
delete_specific_row_from_sheets(r'data/算例5/344New/D3BF2080.xlsx', r'data/算例5/344New/344.xlsx')
