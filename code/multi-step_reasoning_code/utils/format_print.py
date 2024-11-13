def format_print(seq, word_color_dict):
    """
        seq: 列表
        word_color_dict: 字典，key是特殊字符串，value是颜色
    """
    color_index_dict = {
        'r': '\033[31m',
        'g': '\033[32m',
        'y': '\033[33m',
        'b': '\033[34m',
        'purple': '\033[35m',
        'cyan': '\033[36m',
        'w': '\033[37m',
        'reset': '\033[0m',
        'r_bg': '\033[0;30;41m',      # 红底黑字
        'g_bg': '\033[0;30;42m',      # 绿底黑字
        'y_bg': '\033[0;30;43m',      # 黄底黑字
        'b_bg': '\033[0;30;44m',      # 蓝底黑字
        'purple_bg': '\033[0;30;45m',   # 紫底黑字
        'cyan_bg': '\033[0;30;46m',     # 青底黑字
        'w_bg': '\033[0;30;47m',      # 白底黑字
    }
    # 如果输入为字符串，则按照“,”或者空格分割为列表
    if isinstance(seq, str):
        if ', ' in seq:
            seq = seq.split(', ')
        elif ',' in seq:
            seq = seq.split(',')
        else:
            seq = seq.split(' ')
        
    for word in seq:
        if word in word_color_dict:
            print(color_index_dict[word_color_dict[word]], word, color_index_dict['reset'], end='')
        else:
            print(word, end=' ')
    print()