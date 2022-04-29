import os


def find_img(root, relative_root=''):
    # 递归地找到一个文件夹下的所有图片路径，返回该文件夹的绝对路径，和其下的所有图片在该文件夹下的相对路径
    if os.path.isdir(root):
        sub_list = []
        for i in os.listdir(root):
            sub_list.extend(find_img(os.path.join(root, i), i))
        return [os.path.join(relative_root, i) for i in sub_list]
    else:
        return [os.path.join(root.split(os.sep)[-1])]


if __name__ == "__main__":
    result = find_img(r"D:\working_directory\data\test")

