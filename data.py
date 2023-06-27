import itertools
import os
import json

clean_folder_list = ['all', 'dress', 'lower', 'upper']
all_part_list = ['1', '2', '3', '4', '5', '6']
all_folder_list = ['image', 'image_text', 'image_annotation']

clean_root = '/Users/lxy/Downloads/tb/'
all_root = '/Users/lxy/Downloads/tb_train_dataset/train_dataset_part'


def save_item_list():
    clean_path_list = []
    all_path_list = []

    for folder in clean_folder_list:
        image_folder_path = clean_root + folder + '/'
        for item_folder in os.listdir(image_folder_path):
            if item_folder not in clean_path_list:
                clean_path_list.append(image_folder_path + item_folder)

    for part in all_part_list:
        image_folder_path = all_root + part + '/image/'
        for item_folder in os.listdir(image_folder_path):
            if item_folder not in all_path_list:
                all_path_list.append(image_folder_path + item_folder)

    with open('dataset/clean_list.txt', 'w') as f:
        for item in clean_path_list:
            f.write(item + '\n')

    with open('dataset/all_list.txt', 'w') as f:
        for item in all_path_list:
            f.write(item + '\n')


def get_item_list():
    clean_path_list = []
    all_path_list = []
    clean_list = []
    all_list = []
    with open('dataset/clean_list.txt', 'r') as f:
        for line in f.readlines():
            clean_path_list.append(line.strip())
            clean_list.append(line.strip().split('/')[-1])

    with open('dataset/all_list.txt', 'r') as f:
        for line in f.readlines():
            all_path_list.append(line.strip())
            all_list.append(line.strip().split('/')[-1])

    return clean_path_list, all_path_list, clean_list, all_list


def split_data(clean_path_list, all_path_list, clean_list, all_list):
    display_path_list = []
    white_path_list = []
    gt_path_list = []
    for item_path in clean_path_list:
        for img in os.listdir(item_path):
            gt_path_list.append(item_path + '/' + img)
    for (i, item_path) in enumerate(all_path_list):
        if item_path not in clean_list:
            for img in os.listdir(item_path):
                img_path = item_path + '/' + img
                annotation_path = img_path.replace('image', 'image_annotation').replace('jpg', 'json')
                with open(annotation_path, 'r') as f:
                    annotation = json.loads(f.read())
                    if annotation['annotations'][0]['display'] == 1:
                        display_path_list.append(img_path)
                    else:
                        white_path_list.append(img_path)
        else:
            for img in os.listdir(item_path):
                img_path = item_path + '/' + img
                annotation_path = img_path.replace('image', 'image_annotation').replace('jpg', 'json')
                with open(annotation_path, 'r') as f:
                    annotation = json.loads(f.read())
                    if annotation['annotations'][0]['display'] == 1:
                        display_path_list.append(img_path)

    with open('dataset/display_list.txt', 'w') as f:
        for item_path in display_path_list:
            f.write(item_path + '\n')

    with open('dataset/white_list.txt', 'w') as f:
        for item_path in white_path_list:
            f.write(item_path + '\n')

    with open('dataset/gt_list.txt', 'w') as f:
        for item_path in gt_path_list:
            f.write(item_path + '\n')


def get_pair(clean_path_list, all_path_list, clean_list, all_list):
    positive_pair_list = {}
    negative_pair_list = []
    clean_to_all_dict = {}

    if os.path.exists('dataset/clean_to_all_dict.json'):
        with open('dataset/clean_to_all_dict.json', 'r') as f:
            clean_to_all_dict = json.loads(f.read())
    else:
        for (i, clean_item) in enumerate(clean_list):
            for (j, all_item) in enumerate(all_list):
                if all_item == clean_item:
                    clean_to_all_dict[clean_path_list[i]] = all_path_list[j]
        with open('dataset/clean_to_all_dict.json', 'w') as f:
            json.dump(clean_to_all_dict, f)

    for (i, clean_item) in enumerate(clean_list):
        if clean_item in all_list:
            all_item_path = clean_to_all_dict[clean_path_list[i]]
            all_img_dict = {}
            positive_img_path1 = ''
            positive_img_path2 = ''
            for img in os.listdir(all_item_path):
                if '.jpg' not in img:
                    continue
                img_path = all_item_path + '/' + img
                img_size = os.path.getsize(img_path)
                all_img_dict[img_path] = img_size
            for img in os.listdir(clean_path_list[i]):
                if '.jpg' not in img:
                    continue
                img_path = clean_path_list[i] + '/' + img
                img_size = os.path.getsize(img_path)
                # find filtered clean image in original all images
                for key, value in all_img_dict.items():
                    if value == img_size:
                        if positive_img_path1 == '':
                            positive_img_path1 = key
                            break
                        else:
                            positive_img_path2 = key
                            break
                if positive_img_path1 != '' and positive_img_path2 == '':
                    continue

                # collect positive pair and negative pair
                positive_pair_list[positive_img_path1] = positive_img_path2
                if positive_img_path1 == '' and positive_img_path2 == '':
                    continue
                all_img_path_list = list(all_img_dict.keys())
                all_img_path_list.remove(positive_img_path1)
                if positive_img_path2 != positive_img_path1:
                    all_img_path_list.remove(positive_img_path2)
                else:
                    print(img_path)
                for negative_img_path in all_img_path_list:
                    negative_pair_list.append((positive_img_path1, negative_img_path))
                    negative_pair_list.append((positive_img_path2, negative_img_path))
                for p, q in itertools.combinations(all_img_path_list, 2):
                    negative_pair_list.append((p, q))

    with open('dataset/positive_pair.txt', 'w') as f:
        for key, value in positive_pair_list.items():
            f.write(key + ',' + value + '\n')

    with open('dataset/negative_pair.txt', 'w') as f:
        for pair in negative_pair_list:
            f.write(pair[0] + ',' + pair[1] + '\n')


if __name__ == '__main__':
    # save_item_list()

    # cleanPathList, allPathList, cleanList, allList = get_item_list()
    # split_data(cleanPathList, allPathList, cleanList, allList)

    cleanPathList, allPathList, cleanList, allList = get_item_list()
    get_pair(cleanPathList, allPathList, cleanList, allList)
