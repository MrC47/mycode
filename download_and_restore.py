import os
from tqdm import tqdm
from datasets import load_dataset

def download_and_restore_pacs(cache_dir = './datasets_cache', save_dir = './mydatasets/pacs'):

    pacs = load_dataset(path = 'flwrlabs/pacs',
                        cache_dir = cache_dir)

    label_name = pacs['train'].features['label'].names
    domain_name = ['photo','art_painting','cartoon','sketch']

    for i, item in enumerate(tqdm(pacs['train'],desc = '正在还原图片')):
        img = item['image']

        l_name = label_name[item['label']]
        for j in range(4):
            if item['domain'] == domain_name[j]:
                d_name = domain_name[j]

        target_dir = os.path.join(save_dir, d_name, l_name)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        file_name = f'{i:06d}.jpg'
        img.save(os.path.join(target_dir, file_name))

def download_and_restore_vlcs(cache_dir = './datasets_cache', save_dir = './mydatasets/vlcs'):
    pass

if __name__ == '__main__':
    download_and_restore_pacs()