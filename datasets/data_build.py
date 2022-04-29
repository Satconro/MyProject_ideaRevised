import os
import random
import shutil

def random_pick():
    dir_root = r"C:\Luoshimin\8.Working Directory\dataset\EUVP"
    relative = r"Paired\underwater_scenes\validation"
    target_dir = r"C:\Luoshimin\8.Working Directory\dataset\test\EUVP"
    img_pick = 50

    all_imgs = os.listdir(os.path.join(dir_root, relative))
    img_to_pick = random.sample(all_imgs, img_pick if len(all_imgs) > img_pick else len(all_imgs))
    if not os.path.exists(os.path.join(target_dir, relative)):
        os.makedirs(os.path.join(target_dir, relative))
    for img in img_to_pick:
        shutil.copyfile(os.path.join(dir_root, relative, img), os.path.join(target_dir, relative, img))

def pick_from_RUIE():
    dir = r"C:\Luoshimin\8.Working Directory\dataset\RUIE\UIQS"
    target_dir = r"C:\Luoshimin\8.Working Directory\dataset\test\RUIE\UIQS"
    img_pick = 50

    subsets = os.listdir(dir)
    for sub in subsets:
        all_imgs = os.listdir(os.path.join(dir, sub))
        img_to_pick = random.sample(all_imgs, img_pick if len(all_imgs) > img_pick else len(all_imgs))
        if not os.path.exists(os.path.join(target_dir, sub)):
            os.makedirs(os.path.join(target_dir, sub))
        for img in img_to_pick:
            shutil.copyfile(os.path.join(dir, sub, img), os.path.join(target_dir, sub, img))
        print("{} is done".format(sub.split(r"\\")[-1]))

def pick_from_RUIE_2():
    dir = r"C:\Luoshimin\8.Working Directory\dataset\EUVP\Paired"
    target_dir = r"C:\Luoshimin\8.Working Directory\dataset\test\EUVP"
    img_pick = 50

    subsets = os.listdir(dir)
    for sub in subsets:
        all_imgs = os.listdir(os.path.join(dir, sub))
        img_to_pick = random.sample(all_imgs, img_pick if len(all_imgs) > img_pick else len(all_imgs))
        if not os.path.exists(os.path.join(target_dir, sub)):
            os.makedirs(os.path.join(target_dir, sub))
        for img in img_to_pick:
            shutil.copyfile(os.path.join(dir, sub, img), os.path.join(target_dir, sub, img))
        print("{} is done".format(sub.split(r"\\")[-1]))

def pick_from_EUVP():
    dir = r"C:\Luoshimin\8.Working Directory\underwater_image_enhancement\dataset\2. EUVP\Paired"
    target_dir = r"C:\Luoshimin\8.Working Directory\underwater_image_enhancement\My_Dataset\Synthetic_Underwater_Image"
    img_pick = 700

    subsets = os.listdir(dir) # dark、image、scenes
    for sub in subsets:
        clear = "trainB"
        distorted = "trainA"
        # Choose imgs from trainB
        all_clear_imgs = os.listdir(os.path.join(dir, sub, clear))
        img_to_pick = random.sample(all_clear_imgs, img_pick if len(all_clear_imgs) > img_pick else len(all_clear_imgs))
        for img in img_to_pick:
            if not os.path.exists(os.path.join(target_dir, clear)):
                os.makedirs(os.path.join(target_dir, clear))
            if not os.path.exists(os.path.join(target_dir, distorted)):
                os.makedirs(os.path.join(target_dir, distorted))
            shutil.copyfile(os.path.join(dir, sub, clear, img), os.path.join(target_dir, clear, img))
            shutil.copyfile(os.path.join(dir, sub, distorted, img), os.path.join(target_dir, distorted, img))
        print("{} is done".format(sub.split("\\")[-1]))


if __name__ == "__main__":
    random_pick()
    print("Is done")