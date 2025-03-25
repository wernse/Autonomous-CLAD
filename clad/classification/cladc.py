from clad.classification.cladc_utils import *
import os
from torch.utils.data import ConcatDataset
#


def get_cladc_train(root: str, transform: Callable = None, img_size: int = 64, avalanche=False) \
        -> Sequence[CladClassification]:
    """
    Returns a sequence of training sets that are chronologically ordered, defined as in the ICCV '21 challenge.

    :param root: root path to the dataset
    :param transform: a callable transformation for the data images
    :param img_size: the width/height of the images, default is 64 by 64.
    :param avalanche: If true, this will return AvalancheDataset objects.
    """
    task_dicts = [{'date': '20191111', 'period': 'Daytime'},
                  {'date': '20191111', 'period': 'Night'},
                  {'date': '20191117', 'period': 'Daytime'},
                  {'date': '20191117', 'period': 'Night'},
                  {'date': '20191120', 'period': 'Daytime'},
                  {'date': '20191121', 'period': 'Night'}, ]

    match_fn = (create_match_dict_fn(td) for td in task_dicts)

    # All training images are part of the original validation set of SODA10M.
    annot_file = os.path.join(root, 'SSLAD-2D', 'labeled', 'annotations', 'instance_val.json')
    train_sets = [get_matching_classification_set(root, annot_file, mf, img_size=img_size, transform=transform) for mf
                  in match_fn]
    from collections import Counter

    for ts in train_sets:
        ts.chronological_sort()
        # label_counts = Counter()

        # for data, target, non_aug in ts:
        #     label_counts[target] += 1  # Count label occurrences
        # print(label_counts)

    # Print out the counts for each category_id
    # for cat_id, count in category_counts.items():
    #     print(f"Category ID {cat_id}: {count} instances")
    SODA_CATEGORIES = {
        0: "Pedestrain",
        1: "Cyclist",
        2: "Car",
        3: "Truck",
        4: "Tram (Bus)",
        5: "Tricycle"
    }

    d1 = {3: 1324, 2: 2498, 0: 300, 4: 157, 1: 821, 5: 57}
    n1 = {2: 721, 3: 400, 4: 33}
    d2 = {2: 3066, 3: 2051, 0: 972, 1: 442, 4: 190, 5: 21}
    n2 = {2: 1722, 3: 744, 4: 94}
    d3 = {2: 2939, 3: 895, 4: 367, 0: 259, 1: 55, 5: 2}
    n3 = {2: 1235, 1: 397, 0: 303, 4: 106, 3: 77, 5: 1}


    vd1 = {3: 4597, 2: 16254, 4: 1186, 0: 3199, 1: 4611, 5: 203}
    vn1 = {3: 115, 2: 2179, 4: 162, 0: 343, 1: 115, 5: 3}
    return train_sets


def get_cladc_val(root: str, transform: Callable = None, img_size: int = 64, avalanche=False) -> CladClassification:
    """
    Returns the default validation set of the ICCV '21 benchmark.
    """

    def day_val_match_fn_1(obj_id, img_dic, obj_dic):
        img_annot = img_dic[obj_dic[obj_id]['image_id']]
        date = img_annot["date"]
        result = img_annot['period'] == "Daytime" and not (date == "20191120" or date == "20191117" or date == "20191111" or
                    (date == "20191121" and img_annot['period'] == "Night"))
        return result

    def night_val_match_fn_1(obj_id, img_dic, obj_dic):
        img_annot = img_dic[obj_dic[obj_id]['image_id']]
        date = img_annot["date"]
        result = img_annot['period'] == "Night" and not (date == "20191120" or date == "20191117" or date == "20191111" or
                    (date == "20191121" and img_annot['period'] == "Night"))
        return result

    def val_match_fn_1(obj_id, img_dic, obj_dic):
        img_annot = img_dic[obj_dic[obj_id]['image_id']]
        date = img_annot["date"]
        result = not (date == "20191120" or date == "20191117" or date == "20191111" or
                    (date == "20191121" and img_annot['period'] == "Night"))
        return result

    def val_match_fn_2(obj_id, img_dic, obj_dic):
        img_annot = img_dic[obj_dic[obj_id]['image_id']]
        time = img_annot['time']
        date = img_annot["date"]
        label = obj_dic[obj_id]['category_id']
        return label or (label and date == "20181015" and (time == '152030' or time == '160000'))

    annot_file_1 = os.path.join(root, 'SSLAD-2D', 'labeled', 'annotations', 'instance_val.json')
    annot_file_2 = os.path.join(root, 'SSLAD-2D', 'labeled', 'annotations', 'instance_train.json')

    dataset_day = get_matching_classification_set(root, annot_file_1, day_val_match_fn_1, img_size=img_size, transform=transform)
    dataset_night = get_matching_classification_set(root, annot_file_1, night_val_match_fn_1, img_size=img_size, transform=transform)

    dataset_2 = get_matching_classification_set(root, annot_file_2, val_match_fn_2, img_size=img_size, transform=transform)

    val_set_day = ConcatDataset([dataset_day,dataset_2])

    val_set = [
        val_set_day,
        dataset_night
    ]

    return val_set


def get_cladc_test(root: str, transform=None, img_size: int = 64, avalanche=False):
    """
    Returns the full test set of the CLAD-C benchmark. Some domains are overly represented, so for a fair
    evaluation see get_cladc_domain_test.
    """

    annot_file = os.path.join(root, 'SSLAD-2D', 'labeled', 'annotations', 'instance_test.json')
    test_set = get_matching_classification_set(root, annot_file, lambda *args: True, img_size=img_size,
                                               transform=transform)

    if avalanche:
        from avalanche.benchmarks.utils import AvalancheDataset
        return [AvalancheDataset(test_set)]
    else:
        return test_set


def get_cladc_domain_test(root: str, transform: Callable = None, img_size: int = 64, avalanche=False):
    """
    Returns fine-grained domain sets of the test set for each available combination, note that not all the
    combinations of domains exist.
    """
    annot_file = os.path.join(root, 'labeled', 'annotations', 'instance_test.json')
    test_sets = get_cladc_domain_sets(root, annot_file, ["period", "weather", "city", "location"],
                                      img_size=img_size, transform=transform)

    if avalanche:
        from avalanche.benchmarks.utils import AvalancheDataset
        return [AvalancheDataset(ts) for ts in test_sets if len(ts) > 0]
    else:
        return [ts for ts in test_sets if len(ts) > 0]


def cladc_avalanche(root: str, train_trasform: Callable = None, test_transform: Callable = None, img_size: int = 64):
    """
    Creates an Avalanche benchmark for CLADC, with the default Avalanche functinalities.
    """
    from avalanche.benchmarks.scenarios.generic_benchmark_creation import create_multi_dataset_generic_benchmark

    train_sets = get_cladc_train(root, train_trasform, img_size, avalanche=True)
    test_sets = get_cladc_val(root, test_transform, img_size, avalanche=True)

    return create_multi_dataset_generic_benchmark(train_datasets=train_sets, test_datasets=test_sets)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Rectangle
    import torch
    import os
    from PIL import Image


    def show_image_with_bounding_box(train_set, index=0):
        print("hello")
        # 17 for night
        # 120, 195 for day
        for i in [17, 120, 195]:
            # Extract the image, label, and metadata
            img, label, metadata = train_set[index]
            img_ann = train_set.img_annotations.get(i)  # Get image annotations
            file_name = img_ann.get('file_name')
            root = '/raid/wernsen/clad/data/SSLAD-2D/labeled/val'

            # Read the image file
            img_path = os.path.join(root, file_name)
            img = Image.open(img_path).convert("RGB")

            # Retrieve objects related to the image
            objs = [v for k, v in train_set.obj_annotations.items() if v.get('image_id') == img_ann.get('id')]

            # Plot the image
            fig, ax = plt.subplots(1)
            ax.imshow(img)
            #
            # # Draw the bounding boxes
            # for obj in objs:
            #     bbox = obj['bbox']  # Format: [x, y, width, height]
            #     x, y, width, height = bbox
            #     rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            #     ax.add_patch(rect)

            # Add title with label
            plt.axis("off")
            # plt.savefig(f'../../figures/no_bb_img{i}.pdf', bbox_inches='tight')
            plt.show()


    root = '/raid/wernsen/clad/data'  # Update this to your dataset's root path
    train_set = get_cladc_train(root, transform=None)
    train_set = train_set[0]
    # Show the first image with its bounding box
    show_image_with_bounding_box(train_set, index=0)

    root = '/raid/wernsen/clad/data'  # Update this to your dataset's root path
    ds = get_cladc_train(root, transform=lambda x: x)[1]

    img_index = 2

    img_annot = ds.img_annotations[img_index]
    img_id = img_annot['id']
    obj_ids = [obj for obj in ds.obj_annotations if ds.obj_annotations[obj]['image_id'] == img_id]

    fig, ax = plt.subplots(1, 1, figsize=(19, 11))
    img = Image.open(os.path.join(ds.img_folder, img_annot['file_name']))
    ax.imshow(img)



    for obj in obj_ids:
        bbox = ds.obj_annotations[obj]['bbox']
        ax.add_patch(Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, lw=2, edgecolor='r'))

    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()






    SODA_CATEGORIES = {
        0: "Pedestrain",
        1: "Cyclist",
        2: "Car",
        3: "Truck",
        4: "Tram (Bus)",
        5: "Tricycle"
    }

    ax_h, ax_w = 10, 10
    start_idx = 100
    fig, axes = plt.subplots(ax_w, ax_h, figsize=(2 * ax_h, 2 * ax_w))
    iter_axes = iter(axes.flatten())

    for i in range(start_idx, start_idx + ax_w * ax_h):
        ax = next(iter_axes)
        img, label, non_aug = ds[i]
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(SODA_CATEGORIES[label])

    plt.show()


    # ####### ####### ####### ####### ####### ####### ######


    import random
    import matplotlib.pyplot as plt

    root = '/raid/wernsen/clad/data'  # Update this to your dataset's root path
    ds = get_cladc_train(root, transform=lambda x: x)[1]

    SODA_CATEGORIES = {
        0: "Pedestrain",
        1: "Cyclist",
        2: "Car",
        3: "Truck",
        4: "Tram (Bus)",
        5: "Tricycle"
    }

    # Create the figure and axes
    # Rows = number of categories, columns = 4
    n_categories = len(SODA_CATEGORIES)

    def get_4_images_per_category_random(ds, categories):
        """
        Returns a dictionary with exactly 4 images (and associated labels) per category,
        chosen randomly from the entire dataset.
        """
        # Make a dict to hold up to 4 images for each category
        images_dict = {cat_idx: [] for cat_idx in categories.keys()}

        # Shuffle the entire dataset indices
        all_indices = list(range(len(ds)))
        random.shuffle(all_indices)

        # Collect images until we have 4 for each category
        for idx in all_indices:
            img, label, non_aug = ds[idx]

            # Only add if we still need more images for that label
            if len(images_dict[label]) < 4:
                images_dict[label].append(img)

            # If we've collected 4 images for every category, we can stop
            if all(len(imgs) == 4 for imgs in images_dict.values()):
                break

        return images_dict


    # Example usage:
    SODA_CATEGORIES = {
        0: "Pedestrian",
        1: "Cyclist",
        2: "Car",
        3: "Truck",
        4: "Tram (Bus)",
        5: "Tricycle"
    }

    # Grab a new random set of 4 images per category
    images_dict = get_4_images_per_category_random(ds, SODA_CATEGORIES)
    fig, axes = plt.subplots(
        nrows=n_categories,
        ncols=4,
        figsize=(4 * 2, n_categories * 2)  # feel free to adjust
    )

    # Plot each category in one row
    for row_idx, (cat_idx, cat_images) in enumerate(images_dict.items()):
        for col_idx, img in enumerate(cat_images):
            ax = axes[row_idx][col_idx]

            # Convert image to 64x64 using PIL
            # (assuming img is a NumPy array or a PIL image)
            pil_img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
            img_resized = pil_img.resize((64, 64))

            ax.imshow(img_resized)
            ax.set_xticks([])
            ax.set_yticks([])

            # Put the category name on the left side of each row
            if col_idx == 0:
                ax.set_ylabel(SODA_CATEGORIES[cat_idx], fontsize=10)

    plt.tight_layout()
    plt.show()
