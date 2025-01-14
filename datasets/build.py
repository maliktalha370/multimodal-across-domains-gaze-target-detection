import os

from torch.utils.data import DataLoader

from .GazeFollow import GazeFollow
from .GOO import GOO
from .VideoAttentionTarget import VideoAttentionTargetImages


def get_loader(name: str, root_dir: str,infer=False,  input_size=224, output_size=64, batch_size=48, num_workers=6, is_train=True):

    if name == "gazefollow":
        labels = os.path.join(root_dir, "train_annotations_elm.txt" if is_train else "val_annotations_elm.txt")
        dataset = GazeFollow(root_dir, labels, infer, input_size=input_size, output_size=output_size, is_test_set=not is_train)
        loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=is_train, num_workers=0, pin_memory=True
        )
    elif name == "videoattentiontarget":
        labels = os.path.join(root_dir, "annotations/train" if is_train else "annotations/test")
        dataset = VideoAttentionTargetImages(
            root_dir, labels, infer,  input_size=input_size, output_size=output_size, is_test_set=not is_train
        )
        loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers, pin_memory=True
        )
    elif name == "goo":
        labels = os.path.join(root_dir, "..", "oneshotrealhumansNew.pickle" if is_train else "testrealhumansNew.pickle")
        dataset = GOO(root_dir, labels, input_size=input_size, output_size=output_size, is_test_set=not is_train)
        loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers, pin_memory=True
        )
    else:
        raise ValueError(f"Invalid dataset: {name}")

    return loader


def get_dataset(config, infer=True):
    source_loader = get_loader(
        config.source_dataset,
        config.source_dataset_dir,
        input_size=config.input_size,
        output_size=config.output_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        is_train=True,

    )

    target_loader = get_loader(
        config.target_dataset,
        config.target_dataset_dir,
        input_size=config.input_size,
        output_size=config.output_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        is_train=True,
    )

    target_test_loader = get_loader(
        config.target_dataset,
        config.target_dataset_dir,
        infer,
        input_size=config.input_size,
        output_size=config.output_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        is_train=False,
    )

    return source_loader, target_loader, target_test_loader
