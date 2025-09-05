# save as: view_oct5k_pairs.py
# usage:
#   python view_oct5k_pairs.py --root /path/to/OCT5k --split train
# optional:
#   --resize 348 348

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import torch
from torchvision import transforms

from OCT5k import OCT5kDataset  # import the class from the previous snippet


def make_transforms(opt_resize):
    """
    Default transforms:
      - image -> (1,H,W) float [0,1]
      - mask  -> (H,W) Long with class ids
    If --resize is given, applies bilinear/nearest appropriately.
    """
    if opt_resize is None:
        img_tf = None
        msk_tf = None
    else:
        h, w = opt_resize
        img_tf = transforms.Compose([
            transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),  # (1,H,W) float in [0,1] given 'L' input
        ])
        msk_tf = transforms.Compose([
            transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor(),                 # (1,H,W) uint8
            transforms.Lambda(lambda t: t.squeeze(0).long()),  # (H,W) Long
        ])
    return img_tf, msk_tf


class PairViewer:
    def __init__(self, dataset: OCT5kDataset):
        self.ds = dataset
        self.idx = 0

        # figure + axes
        self.fig, (self.ax_img, self.ax_msk) = plt.subplots(1, 2, figsize=(10, 5))
        self.fig.canvas.manager.set_window_title(f"OCT5k Viewer [train]")

        # add a Next button
        btn_ax = self.fig.add_axes([0.45, 0.01, 0.1, 0.06])  # [left, bottom, width, height] in figure coords
        self.next_button = Button(btn_ax, "Next")
        self.next_button.on_clicked(self.on_next)

        # connect key bindings: Right arrow / 'n' for next, 'q' to quit
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        # draw first pair
        self.im_img = None
        self.im_msk = None
        self.update_display()

        # tidy layout
        plt.tight_layout(rect=[0, 0.08, 1, 1])  # leave space for the button

    def fetch(self, idx):
        sample = self.ds[idx]
        if self.ds.return_filename:
            img, msk, stem = sample
        else:
            img, msk = sample
            stem = f"index_{idx}"
        # img: (1,H,W) or (C,H,W) tensor; msk: (H,W) Long
        if isinstance(img, torch.Tensor):
            img_np = img.squeeze(0).cpu().numpy() if img.ndim == 3 else img.cpu().numpy()
        else:
            # if a PIL sneaks through (shouldn't, given defaults), convert to numpy gray
            import numpy as np
            img_np = np.array(img, dtype=float) / 255.0

        if isinstance(msk, torch.Tensor):
            msk_np = msk.cpu().numpy()
        else:
            import numpy as np
            msk_np = np.array(msk)

        return img_np, msk_np, stem

    def update_display(self):
        img_np, msk_np, stem = self.fetch(self.idx)

        # image
        if self.im_img is None:
            self.im_img = self.ax_img.imshow(img_np, cmap="gray", interpolation="nearest")
            self.ax_img.set_title(f"Image: {stem}")
            self.ax_img.axis("off")
        else:
            self.im_img.set_data(img_np)
            self.ax_img.set_title(f"Image: {stem}")

        # mask
        if self.im_msk is None:
            self.im_msk = self.ax_msk.imshow(msk_np, interpolation="nearest")  # categorical; default colormap ok
            self.ax_msk.set_title("Mask (class IDs)")
            self.ax_msk.axis("off")
        else:
            self.im_msk.set_data(msk_np)

        self.fig.canvas.draw_idle()

    def on_next(self, event=None):
        self.idx = (self.idx + 1) % len(self.ds)
        self.update_display()

    def on_key(self, event):
        if event.key in ("right", "n"):
            self.on_next()
        elif event.key in ("q", "escape"):
            plt.close(self.fig)


def main():
    parser = argparse.ArgumentParser(description="View OCT5k image/mask pairs with a Next button.")
    parser.add_argument("--root", type=Path, required=True, help="Path to OCT5k root containing split folders.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"], help="Dataset split.")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("H", "W"), help="Optional resize, e.g., --resize 348 348")
    args = parser.parse_args()

    img_tf, msk_tf = make_transforms(args.resize)

    ds = OCT5kDataset(
        root=args.root,
        split=args.split,
        img_transform=img_tf,
        msk_transform=msk_tf,
        return_filename=True,
        strict=False,  # ignore any unpaired stragglers silently
    )

    viewer = PairViewer(ds)
    plt.show()


if __name__ == "__main__":
    main()
