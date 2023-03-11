import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def im2col(img, filter_h=3, filter_w=3, stride=1):
    H, W = img.shape
    out_h = (H - filter_h)//stride + 1
    out_w = (W - filter_w)//stride + 1

    col = np.zeros((filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[y, x, :, :] = img[y:y_max:stride, x:x_max:stride]

    col = col.transpose(2, 3, 0, 1).reshape(out_h*out_w, -1)
    return col


length = 3
weights = np.array([[128, 64, 32],
                    [1, 0, 16],
                    [2, 4, 8]])

directory = input("Please enter a directory to a photo, or 'q' to quit: ")

while directory.lower() != 'q':
    print("directory: %s" % directory)

    try:
        with Image.open(directory) as im:
            original = np.asarray(im)
            print("image size:", original.shape[:-1])
            print("converting to grayscale image...")
            data = np.asarray(im.convert("L"))
    except FileNotFoundError:
        print("No such file or directory.")
        directory = input("Please enter a directory to a photo, or 'q' to quit: ")
        continue

    print("running the LBP algorithm...")

    col = im2col(data)
    new_img = np.dot(col > col[:, 5].reshape((-1, 1)), weights.flatten()).reshape((data.shape[0]-2, data.shape[1]-2))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))

    for ax in ax1, ax2, ax3:
        ax.set_xticks([])
        ax.set_yticks([])

    ax1.imshow(original)
    ax1.set_title("original image")
    ax2.imshow(data, cmap=plt.cm.gray, vmin=0, vmax=255)
    ax2.set_title("grayscale image")
    ax3.imshow(new_img, cmap=plt.cm.gray, vmin=0, vmax=255)
    ax3.set_title("after running the LBP algorithm")

    plt.show()
    
    directory = input("Please enter a directory to a photo, or 'q' to quit: ")
