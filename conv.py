import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def im2col(img, filter_h=3, filter_w=3):
    H, W = img.shape
    out_h = H - filter_h + 1
    out_w = W - filter_w + 1

    col = np.zeros((filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + out_h
        for x in range(filter_w):
            x_max = x + out_w
            col[y, x, :, :] = img[y:y_max, x:x_max]

    col = col.transpose(2, 3, 0, 1).reshape(out_h*out_w, -1)
    return col


directory = input("Please enter a directory to a photo, or 'q' to quit: ")

while directory != 'q':
    print("directory:", directory)

    try:
        with Image.open(directory) as f:
            ori = np.asarray(f)
            print("image size:", ori.shape[:-1])
            print("converting to gray scale image...")
            img = np.asarray(f.convert('L'))
    except FileNotFoundError:
        print("No such file or directory.")
        directory = input("Please enter a directory to a photo, or 'q' to quit: ")
        continue

    print("running the convolution algorithm...")

    filter = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
    print("default filter:", filter, sep='\n')
    answer = input("Do you want to change the filter? (y/N) ").lower()
    if answer == 'y':
        try:
            height = int(input("rows: "))
            width = int(input("cols: "))
            filter = np.zeros((height, width))
            for i in range(height):
                for j in range(width):
                    filter[i, j] = int(input("(%d, %d): " % (i, j)))
            print("filter:")
            print(filter)
        except:
            print("An error occurred!")
            directory = input("Please enter a directory to a photo, or 'q' to quit: ")
            continue

    col = im2col(img, *filter.shape)

    out = np.dot(col, filter.flatten()).reshape(img.shape[0] - filter.shape[0] + 1, img.shape[1] - filter.shape[1] + 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    for ax in ax1, ax2, ax3:
        ax.set_xticks([])
        ax.set_yticks([])

    ax1.imshow(ori)
    ax1.set_title("original image")
    ax2.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)
    ax2.set_title("gray scale image")
    ax3.imshow(out, cmap=plt.cm.gray, vmin=0, vmax=255)
    ax3.set_title("after the convolution algorithm")
    plt.show()

    directory = input("Please enter a directory to a photo, or 'q' to quit: ")
