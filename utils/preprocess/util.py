import matplotlib.pyplot as plt


def im_show(im1, im2, title):
    """
    Display two image in a figure

    :param im1: first image
    :param im2: second image
    :param title: Window name
    @author:Amit
    """
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(im1, cmap='gray')
    axes[1].imshow(im2, cmap='gray')
    fig.canvas.manager.set_window_title(title)
    fig.suptitle(title)
    plt.show()


def im_show3(im_arr, title):
    """
    Display images 3 in a row from arbitrary number of images in array
    Used for attention visualising
    :param im_arr: array of images
    :param title: Window name
    @author:Amit
    """
    fig, axes = plt.subplots(len(im_arr)//3 if len(im_arr) % 3 == 0 else len(im_arr)//3 + 1, 3)
    count = 0
    for i in range(len(im_arr)):
        axes[count // 3][count % 3].imshow(im_arr[i])
        axes[count // 3][count % 3].set_title("Head " + str(i))
        count = count + 1
    fig.canvas.manager.set_window_title(title)
    fig.suptitle(title)
    plt.show()
