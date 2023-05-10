import numpy as np
import matplotlib.pyplot as plt
import gc


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    del mask
    gc.collect()


# def show_masks_on_image(raw_image, masks):
#     plt.imshow(raw_image, cmap='gray')
#     ax = plt.gca()
#     ax.set_autoscale_on(False)
#     for mask in masks:
#         show_mask(mask, ax=ax, random_color=True)
#     plt.axis("off")
#     plt.show()
#     del mask
#     gc.collect()


def show_masks_on_image(raw_image, masks):
    plt.imshow(raw_image, cmap='gray')
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for mask in masks:
        show_mask(mask, ax=ax, random_color=True)
    plt.axis("off")
    plt.gcf().canvas.draw()
    final_image = np.array(plt.gcf().canvas.renderer.buffer_rgba())
    plt.close()
    del mask
    gc.collect()
    return final_image


# def show_points(image, coords, labels, ax, marker_size=375):
#     plt.imshow(image)
#     pos_points = coords[labels == 1]
#     neg_points = coords[labels == 0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
#     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_points(image, coords, labels, ax, marker_size=375):
    fig, ax = plt.subplots()
    ax.imshow(image)
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    fig.canvas.draw()
    image_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_data = image_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image_data
