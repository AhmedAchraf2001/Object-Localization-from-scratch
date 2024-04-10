def plot_bb(image, gt_coords, pred_coords=[], norm=False):
    if norm:
      image*= 255.   # Denormalization
      image = image.astype('uint8')
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    row, col = gt_coords
    row*= 144
    col*= 144
    draw.rectangle((col, row, col+52, row+52), outline='green', width=2)

    if len(pred_coords) == 2:
       row, col = pred_coords
       row*= 144
       col*= 144
       draw.rectangle((col, row, col+52, row+52), outline='red', width=2)
    return image