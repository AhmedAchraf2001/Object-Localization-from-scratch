def create_example():
    class_id = np.random.randint(0, 9)
    image = np.ones((144, 144, 3))*255  #make it white image (blank)
    row = np.random.randint(0, 72)
    col = np.random.randint(0, 72)
    image[row: row +72, col: col+72] = np.array(emojis[class_id]['image'])
    return image.astype('uint8'), class_id,  (row+10)/144, (col+10)/144