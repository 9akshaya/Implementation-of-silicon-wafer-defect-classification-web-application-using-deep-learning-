from PIL import Image

# Open the grayscale image
gray_image = Image.open("C:\\Users\\ADMIN\\Desktop\\flask project\\y186LSWMD.png")

# Convert the grayscale image to RGB mode
colored_image = gray_image.convert('RGB')

# Save the colored image
colored_image.save("C:\\Users\\ADMIN\\Desktop\\flask project\\ycak186LSWMD.png")
