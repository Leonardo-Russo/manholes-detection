import cv2

# Load an image
image_path = 'manholes_test.jpg'
image = cv2.imread(image_path)

# Check if the image was successfully loaded
if image is None:
    print("Error: Could not read image.")
else:
    # Display the image
    cv2.imshow('Image', image)
    # Wait for a key press to exit
    cv2.waitKey(0)
    # Close the image window
    cv2.destroyAllWindows()


# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize the image
resized_image = cv2.resize(gray_image, (640, 480))

# Apply Gaussian Blur
blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)



# Detect edges using Canny
edges = cv2.Canny(blurred_image, 100, 200)

# Display the edges
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

