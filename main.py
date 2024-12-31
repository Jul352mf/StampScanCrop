import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import streamlit as st

def load_image(image_path):
    """Load an image from a specified path."""
    print("Loading the image...")
    return cv2.imread(image_path)

def find_combined_roi(image, padding=10):
    """Find a single combined ROI for all stamps in the image with a fallback for small regions."""
    print("Finding combined ROI...")
    # Use adaptive threshold to enhance edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Find contours on the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return None

    # Calculate contour areas
    contour_areas = [cv2.contourArea(contour) for contour in contours]
    if not contour_areas:
        print("No valid contours found.")
        return None

    # Filter out extremely small contours (less than 0.5% of the image area)
    image_area = image.shape[0] * image.shape[1]
    min_valid_area = image_area * 0.005
    filtered_contours = [contour for contour, area in zip(contours, contour_areas) if area >= min_valid_area]

    # If no contours pass the filter, fallback to the largest contour
    if not filtered_contours:
        print("No valid contours remain after filtering. Using fallback to largest contour.")
        largest_contour = contours[np.argmax(contour_areas)]
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (max(0, x - padding), max(0, y - padding), min(image.shape[1], x + w + padding), min(image.shape[0], y + h + padding))

    # Combine valid contours into one bounding box
    x_min, y_min = image.shape[1], image.shape[0]
    x_max, y_max = 0, 0

    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    # Apply padding and ensure it stays within bounds
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image.shape[1], x_max + padding)
    y_max = min(image.shape[0], y_max + padding)

    return (x_min, y_min, x_max, y_max)

def crop_to_roi(image, roi):
    """Crop the image to the ROI."""
    print("Cropping to ROI...")
    x_min, y_min, x_max, y_max = roi
    return image[y_min:y_max, x_min:x_max]

def refine_cropped_image(cropped_image):
    """Remove excess black space after cropping."""
    print("Refining cropped image...")
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Find contours of the remaining content
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No content found after refinement.")
        return cropped_image

    # Get the bounding box of the content
    x_min, y_min = cropped_image.shape[1], cropped_image.shape[0]
    x_max, y_max = 0, 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    return cropped_image[y_min:y_max, x_min:x_max]

def add_border(image, border_size=5):
    """Add a fixed-size border to the image."""
    print("Adding border...")
    return cv2.copyMakeBorder(
        image, border_size, border_size, border_size, border_size,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

def process_stamps(image_path):
    print(f"Processing: {image_path}")
    image = load_image(image_path)
    if image is None:
        print("Image loading failed.")
        return

    # Step 1: Find combined ROI
    roi = find_combined_roi(image)
    if roi is None:
        print("No valid ROI found.")
        return

    # Step 2: Crop the image to the ROI
    cropped_image = crop_to_roi(image, roi)

    # Step 3: Refine the cropped image
    refined_image = refine_cropped_image(cropped_image)

    # Step 4: Add border
    final_image = add_border(refined_image)



    # Build and save final path
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(f"{base_name}_cropped.jpg")
    cv2.imwrite(output_path, final_image)
    print(f"Saved cropped image to: {output_path}")

    # IMPORTANT: Return the path so Streamlit knows it succeeded
    return output_path


##########################
# Streamlit App
##########################
st.title("Stamp Processing App")


# File uploader
uploaded_file = st.file_uploader("Upload a stamp image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image in memory
    file_bytes = uploaded_file.read()
    # Use np.frombuffer in newer versions of NumPy
    image_np = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not read the uploaded image. Please try again.")
    else:
        # Temporary save for processing with existing module logic
        temp_input_path = "temp_input.jpg"
        with open(temp_input_path, "wb") as f:
            f.write(file_bytes)

        # Run the full pipeline
        output_path = process_stamps(temp_input_path)

        if output_path is None:
            st.error("No valid ROI found or image processing failed.")
        else:
            st.success(f"Processed image saved to: {output_path}")

            # Display final image in the Streamlit app
            final_image = cv2.imread(output_path)
            # Convert BGR -> RGB for display in Streamlit
            final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
            st.image(final_image_rgb, caption="Processed Stamp")

            # Optional: Provide a download button for final image
            with open(output_path, "rb") as f:
                btn = st.download_button(
                    label="Download Processed Image",
                    data=f,
                    file_name=os.path.basename(output_path),
                    mime="image/jpeg"
                )
