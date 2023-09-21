import streamlit as st
import json
import pandas as pd
import cv2
import numpy as np
import math

st.set_page_config(
    page_title="My Streamlit App",
    page_icon=":chart_with_upwards_trend:",
    layout="centered",  # You can also set the layout if needed
    initial_sidebar_state="expanded",  # You can choose "expanded" or "collapsed"
)


def display_shot_data():
    # Upload JSON File
    uploaded_file = st.file_uploader("Upload a JSON file", type=["json"])

    # Check if a file is uploaded
    if uploaded_file is not None:
        # Read JSON data
        data = json.load(uploaded_file)

        # Convert the data to a DataFrame (optional)
        df = pd.json_normalize(data)

        # Display the DataFrame (optional)
        st.write("### Data Preview")
        df_copy = df.copy()
        columns_to_remove = ['signal1','signal2', 'signal3', 'beforeCross1', 'beforeCross2', 'beforeCross3']

        # Remove the specified columns from the copy
        df_copy.drop(columns=columns_to_remove, inplace=True)
        st.table(df_copy)
        

        # Interactive Chart 1 - Scatter Plot
        st.write("### Interactive Scatter Plot")
        if df.empty:
            st.write("No data to plot. Please upload a valid JSON file.")
        else:
            data = {'Signal1': df['signal1'][0], 'Signal2': df['signal2'][0], 'Signal3': df['signal3'][0]}
            chart_data = pd.DataFrame(data)

            st.line_chart(chart_data, color=[ "#ff0000","#ffaa00", "#008000"])

        # Interactive Chart 2 - Bar Chart
        st.write("### Interactive Bar Chart")
        if df.empty:
            st.write("No data to plot. Please upload a valid JSON file.")
        else:
            data = {'Signal1': df['beforeCross1'][0], 'Signal2': df['beforeCross2'][0], 'Signal3': df['beforeCross3'][0]}
            chart_data = pd.DataFrame(data)

            st.line_chart(chart_data, color=["#ff0000","#ffaa00", "#008000"])


def get_coordinates_data():
    
    def get_circle(image):
        # Apply blurring, thresholding, and closing to make the circles rounder and hollow

        _, thresh = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find the contours of the circles
        contours, _ = cv2.findContours(
            close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize variables to track the largest circle
        maxRadius = 0
        maxCenter = (0, 0)

        for cnt in contours:
            # Fit an enclosing circle to the contour
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            if radius > maxRadius:
                maxRadius = radius
                maxCenter = center

        # Calculate the ROI coordinates
        x, y = maxCenter[0] - maxRadius, maxCenter[1] - maxRadius
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        w, h = 2 * maxRadius, 2 * maxRadius

        # Extract the ROI from the image
        roi1 = image.copy()[y:y + h, x:x + w]
        

        cv2.circle(image, maxCenter, maxRadius, (0, 255, 0), 3)
        dot_color = (0, 255, 0)  # Red
        dot_radius = 50
        cv2.circle(image, maxCenter, dot_radius, dot_color, 3)
        roi2 = image[y:y + h, x:x + w]
        maxCenter = (maxCenter[0] - x, maxCenter[1] - y)
        return roi1, roi2, maxCenter, maxRadius

    def apply_blur(image, blur_type, kernel_size):
        if blur_type == 'Gaussian':
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif blur_type == 'Median':
            return cv2.medianBlur(image, kernel_size)
    # Function to process the ROI
    def find_blobs(roi):
        # Apply thresholding to the ROI
        retval, threshold = cv2.threshold(roi, 200, 250, cv2.THRESH_BINARY)
        inverted = cv2.bitwise_not(threshold)

        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 0
        params.maxThreshold = 255

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 20

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.1

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01

        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            detector = cv2.SimpleBlobDetector(params)
        else:
            detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(inverted)

        for keypoint in keypoints:
            keypoint.size *= 10  # You can adjust the factor as needed

        # Draw detected blobs as red circles.
        im_with_keypoints = cv2.drawKeypoints(inverted, keypoints, np.array([]), (0, 0, 255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return im_with_keypoints, keypoints

    def get_xy_from_blobs(points, image):
        diameter = 210  # measured in mm
        height, width = image.shape[:2]  # Extract height and width from the image shape
        size = math.ceil(height / 1000)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        xy_points = []
        for point in points:
            real_x = round((point.pt[0] / width) * diameter, 1)  # Round to one decimal place
            real_y = round((abs(height - point.pt[1]) / height) * diameter, 1)  # Round to one decimal place
            xy_points.append((real_x, real_y))
            cv2.putText(image, f"{real_x}, {real_y}", (int(point.pt[0] - size * 100), int(point.pt[1] - size * 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (0, 255, 0), int(size * 3))
            
        return image, xy_points
    
    st.title('Image Blur App with OpenCV')

    # Upload an image
    uploaded_image = st.file_uploader(
        'Upload an image', type=['jpg', 'png', 'jpeg'])

    if uploaded_image is not None:
        image_bytes = uploaded_image.read()
        image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Read the uploaded image
        height, width = image.shape[:2]
        # Display the original image
        #st.image(image, caption='Original Image', use_column_width=True)

        roi_image, with_line, center, radius = get_circle(image)

        blur_type = "Median"
        rounded_integer = round(height/200)
        # Make it an odd number (if it's even)
        kernel_size = rounded_integer + 1 if rounded_integer % 2 == 0 else rounded_integer
        # Apply blur to the image
        blurred_image = apply_blur(roi_image, blur_type, kernel_size)

        image_with_blobs, points = find_blobs(blurred_image)

        final_image, xy_points = get_xy_from_blobs(points, roi_image)
        
        st.image(final_image, caption='Coordinates of shot',
                 use_column_width=True)
        
        # Create a Streamlit app
        st.title("List of Coordinates")

        # Display the coordinates in a table
        st.table(xy_points)

if __name__ == '__main__':
    function_choice = st.selectbox("Select a function:", ["Get coordinates", "Display shot data"])
    if function_choice == "Get coordinates":
        result = get_coordinates_data()
    elif function_choice == "Display shot data":

        result = display_shot_data()
    
