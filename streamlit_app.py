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
        columns_to_remove = ['signal1', 'signal2', 'signal3',
                             'beforeCross1', 'beforeCross2', 'beforeCross3']

        # Remove the specified columns from the copy
        df_copy.drop(columns=columns_to_remove, inplace=True)
        st.table(df_copy)

        # Interactive Chart 1 - Scatter Plot
        st.write("### Interactive Scatter Plot")
        if df.empty:
            st.write("No data to plot. Please upload a valid JSON file.")
        else:
            data = {'Signal1': df['signal1'][0],
                    'Signal2': df['signal2'][0], 'Signal3': df['signal3'][0]}
            chart_data = pd.DataFrame(data)

            st.line_chart(chart_data, color=["#ff0000", "#ffaa00", "#008000"])

        # Interactive Chart 2 - Bar Chart
        st.write("### Interactive Bar Chart")
        if df.empty:
            st.write("No data to plot. Please upload a valid JSON file.")
        else:
            data = {'Signal1': df['beforeCross1'][0],
                    'Signal2': df['beforeCross2'][0], 'Signal3': df['beforeCross3'][0]}
            chart_data = pd.DataFrame(data)

            st.line_chart(chart_data, color=["#ff0000", "#ffaa00", "#008000"])


def get_coordinates_data(user_input, is_yellow):

    def replace_yellow(image):
        lower_yellow = np.array([10, 0, 0])
        upper_yellow = np.array([40, 255, 255])

        # Create a mask for yellow regions in the image
        yellow_mask = cv2.inRange(image, lower_yellow, upper_yellow)
        black_mask = cv2.bitwise_not(yellow_mask)
        black_background = np.zeros_like(image)
        result_image = cv2.bitwise_and(image, image, mask=black_mask)
        image = cv2.add(result_image, black_background)
        return image

    def apply_blur(image, blur_type, kernel_size):
        if blur_type == 'Gaussian':
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif blur_type == 'Median':
            return cv2.medianBlur(image, kernel_size)
    # Function to process the ROI

    def find_blobs_gray(roi):
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

    def find_blobs_yellow(image):
        # Apply thresholding to the ROI
        width, height = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        retval, threshold = cv2.threshold(gray, 100, 225, cv2.THRESH_BINARY)
        #inverted = cv2.bitwise_not(threshold)

        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 100
        params.maxThreshold = 255

        # Filter by Area.
        params.filterByArea = True
        params.minArea = int(width/1000)

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.8

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5

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
        keypoints = detector.detect(threshold)

        # Remove keypoints that are too close to each other
        min_distance = 50  # Set the minimum distance between keypoints
        filtered_keypoints = []
        for i, kp1 in enumerate(keypoints):
            is_close = False
            for j, kp2 in enumerate(keypoints[i+1:]):
                if cv2.norm(np.array(kp1.pt), np.array(kp2.pt)) < min_distance:
                    is_close = True
                    break
            if not is_close:
                filtered_keypoints.append(kp1)

        for keypoint in filtered_keypoints:
            keypoint.size *= 10  # You can adjust the factor as needed

        # Draw detected blobs as red circles.
        im_with_keypoints = cv2.drawKeypoints(gray, filtered_keypoints, np.array([]), (0, 255, 0),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return im_with_keypoints, filtered_keypoints

    def get_target_circle(colored_image):
        image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY)
        # Step 2: Apply blurring and thresholding to enhance circle shapes
        _, binary_image = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Step 3: Apply morphological closing to make circles smoother and close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

        # Step 4: Find contours of the circles in the processed image
        contours, _ = cv2.findContours(
            closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 5: Set a circularity threshold
        circularity_threshold = 0.8
        maxCenter = 0
        maxRadius = 0

        for contour in contours:
            # Step 6: Fit an enclosing circle to the contour
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            # Step 7: Calculate the contour area and the circularity
            area = cv2.contourArea(contour)

            circle_area = radius * radius * np.pi
            if area == 0 or circle_area == 0:
                continue
            circularity = area / circle_area

            # Step 8: Draw only the contours that have high circularity
            if circularity > circularity_threshold:
                if radius > maxRadius:
                    maxRadius = radius
                    maxCenter = center

        # Step 7: Calculate the ROI coordinates
        x, y = maxCenter[0] - maxRadius, maxCenter[1] - maxRadius
        x = max(0, x)
        y = max(0, y)
        w, h = 2 * maxRadius, 2 * maxRadius

        # Step 8: Extract the ROI from the image
        cropped_image = colored_image[y:y + h, x:x + w]

        # Step 9: Draw circles on the original image for visualization
        cv2.circle(image, maxCenter, maxRadius, (0, 255, 0), 3)
        dot_color = (0, 255, 0)  # Red
        dot_radius = 50
        cv2.circle(image, maxCenter, dot_radius, dot_color, 3)

        # Step 10: Extract the second ROI with circles drawn
        cropped_gray = image[y:y + h, x:x + w]

        return cropped_image, cropped_gray

    def get_xy_from_blobs(points, image):
        diameter = user_input  # measured in mm
        # Extract height and width from the image shape
        height, width = image.shape[:2]
        size = math.ceil(height / 1000)
        xy_points = []
        for point in points:
            # Round to one decimal place
            real_x = round((point.pt[0] / width) * diameter, 1)
            # Round to one decimal place
            real_y = round((abs(height - point.pt[1]) / height) * diameter, 1)
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
        colored_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        colored_image_rgb = cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB)
        height, width = colored_image.shape[:2]
        # Display the original image
        st.image(colored_image_rgb, caption='Original Image',
                 use_column_width=True)
        cropped_image, cropped_gray = get_target_circle(colored_image)
        if(is_yellow):
            hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
            replaced = replace_yellow(hsv_image)
            blurred_image = cv2.cvtColor(replaced, cv2.COLOR_HSV2BGR)
            image_with_blobs, points = find_blobs_yellow(blurred_image)
        else:
            blur_type = "Median"
            rounded_integer = round(height/200)
            # Make it an odd number (if it's even)
            kernel_size = rounded_integer + 1 if rounded_integer % 2 == 0 else rounded_integer
            # Apply blur to the image
            blurred_image = apply_blur(cropped_gray, blur_type, kernel_size)
            image_with_blobs, points = find_blobs_gray(blurred_image)

        final_image, xy_points = get_xy_from_blobs(points, cropped_image)
        final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

        st.image(final_image, caption='Coordinates of shot',
                 use_column_width=True)

        # Create a Streamlit app
        st.title("List of Coordinates")

        # Display the coordinates in a table
        st.table(xy_points)


if __name__ == '__main__':
    function_choice = st.selectbox(
        "Select a function:", ["Get coordinates", "Display shot data"])

    if function_choice == "Get coordinates":
        user_input = st.text_input("Enter diameter in mm:")
        is_yellow = st.checkbox('Yellow target')
        if(user_input):
            result = get_coordinates_data(int(user_input), is_yellow)
        else:
            result = get_coordinates_data(210, is_yellow)
    elif function_choice == "Display shot data":

        result = display_shot_data()
