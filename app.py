import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from fuzzywuzzy import fuzz
import numpy as np
from paddleocr import PaddleOCR


def image_alignment_original_to_annotate(img_orig,img_annot):
    gray_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    gray_annot = cv2.cvtColor(img_annot, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_orig, None)
    kp2, des2 = sift.detectAndCompute(gray_annot, None)
    matcher = cv2.FlannBasedMatcher()
    matches = matcher.match(des1, des2)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    img1_warped = cv2.warpPerspective(img_orig, H, (img_annot.shape[1], img_annot.shape[0]))
    return img1_warped

def filter_image_original_contour_detection_and_rectangels(img1_warped,img_annot):
    diff_img = cv2.absdiff(img1_warped,img_annot)
    kernel = np.array([[-1,-1,-1], 
                        [-1, 9,-1],
                        [-1,-1,-1]])
    sharpened = cv2.filter2D(diff_img, -1, kernel)
    # Define the lower and upper boundaries of the color to be thresholde
    lower_color = np.array([0, 0, 0])  # lower boundary of black color in RGB color space
    upper_color = np.array([20,20,20])  # upper boundary of black color in RGB color space

    # Threshold the image to get only the pixels within the defined color range
    mask = cv2.inRange(diff_img, lower_color, upper_color)
    mask_inv = cv2.bitwise_not(mask)
    # Create a copy of the original image
    output = sharpened.copy()
    # Define the desired color in BGR format
    color = (0, 255, 0)
    # # Assign the desired color to all non-black pixels
    output[mask_inv > 0] = color
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    denoised_image = cv2.erode(thresh, kernel)
    kernel = np.ones((3,1), np.uint8)  # note this is a horizontal kernel
    image_dilate = cv2.dilate(denoised_image, kernel, iterations=1)
    kernel = np.ones((7,1), np.uint8)
    image_erode = cv2.erode(image_dilate, kernel)  
    image_cleaned = cv2.medianBlur(image_erode, 3)
    kernel = np.ones((3,3), np.uint8)
    img_dilation = cv2.dilate(image_cleaned, kernel, iterations=2) 
    kernel = np.ones((1,11), np.uint8)
    img_cleaned = cv2.morphologyEx(img_dilation, cv2.MORPH_CLOSE, kernel)
    rectangles = []
    rect_cnts,_ = cv2.findContours(img_cleaned,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    for i in range(len(rect_cnts)):    
        x ,y, w, h = cv2.boundingRect(rect_cnts[i])
        a=w*h    
        aspectRatio = float(w)/h
        if  aspectRatio < 1.5 or a<2000:
            continue          
        approx = cv2.approxPolyDP(rect_cnts[i], 0.05* cv2.arcLength(rect_cnts[i], True), True)
        width=w
        height=h   
        start_x=x
        start_y=y
        end_x=start_x+width
        end_y=start_y+height      
        rectangles.append((start_x,start_y,end_x,end_y))
    image_dilate[img_cleaned != 0] = img_cleaned[img_cleaned != 0]   
    cnts,_ = cv2.findContours(image_dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contour_ids = [i for i in range(len(cnts)) if cv2.contourArea(cnts[i]) < 50000 and cv2.arcLength(cnts[i], True) > 400 ]
    return contour_ids,rectangles,cnts

def get_mapping_dict_rectangle_contour_original(rectangles,img_annot,contour_ids,contours,file_list):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    test_img = img_annot.copy()
    final_texts = []
    rectangle_contour_dict = {}
    for i in range(len(rectangles)):
        final_text = ''
        cropped = test_img[rectangles[i][1]:rectangles[i][3],rectangles[i][0]:rectangles[i][2]]
        result = ocr.ocr(cropped,cls=False)
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                final_text = final_text + ' ' + line[1][0]           
        rectangle_contour_dict[i] = final_text

    maskdict = {}
    for i in range(len(rectangles)):
        rectangle = rectangles[i]
        x_mid = (rectangle[0] + rectangle[2]) / 2
        y_mid = (rectangle[1] + rectangle[3]) / 2
        point = (x_mid,y_mid)
        for j in contour_ids:
            test_cnt = contours[j]        
            result = cv2.pointPolygonTest(test_cnt, point, False)
            if result > 0:
                maskdict[i] = j
                break
            
    cnts_id_remove = []
    # with open(text_to_remove, 'r') as file:
    #     lines = file.readlines()    
    
    for line in file_list:
        if line.startswith("#"):
            continue
        scores = np.array([1 if line.lower().strip()  in rectangle_contour_dict[key].lower()  else 0 for key in rectangle_contour_dict.keys() ]) 
        for i in np.where(scores == 1)[0]:
            cnts_id_remove.append(maskdict.get(i))
    return cnts_id_remove,maskdict,rectangle_contour_dict   


def get_final_image(contour_ids,contour_id_remove,img_annot,contours,img1_warped):
    src_img = img1_warped.copy()
    id_set = set(contour_ids)
    cnts_id_set = set(contour_id_remove)
    diff_set = id_set - cnts_id_set
    # cnts_id_remove
    mask = np.zeros_like(img_annot[:,:,0])
    for i in diff_set:
        test_cnt = contours[i].reshape(-1,2)
        cv2.fillPoly(mask, [test_cnt], 255)
    kernel = np.ones((3,3), np.uint8)
    dilated_contour = cv2.dilate(mask, kernel)
    result_img = cv2.bitwise_and(img_annot, img_annot, mask=dilated_contour)
    src_img[dilated_contour != 0] = result_img[dilated_contour != 0]    
    # cv2.imwrite(output_path,src_img)
    return src_img      




# Allow the user to upload multiple files


st.set_page_config(layout="wide")
st.set_option('deprecation.showfileUploaderEncoding', False)


# Display images in three columns
col1, col2, col3 = st.columns(3)

col1.width = 0.3
col2.width = 0.3
col3.width = 0.3

# If the user uploaded any files
with col1:
    uploaded_file = st.file_uploader("Choose Base Image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    if uploaded_file is not None:
        # Read the uploaded image using OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_orig = cv2.imdecode(file_bytes, 1)
        img = Image.open(uploaded_file)
        st.image(img, use_column_width=True)

with col2:
    uploaded_file1 = st.file_uploader("Choose Annotated Image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    # If the user uploaded any files
    if uploaded_file1 is not None:
        # Read the uploaded image using OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file1.read()), dtype=np.uint8)
        img_annot = cv2.imdecode(file_bytes, 1)
        img1 = Image.open(uploaded_file1)
        st.image(img1, use_column_width=True)

with col3:
    # Allow the user to input the file contents as a string
    file_contents = st.text_area("Enter file contents:",height=20)

    # Convert the string to a list
    file_list = file_contents.split("\n")
    if uploaded_file1 is not None and uploaded_file1 is not None:
        file_list = file_contents.split("\n")
        img1_warped = image_alignment_original_to_annotate(img_orig,img_annot)
        contour_ids,rectangles,contours = filter_image_original_contour_detection_and_rectangels(img1_warped,img_annot)
        cnts_id_remove,maskdict,myDict = get_mapping_dict_rectangle_contour_original(rectangles,img_annot,contour_ids,contours,file_list)
        src_img = get_final_image(contour_ids,cnts_id_remove,img_annot,contours,img1_warped)    

    # Display the image using Streamlit
        st.image(src_img, channels="BGR")
        # image_array = np.array(src_img)
        # print(image_array.shape)
        # st.download_button("Download image", data=image_array.tobytes(), file_name="image.jpg")
        # image = Image.open(uploaded_file)
        # st.image(image, caption=uploaded_file.name, use_column_width=True)