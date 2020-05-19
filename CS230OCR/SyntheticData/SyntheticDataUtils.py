import pandas as pd
import numpy as np

import cv2
import mnist


def load_mnist_data(path=None):
    if path is None:
        train_images_array = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
        train_labels_array = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
    else:
        train_images_array = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz", path)
        train_labels_array = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz", path)
        
    image_dict_array = [ 
        { 
            "image" : train_images_array[i],
            "label" : train_labels_array[i]
        } for i in range(len(train_images_array))
    ]
    return image_dict_array


def get_random(image_dict_array):

    random_index = int(np.floor(len(image_dict_array) * np.random.rand()))

    use_label = image_dict_array[random_index]["label"]
    use_image = image_dict_array[random_index]["image"]
    return use_label, use_image


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def get_num_zeros_by_row(image):
    num_rows = image.shape[0]
    results_array = []
    for num_row, row in enumerate( image ):
        rightmost_nonzero_index = max([ i if row[i] > 0 else 0 for i in range(len(row))])
        leftmost_nonzero_index = min([ i if row[i] > 0 else len(row) for i in range(len(row))])
        
        if rightmost_nonzero_index > 0:
            rightmost_nonzero_index += 1
        num_right_zeros = len(row) - rightmost_nonzero_index 
        
        num_left_zeros = leftmost_nonzero_index

        results_array.append({"row" : num_row, "left" : num_left_zeros, "right" : num_right_zeros})
    results_df = pd.DataFrame(results_array)
    results_df.index = results_df["row"]
    results_df.sort_index(ascending=True, inplace=True)
    return results_df[["left", "right"]]


def concat_images_remove_gap(left_image, right_image, gap_tolerance = 0):
    left_zeros_df = get_num_zeros_by_row(left_image)
    right_zeros_df = get_num_zeros_by_row(right_image)

    join_zeros_df = pd.concat( [ left_zeros_df[["right"]], right_zeros_df[["left"]]], axis=1)
    join_zeros_df["gap"] = join_zeros_df["right"] + join_zeros_df["left"]
    min_gap = min(join_zeros_df["gap"])
    #print("Min gap = ", min_gap)

    zeros_to_remove = min_gap - gap_tolerance

    #print("Zeros to remove = ", zeros_to_remove)
    
    rows_array = []
    for num_row in range(len(left_image)):
        left_image_right_side_zeros = int(join_zeros_df.loc[num_row]["right"])
        right_image_left_side_zeros = int(join_zeros_df.loc[num_row]["left"])
        total_num_zeros = left_image_right_side_zeros + right_image_left_side_zeros

        try:
            left_image_row_no_zeros = left_image[num_row][:-left_image_right_side_zeros]
        except:
            raise Exception(left_image_right_side_zeros, num_row, left_image[num_row])
        assert( len(left_image_row_no_zeros) + left_image_right_side_zeros == len(left_image[num_row]))
        
        right_image_row_no_zeros = right_image[num_row][right_image_left_side_zeros:]
        assert( len(right_image_row_no_zeros) + right_image_left_side_zeros == len(right_image[num_row]))
        
        assert len(left_image_row_no_zeros) + len(right_image_row_no_zeros) + total_num_zeros == len(left_image[num_row]) + len(right_image[num_row])

        num_zeros_to_add = total_num_zeros - zeros_to_remove
        concatenated_row = np.concatenate( [ left_image_row_no_zeros, np.zeros([num_zeros_to_add]), right_image_row_no_zeros ] )
        rows_array.append(concatenated_row)

    output_image = np.array(rows_array)
    return output_image


def randomly_pad_width(image, target_width):
    image_width = image.shape[1]
    assert target_width >= image_width, "Target width is less than image width!"
    diff_width = target_width - image_width

    left_num_zeros = int(np.floor(np.random.rand() * diff_width))
    right_num_zeros = diff_width - left_num_zeros

    new_image = np.array( [ np.concatenate( 
        [ np.zeros([left_num_zeros]), image[i], np.zeros([right_num_zeros]) ] 
    ) for i in range(image.shape[0]) ] )
    
    return new_image


def add_noise(image, p_noise = 1e-2):
    [ height, width ] = image.shape
    for i in range(height):
        for j in range(width):
            if np.random.rand() < p_noise:
                pixel_value = image[i][j]
                if pixel_value == 0: # if black, add random greyscale
                    new_value = int(np.floor( np.random.rand() * 256 ))
                else: # if not black, randomly darken
                    new_value = int(np.floor( np.random.rand() * pixel_value ))
                image[i][j] = new_value
    return image


def concat_images(image_array):
    return np.concatenate(image_array, axis=1)


def create_digit_string(image_dict_array, config_dict):
    num_images = len(image_dict_array)

    total_width = config_dict.get("Total width")
    max_num_digits = config_dict.get("Max num digits")
    max_rotation_degrees = config_dict.get("Max rotation degrees")
    p_gaussian_noise = config_dict.get("P Gaussian noise")

    num_digits = int(1 + np.floor(np.random.rand() * max_num_digits))
    #print(" ".join(["Using", str(num_digits), "digits"]))

    concat_image_array = []
    concat_digit_array = []

    for i in range(num_digits):
        # grab a random image
        use_label, use_image = get_random(image_dict_array)
        random_index = int(np.floor(num_images * np.random.rand()))

        use_label = image_dict_array[random_index]["label"]
        use_image = image_dict_array[random_index]["image"]

        # apply a random rotation
        use_angle = max_rotation_degrees * (np.random.rand() - 0.5)
        modified_image = rotate_image(use_image, use_angle)

        concat_image_array.append(modified_image)
        concat_digit_array.append(use_label)

    # join images laterally, removing gaps
    concatenated_image = concat_image_array[0]
    for right_image in concat_image_array[1:]:
        concatenated_image = concat_images_remove_gap(concatenated_image, right_image)

    # now add a random number from above
    __, random_image = get_random(image_dict_array)

    random_image = randomly_pad_width( random_image, concatenated_image.shape[1] )
    test_image = ( concat_images_remove_gap( random_image.T, concatenated_image.T ) ).T
    concatenated_image = test_image[test_image.shape[0] - concatenated_image.shape[0]:]

    # now add a random number from below
    __, random_image = get_random(image_dict_array)

    random_image = randomly_pad_width( random_image, concatenated_image.shape[1] )
    test_image = ( concat_images_remove_gap( concatenated_image.T, random_image.T ) ).T
    concatenated_image = test_image[:concatenated_image.shape[0]]

    # now pad the width
    concatenated_image = randomly_pad_width(concatenated_image, total_width)

    concatenated_image = add_noise(concatenated_image, p_gaussian_noise)
    concatenated_label = "".join([str(x) for x in concat_digit_array])
    
    return {"Label" : concatenated_label, "Image" : concatenated_image, "Metadata" : {}}

