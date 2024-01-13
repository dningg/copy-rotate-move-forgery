import cv2
import numpy as np
import mahotas
from scipy.spatial import distance as dist

def compare_moments(m1, m2):
    return np.linalg.norm(m1 - m2)

def compare_blocks_distance(p, q, num_blocks_row):
    i, j = p // num_blocks_row, p % num_blocks_row
    k, l = q // num_blocks_row, q % num_blocks_row
    return (i - k) ** 2 + (j - l) ** 2

def detect_copy_move(image_blocks, threshold_D1, threshold_D2, order, num_blocks_row):
    forged_blocks = []
    num_blocks = len(image_blocks)

    for p in range(num_blocks):
        for q in range(p + 1, num_blocks):
            distance = compare_blocks_distance(p, q, num_blocks_row)
            if distance > threshold_D2:
                continue

            features_p = mahotas.features.zernike_moments(image_blocks[p], 0.5, degree=order)
            features_q = mahotas.features.zernike_moments(image_blocks[q], 0.5, degree=order)

            similarity = compare_moments(features_p, features_q)
            if similarity < threshold_D1:
                forged_blocks.append((p, q))

    return forged_blocks

def visualize_detection(image, forged_blocks, block_size):
    result_image = np.zeros_like(image, dtype=np.uint8)

    for (p, q) in forged_blocks:
        i, j = p // (image.shape[1] // block_size), p % (image.shape[1] // block_size)
        x1, y1 = j * block_size, i * block_size
        x2, y2 = (j + 1) * block_size, (i + 1) * block_size
        result_image[y1:y2, x1:x2] = 255

    return result_image

def main():
    # Load ảnh nghi ngờ
    suspicious_image = cv2.imread('japan_tower_gcs500_copy_rb5.png', cv2.IMREAD_GRAYSCALE)

    # Thiết lập các tham số
    block_size = 24
    overlap = 4
    order = 5
    threshold_D1 = 300
    threshold_D2 = 50

    # Chia hình ảnh thành các khối
    blocks = [suspicious_image[i:i+block_size, j:j+block_size] for i in range(0, suspicious_image.shape[0] - block_size, overlap)
              for j in range(0, suspicious_image.shape[1] - block_size, overlap)]

    # Gọi hàm detect_copy_move và nhận danh sách forged_blocks
    num_blocks_row = suspicious_image.shape[1] // block_size
    forged_blocks = detect_copy_move(blocks, threshold_D1, threshold_D2, order, num_blocks_row)

    # Hiển thị hình ảnh với các khối bị trùng lặp được tô màu
    result_image = visualize_detection(suspicious_image, forged_blocks, block_size)

    # Hiển thị hình ảnh gốc và kết quả
    cv2.imshow('Original Image', suspicious_image)
    cv2.imshow('Detected Blocks', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
