import os

def prune():
    # Prune results from test set
    for index in range(1, 100):
        ipp_hd_image_path = f"images_1920x1080/image_{index}_ipp.jpg"
        npp_v1_hd_image_path = f"images_1920x1080/image_{index}_npp_v1.jpg"
        npp_v2_hd_image_path = f"images_1920x1080/image_{index}_npp_v2.jpg"

        ipp_4k_image_path = f"images_3840x2160/image_{index}_ipp.jpg"
        npp_v1_4k_image_path = f"images_3840x2160/image_{index}_npp_v1.jpg"
        npp_v2_4k_image_path = f"images_3840x2160/image_{index}_npp_v2.jpg"

        os.remove(ipp_hd_image_path)
        os.remove(npp_v1_hd_image_path)
        os.remove(npp_v2_hd_image_path)
        os.remove(ipp_4k_image_path)
        os.remove(npp_v1_4k_image_path)
        os.remove(npp_v2_4k_image_path)

if __name__ == "__main__":
    prune()
