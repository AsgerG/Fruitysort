import os
import cv2
import matplotlib.pyplot as plt
import json

with open("config.json") as json_data_file:
    config = json.load(json_data_file)

cropped = True
data_path = config['files']['folder_path'] + 'data/'

# Load data
filenames = {}
for folder in os.listdir(data_path + "generated_data/"):
    filenames[folder] = []
    for file in os.listdir(data_path + f"generated_data/{folder}"):
        filenames[folder].append(file)

investigate_these = []
for label, imgs in filenames.items():
    for imgname in imgs:
        if "boxed" in imgname:
            continue
        print(imgname)
        img = cv2.imread(data_path + f"generated_data/{label}/{imgname}", cv2.IMREAD_COLOR)
        img_h, img_w, channels = img.shape

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        numlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresh, 4, cv2.CV_32S
        )

        # iterate over components
        best_match, best_dist = None, 1000000
        for i in range(numlabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = centroids[i]

            # reject if too small
            if h < img_h * 0.1 or w < img_w * 0.1:
                continue

            # reject if too large
            if h > img_h * 0.9 or w > img_w * 0.9:
                continue

            # pick component that is close to center
            dist = (img_h / 2 - cx) ** 2 + (img_w / 2 - cy) ** 2
            if dist < best_dist:
                best_match = i
                best_dist = dist

        if best_match is not None:
            x = stats[best_match, cv2.CC_STAT_LEFT]
            y = stats[best_match, cv2.CC_STAT_TOP]
            w = stats[best_match, cv2.CC_STAT_WIDTH]
            h = stats[best_match, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[best_match]
            output = img.copy()
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.circle(output, (int(cx), int(cy)), 4, (0, 0, 255), -1)

            plt.imshow(output, cmap="gray")
            plt.plot(105,200,'ro') 
            plt.show()
            print(f"x:{x}, y:{y}, w:{w}, h:{h}")


            from pathlib import Path
            if not Path(data_path + f"generated_data_largest_component/{label}/").exists():
                os.mkdir((data_path + f"generated_data_largest_component/{label}/"))

            extension = f"{imgname.split('.')[0]}_boxed.{imgname.split('.')[1]}"
            plt.savefig(data_path + f"generated_data_largest_component/{label}/{extension}")
        else:
            print(data_path + f"did not find a good match for generated_data/{label}/{imgname}")
            investigate_these.append(img)