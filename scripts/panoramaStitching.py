import cv2
import numpy as np
import matplotlib.pyplot as plt

class PanoramaStitcher:
    def __init__(self, image_paths):
        """
        Initializes the panorama stitcher with image paths.
        """
        self.image_paths = image_paths
        self.images = []
        self.keypoints_list = []
        self.descriptors_list = []
        self.orb = cv2.ORB_create()

    def load_images(self):
        """
        Loads images from file paths and converts them to grayscale.
        """
        for path in self.image_paths:
            image = cv2.imread(path)
            if image is None:
                print(f"⚠️ Warning: Unable to load image {path}")
                continue
            self.images.append(image)

    def detect_keypoints(self):
        """
        Detects ORB keypoints and descriptors for each image.
        """
        for image in self.images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)

            if descriptors is None:
                print("⚠️ Warning: No descriptors found in one image, skipping...")
                continue

            self.keypoints_list.append(keypoints)
            self.descriptors_list.append(descriptors)

    def display_keypoints(self):
        """
        Displays images with detected keypoints.
        """
        fig, axes = plt.subplots(1, len(self.images), figsize=(15, 5))
        for ax, img, kps, path in zip(axes, self.images, self.keypoints_list, self.image_paths):
            img_with_keypoints = cv2.drawKeypoints(img, kps, None, color=(0, 255, 0), flags=0)
            ax.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
            ax.set_title(path.split('/')[-1])
            ax.axis("off")
        plt.show()

    def match_keypoints(self):
        """
        Matches keypoints between consecutive images.
        Returns: List of matches between images.
        """
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches_list = []

        for i in range(1, len(self.descriptors_list)):
            matches = matcher.match(self.descriptors_list[i - 1], self.descriptors_list[i])
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) < 10:  # If too few matches, warn the user
                print(f"⚠️ Warning: Poor match quality between images {i-1} and {i}")

            matches_list.append(matches)
        return matches_list

    def compute_homographies(self, matches_list):
        """
        Computes homography matrices between matched images.
        Returns: List of homography matrices.
        """
        homographies = []

        for i, matches in enumerate(matches_list):
            src_pts = np.float32([self.keypoints_list[i][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.keypoints_list[i + 1][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 4.0)  # Adjusted threshold for better accuracy

            if H is None:
                print(f"⚠️ Warning: Homography computation failed between images {i} and {i+1}")
                continue

            homographies.append(H)

        return homographies

    def crop_black_borders(self, image):
        """
        Crops out black borders from the final stitched image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            return image[y:y+h, x:x+w]

        return image  # If no contours are found, return as is.

    def stitch_images(self):
        """
        Stitches images using ORB keypoint matching and homography transformations.
        Returns: Stitched panorama image.
        """
        matches_list = self.match_keypoints()
        homographies = self.compute_homographies(matches_list)

        if not homographies:
            print("❌ Error: Could not compute homographies. Exiting.")
            return None

        # Initialize panorama with the first image
        panorama = self.images[0]

        for i in range(1, len(self.images)):
            H = homographies[i - 1]

            # Get size for the new warped image
            h1, w1 = panorama.shape[:2]
            h2, w2 = self.images[i].shape[:2]

            # Warp the next image onto the panorama
            warped_img = cv2.warpPerspective(self.images[i], H, (w1 + w2, h1))
            warped_img[0:h1, 0:w1] = panorama  # Merge images
            panorama = warped_img

        # Crop unnecessary black areas
        panorama = self.crop_black_borders(panorama)
        return panorama

    def run(self):
        """
        Runs the full panorama stitching process.
        """
        self.load_images()

        if len(self.images) < 2:
            print("❌ Error: Need at least two images for stitching.")
            return

        self.detect_keypoints()
        self.display_keypoints()

        panorama = self.stitch_images()

        if panorama is not None:
            plt.figure(figsize=(15, 5))
            plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title("Stitched Panorama")
            plt.show()

            cv2.imwrite("stitched_panorama.jpg", panorama)
            print("✅ Panorama saved as 'stitched_panorama.jpg'.")


if __name__ == "__main__":
    # Define image paths
    image_paths = [
        "../input_images/panorama_left.jpg",
        "../input_images/panorama_centre.jpg",
        "../input_images/panorama_right.jpg"
    ]

    # Run the panorama stitcher
    stitcher = PanoramaStitcher(image_paths)
    stitcher.run()
