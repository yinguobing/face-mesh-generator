"""A sample module of loading the TFLite model of FaceMesh from Google."""
import cv2
import tensorflow as tf
import numpy as np


class MeshDetector(object):
    """Face mesh detector"""

    def __init__(self, model_path):
        """Initialization"""
        # Initialize the input image holder.
        self.target_image = None

        # Load the model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)

        # Set model input
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def _preprocess(self, image_bgr):
        """Preprocess the image to meet the model's input requirement.
        Args:
            image_bgr: An image in default BGR format.

        Returns:
            image_norm: The normalized image ready to be feeded.
        """
        image_resized = cv2.resize(image_bgr, (192, 192))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_norm = (image_rgb-127.5)/127.5
        return image_norm

    def get_mesh(self, image):
        """Detect the face mesh from the image given.
        Args:
            image: An image in default BGR format.

        Returns:
            mesh: A face mesh, normalized.
            score: Confidence score.
        """
        # Preprocess the image before sending to the network.
        image = self._preprocess(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = image[tf.newaxis, :]

        # The actual detection.
        self.interpreter.set_tensor(self.input_details[0]["index"], image)
        self.interpreter.invoke()

        # Save the results.
        mesh = self.interpreter.get_tensor(self.output_details[0]["index"])[
            0].reshape(468, 3) / 192
        score = self.interpreter.get_tensor(self.output_details[1]["index"])[0]

        return mesh, score

    def draw_mesh(self, image, mesh, mark_size=2, line_width=1):
        """Draw the mesh on an image"""
        # The mesh are normalized which means we need to convert it back to fit
        # the image size.
        image_size = image.shape[0]
        mesh = mesh * image_size
        for point in mesh:
            cv2.circle(image, (point[0], point[1]),
                       mark_size, (0, 255, 128), -1)

        # Draw the contours.
        # Eyes
        left_eye_contour = np.array([mesh[33][0:2],
                                     mesh[7][0:2],
                                     mesh[163][0:2],
                                     mesh[144][0:2],
                                     mesh[145][0:2],
                                     mesh[153][0:2],
                                     mesh[154][0:2],
                                     mesh[155][0:2],
                                     mesh[133][0:2],
                                     mesh[173][0:2],
                                     mesh[157][0:2],
                                     mesh[158][0:2],
                                     mesh[159][0:2],
                                     mesh[160][0:2],
                                     mesh[161][0:2],
                                     mesh[246][0:2], ]).astype(np.int32)
        right_eye_contour = np.array([mesh[263][0:2],
                                      mesh[249][0:2],
                                      mesh[390][0:2],
                                      mesh[373][0:2],
                                      mesh[374][0:2],
                                      mesh[380][0:2],
                                      mesh[381][0:2],
                                      mesh[382][0:2],
                                      mesh[362][0:2],
                                      mesh[398][0:2],
                                      mesh[384][0:2],
                                      mesh[385][0:2],
                                      mesh[386][0:2],
                                      mesh[387][0:2],
                                      mesh[388][0:2],
                                      mesh[466][0:2]]).astype(np.int32)
        # Lips
        cv2.polylines(image, [left_eye_contour, right_eye_contour], False,
                      (255, 255, 255), line_width, cv2.LINE_AA)
